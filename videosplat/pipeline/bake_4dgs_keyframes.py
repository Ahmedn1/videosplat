#!/usr/bin/env python3
"""
Standalone keyframe baking script for 4DGaussians models.

Run with 4DGaussians root on PYTHONPATH:
  PYTHONPATH=/path/to/4DGaussians python bake_4dgs_keyframes.py \
      --model_path /path/to/model \
      --source_path /path/to/data \
      --n_keyframes 50 \
      --output_dir /path/to/model/keyframes

Outputs one PLY per keyframe in standard 3DGS format (f_dc + f_rest SH3),
compatible with the mkkellogg browser viewer.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData, PlyElement


# ── Dummy camera with just a .time attribute ─────────────────────────────────

class _TimestampCamera:
    def __init__(self, t: float):
        self.time = t


# ── PLY helpers ───────────────────────────────────────────────────────────────

def _construct_attrs(n_dc: int, n_rest: int, n_scale: int, n_rot: int) -> list[str]:
    l = ["x", "y", "z", "nx", "ny", "nz"]
    for i in range(n_dc):
        l.append(f"f_dc_{i}")
    for i in range(n_rest):
        l.append(f"f_rest_{i}")
    l.append("opacity")
    for i in range(n_scale):
        l.append(f"scale_{i}")
    for i in range(n_rot):
        l.append(f"rot_{i}")
    return l


def _save_ply(path: Path, xyz, f_dc, f_rest, opacity, scale, rotation) -> None:
    n = xyz.shape[0]
    normals = np.zeros_like(xyz)

    n_dc   = f_dc.shape[1]    # should be 3 (DC × RGB)
    n_rest = f_rest.shape[1]  # 45 for SH3
    n_scl  = scale.shape[1]   # 3
    n_rot  = rotation.shape[1]  # 4

    attrs = _construct_attrs(n_dc, n_rest, n_scl, n_rot)
    dtype = [(a, "f4") for a in attrs]
    el = np.empty(n, dtype=dtype)

    el["x"], el["y"], el["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    el["nx"] = el["ny"] = el["nz"] = 0.0

    for i in range(n_dc):
        el[f"f_dc_{i}"] = f_dc[:, i]
    for i in range(n_rest):
        el[f"f_rest_{i}"] = f_rest[:, i]

    el["opacity"] = opacity[:, 0]

    for i in range(n_scl):
        el[f"scale_{i}"] = scale[:, i]
    for i in range(n_rot):
        el[f"rot_{i}"] = rotation[:, i]

    PlyData([PlyElement.describe(el, "vertex")]).write(str(path))


# ── Quality stats ─────────────────────────────────────────────────────────────

def _quality_stats(xyz, opacity_logit, scale_log, timestamp: float) -> dict:
    opacity = torch.sigmoid(opacity_logit).detach().cpu().numpy().ravel()
    scales  = torch.exp(scale_log).detach().cpu().numpy()
    s_max   = scales.max(axis=1)
    s_min   = np.maximum(scales.min(axis=1), 1e-8)
    ratio   = s_max / s_min
    return {
        "gaussians":       int(len(opacity)),
        "opacity_p50":     float(np.percentile(opacity, 50)),
        "scale_ratio_p95": float(np.percentile(ratio, 95)),
        "scale_ratio_p99": float(np.percentile(ratio, 99)),
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",  required=True)
    parser.add_argument("--source_path", required=True)
    parser.add_argument("--n_keyframes", type=int, default=24)
    parser.add_argument("--output_dir",  required=True)
    parser.add_argument("--iteration",   type=int, default=-1)
    args = parser.parse_args()

    model_path  = Path(args.model_path)
    output_dir  = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Find iteration checkpoint ──────────────────────────────────────────────
    pc_root = model_path / "point_cloud"
    if args.iteration == -1:
        iters = sorted(
            int(d.name.split("_")[1]) for d in pc_root.iterdir() if d.name.startswith("iteration_")
        )
        iteration = iters[-1]
    else:
        iteration = args.iteration

    ply_path   = pc_root / f"iteration_{iteration}" / "point_cloud.ply"
    deform_dir = pc_root / f"iteration_{iteration}"
    print(f"Loading checkpoint iteration {iteration} from {ply_path}")

    # ── Load model ────────────────────────────────────────────────────────────
    # Pull defaults directly from 4DGaussians argument classes so we never
    # miss a new attribute (like grid_pe) when the codebase changes.
    from arguments import ModelHiddenParams
    from argparse import ArgumentParser

    _parser = ArgumentParser()
    hp_obj  = ModelHiddenParams(_parser)
    hp_defaults = _parser.parse_args([])  # all defaults, no extra flags

    from scene.gaussian_model import GaussianModel
    gaussians = GaussianModel(3, hp_defaults)
    gaussians.load_ply(str(ply_path))
    gaussians.load_model(str(deform_dir))
    gaussians._deformation.eval()

    # ── Collect static attributes that don't deform ───────────────────────────
    from utils.render_utils import get_state_at_time

    timestamps = [i / max(args.n_keyframes - 1, 1) for i in range(args.n_keyframes)]
    all_stats  = []

    print(f"Baking {args.n_keyframes} keyframes…")
    for i, t in enumerate(timestamps):
        cam = _TimestampCamera(t)
        with torch.no_grad():
            xyz_t, scale_t, rot_t, opac_t, shs_t = get_state_at_time(gaussians, cam)

        xyz_np   = xyz_t.detach().cpu().numpy()
        rot_np   = rot_t.detach().cpu().numpy()
        opac_np  = opac_t.detach().cpu().numpy()

        # Clamp needle Gaussians: cap scale ratio at 300:1 in log-space.
        # scale_t is log-scale (N,3); ratio = exp(max-min) ≤ 300 → max-min ≤ log(300).
        import math
        _s_min = scale_t.min(dim=1, keepdim=True).values
        scale_t = torch.min(scale_t, _s_min + math.log(300))
        scale_np = scale_t.detach().cpu().numpy()

        # shs_t shape: (N, total_sh_coeffs, 3)
        n_dc   = gaussians._features_dc.shape[1]   # 1 for DC
        n_feat = shs_t.shape[1]
        n_rest = n_feat - n_dc

        f_dc_np   = shs_t[:, :n_dc, :].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest_np = shs_t[:, n_dc:,  :].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()

        out_path = output_dir / f"keyframe_{i:04d}.ply"
        _save_ply(out_path, xyz_np, f_dc_np, f_rest_np, opac_np, scale_np, rot_np)

        stats = _quality_stats(xyz_t, opac_t, scale_t, t)
        all_stats.append({"frame": i, "timestamp": round(t, 4), **stats})

        if i == 0 or i == args.n_keyframes - 1:
            s = stats
            print(
                f"  frame {i:3d} (t={t:.3f}): {s['gaussians']:,} Gaussians | "
                f"opacity_p50={s['opacity_p50']:.3f} | "
                f"scale_ratio_p99={s['scale_ratio_p99']:.0f}"
            )

    (output_dir / "quality_stats.json").write_text(json.dumps(all_stats, indent=2))
    print(f"Done — {len(all_stats)} keyframes written to {output_dir}")


if __name__ == "__main__":
    main()
