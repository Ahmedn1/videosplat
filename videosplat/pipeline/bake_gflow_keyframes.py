#!/usr/bin/env python3
"""
Bake Gaussian-Flow model → N keyframe PLY snapshots.

Gaussian-Flow uses a deformation network similar to 4DGaussians.
This script mirrors bake_4dgs_keyframes.py but loads Gaussian-Flow's
GaussianModel class.

Run with Gaussian-Flow root on PYTHONPATH:
  python bake_gflow_keyframes.py \
      --model_path  /path/to/model \
      --source_path /path/to/source \
      --n_keyframes 50 \
      --output_dir  /path/to/keyframes
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch


# ── PLY helpers (identical to bake_4dgs_keyframes.py) ────────────────────────

_SH3_N_COEFFS = 16   # (degree+1)^2 for SH degree 3
_SH3_REST     = _SH3_N_COEFFS - 1   # 15 rest per channel → 45 total


def _write_ply(path: Path, xyz, f_dc, f_rest, opacity_logit, log_scale, rotation):
    n = len(xyz)
    sh_rest_cols = [(f"f_rest_{i}", "f4") for i in range(45)]
    dt = np.dtype(
        [("x","f4"),("y","f4"),("z","f4"),
         ("nx","f4"),("ny","f4"),("nz","f4"),
         ("f_dc_0","f4"),("f_dc_1","f4"),("f_dc_2","f4")]
        + sh_rest_cols
        + [("opacity","f4"),
           ("scale_0","f4"),("scale_1","f4"),("scale_2","f4"),
           ("rot_0","f4"),("rot_1","f4"),("rot_2","f4"),("rot_3","f4")]
    )
    v = np.zeros(n, dtype=dt)
    v["x"], v["y"], v["z"] = xyz[:,0], xyz[:,1], xyz[:,2]
    v["f_dc_0"], v["f_dc_1"], v["f_dc_2"] = f_dc[:,0], f_dc[:,1], f_dc[:,2]
    for i in range(min(45, f_rest.shape[1])):
        v[f"f_rest_{i}"] = f_rest[:, i]
    v["opacity"] = opacity_logit
    v["scale_0"], v["scale_1"], v["scale_2"] = log_scale[:,0], log_scale[:,1], log_scale[:,2]
    v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"] = (
        rotation[:,0], rotation[:,1], rotation[:,2], rotation[:,3]
    )

    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        + "".join(f"property float {name}\n" for name in dt.names)
        + "end_header\n"
    )
    with open(path, "wb") as f:
        f.write(header.encode())
        f.write(v.tobytes())


def _cap_log_scales(log_scale: np.ndarray, max_ratio: float = 30.0) -> np.ndarray:
    s_min = log_scale.min(axis=1, keepdims=True)
    cap   = s_min + np.log(max_ratio)
    return np.minimum(log_scale, cap)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",  required=True)
    parser.add_argument("--source_path", required=True)
    parser.add_argument("--n_keyframes", type=int, default=50)
    parser.add_argument("--output_dir",  required=True)
    parser.add_argument("--iteration",   type=int, default=-1)
    args = parser.parse_args()

    model_path  = Path(args.model_path)
    out_dir     = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Find checkpoint ───────────────────────────────────────────────────────
    pc_root = model_path / "point_cloud"
    if args.iteration == -1:
        iteration = max(
            int(d.name.split("_")[1])
            for d in pc_root.iterdir()
            if d.name.startswith("iteration_")
        )
    else:
        iteration = args.iteration

    ply_path   = pc_root / f"iteration_{iteration}" / "point_cloud.ply"
    deform_dir = pc_root / f"iteration_{iteration}"
    print(f"Gaussian-Flow checkpoint: iteration {iteration}")

    # ── Load model ────────────────────────────────────────────────────────────
    try:
        from arguments import ModelHiddenParams, PipelineParams
        from argparse import ArgumentParser as AP
        _p = AP()
        ModelHiddenParams(_p)
        PipelineParams(_p)
        hp = _p.parse_args([])

        from scene.gaussian_model import GaussianModel
        gaussians = GaussianModel(3, hp)
        gaussians.load_ply(str(ply_path))
        gaussians.load_model(str(deform_dir))
        gaussians._deformation.eval()
        print(f"Loaded {gaussians.get_xyz.shape[0]:,} Gaussians")
    except Exception as e:
        print(f"Error loading Gaussian-Flow model: {e}", file=sys.stderr)
        sys.exit(1)

    # ── Bake keyframes ────────────────────────────────────────────────────────

    class _TimestampCamera:
        def __init__(self, t: float):
            self.time = t

    timestamps = [i / max(args.n_keyframes - 1, 1) for i in range(args.n_keyframes)]

    for i, t in enumerate(timestamps):
        cam = _TimestampCamera(t)

        with torch.no_grad():
            # Gaussian-Flow's get_state_at_time mirrors 4DGaussians' interface
            if hasattr(gaussians, "get_state_at_time"):
                xyz, scale, rot, opac, shs = gaussians.get_state_at_time(cam)
            else:
                # Fall back to static attributes
                xyz   = gaussians.get_xyz
                scale = gaussians.get_scaling
                rot   = gaussians.get_rotation
                opac  = gaussians.get_opacity
                shs   = gaussians.get_features

            xyz_np   = xyz.detach().cpu().numpy().astype(np.float32)
            scale_np = scale.detach().cpu().numpy().astype(np.float32)
            rot_np   = rot.detach().cpu().numpy().astype(np.float32)
            opac_np  = opac.detach().cpu().numpy().astype(np.float32).ravel()
            shs_np   = shs.detach().cpu().numpy().astype(np.float32)

        log_scale = np.log(np.maximum(scale_np, 1e-6))
        log_scale = _cap_log_scales(log_scale)

        if shs_np.ndim == 3:
            f_dc   = shs_np[:, 0, :]            # (N, 3)
            f_rest = shs_np[:, 1:, :].reshape(len(shs_np), -1)   # (N, 45)
        else:
            f_dc   = shs_np[:, :3]
            f_rest = shs_np[:, 3:]

        if f_rest.shape[1] < 45:
            pad    = np.zeros((len(f_rest), 45 - f_rest.shape[1]), dtype=np.float32)
            f_rest = np.concatenate([f_rest, pad], axis=1)

        out_path = out_dir / f"keyframe_{i:04d}.ply"
        _write_ply(out_path, xyz_np, f_dc, f_rest, opac_np, log_scale, rot_np)

        if i % 10 == 0 or i == args.n_keyframes - 1:
            print(f"  [{i+1:3d}/{args.n_keyframes}] t={t:.3f} → {out_path.name}")

    print(f"Done — {args.n_keyframes} keyframes → {out_dir}")


if __name__ == "__main__":
    main()
