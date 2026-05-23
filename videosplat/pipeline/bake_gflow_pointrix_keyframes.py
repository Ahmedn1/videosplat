#!/usr/bin/env python3
"""
Bake Pointrix Gaussian-Flow model → N keyframe PLY snapshots.

Loads the .pth checkpoint saved by the Pointrix DefaultTrainer, reconstructs
the GaussianFlowPointCloud, and evaluates the temporal flow model at evenly-
spaced timestamps to produce standard 3DGS PLY keyframes.

Usage:
  python bake_gflow_pointrix_keyframes.py \
      --checkpoint  /path/to/chkpntXXXX.pth \
      --config      /path/to/gflow_config.yaml \
      --gflow_dir   /path/to/Gaussian-Flow \
      --n_keyframes 50 \
      --output_dir  /path/to/keyframes
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch


# ── PLY writer (same format as bake_gflow_keyframes.py) ──────────────────────

def _write_ply(path: Path, xyz, f_dc, f_rest, opacity_logit, log_scale, rotation):
    n = len(xyz)
    sh_rest_cols = [(f"f_rest_{i}", "f4") for i in range(45)]
    dt = np.dtype(
        [("x", "f4"), ("y", "f4"), ("z", "f4"),
         ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
         ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4")]
        + sh_rest_cols
        + [("opacity", "f4"),
           ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
           ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4")]
    )
    v = np.zeros(n, dtype=dt)
    v["x"], v["y"], v["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    v["f_dc_0"], v["f_dc_1"], v["f_dc_2"] = f_dc[:, 0], f_dc[:, 1], f_dc[:, 2]
    for i in range(min(45, f_rest.shape[1])):
        v[f"f_rest_{i}"] = f_rest[:, i]
    v["opacity"] = opacity_logit
    v["scale_0"], v["scale_1"], v["scale_2"] = log_scale[:, 0], log_scale[:, 1], log_scale[:, 2]
    v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"] = (
        rotation[:, 0], rotation[:, 1], rotation[:, 2], rotation[:, 3]
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


# ── Model loading ──────────────────────────────────────────────────────────────

def load_gflow_pointrix(gflow_dir: str, config_path: str, checkpoint_path: str, device: str):
    """
    Create and load a GaussianFlowPointCloud from a Pointrix checkpoint.

    Returns the point_cloud object with all trained parameters.
    """
    sys.path.insert(0, gflow_dir)

    from omegaconf import OmegaConf
    from pointrix.utils.config import load_config
    from model.point import GaussianFlowPointCloud

    cfg = load_config(config_path)
    pc_cfg = cfg.trainer.model.point_cloud

    # Read num_pts from checkpoint first so we can construct the point cloud
    # at the right size — re_init() doesn't re-register GFlow-specific
    # attributes (pos_params, rot_params, feat_params, time_center).
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = dict(ckpt["model"])
    num_pts = state.pop("num_pts", None)
    if num_pts is None:
        # Fall back to inferring from a known parameter
        for k, v in state.items():
            if k.endswith("point_cloud.position") or k == "position":
                num_pts = int(v.shape[0])
                break
    if num_pts is None:
        num_pts = 8

    # Override initializer to random with the right size
    pc_cfg_dict = OmegaConf.to_container(pc_cfg, resolve=True)
    pc_cfg_dict["initializer"] = {
        "init_type": "random",
        "num_points": int(num_pts),
        "radius": 0.5,
        "feat_dim": 3,
    }
    pc_cfg_merged = OmegaConf.create(pc_cfg_dict)

    pc = GaussianFlowPointCloud(pc_cfg_merged)

    # Strip renderer and non-point-cloud keys; load only point_cloud params
    pc_state = {}
    for k, v in state.items():
        if k.startswith("point_cloud."):
            pc_state[k[len("point_cloud."):]] = v
    if not pc_state:
        # Fallback: try loading all non-renderer keys directly
        for k, v in state.items():
            if not k.startswith("renderer") and k not in ("sh_degree",):
                pc_state[k] = v

    missing, unexpected = pc.load_state_dict(pc_state, strict=False)
    if missing:
        print(f"  [warn] missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")

    return pc.to(device).eval()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",  required=True, help="path to chkpntXXXXX.pth")
    parser.add_argument("--config",      required=True, help="path to gflow_config.yaml")
    parser.add_argument("--gflow_dir",   required=True, help="path to Gaussian-Flow repo root")
    parser.add_argument("--n_keyframes", type=int, default=50)
    parser.add_argument("--output_dir",  required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # polyfourier (used by GaussianFlow's flow models) uses Taichi kernels
    # that must be initialised before first use.
    import taichi as ti
    ti.init(arch=ti.cuda if device == "cuda" else ti.cpu)

    print(f"Loading Pointrix Gaussian-Flow checkpoint: {Path(args.checkpoint).name}")

    pc = load_gflow_pointrix(args.gflow_dir, args.config, args.checkpoint, device)
    print(f"  {len(pc):,} Gaussians loaded")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamps = [i / max(args.n_keyframes - 1, 1) for i in range(args.n_keyframes)]

    with torch.no_grad():
        for i, t in enumerate(timestamps):
            pc.set_timestep(float(t))

            xyz  = pc.get_position_flow[0].cpu().numpy().astype(np.float32)   # [N, 3]
            rot  = pc.get_rotation_flow[0].cpu().numpy().astype(np.float32)   # [N, 4]
            shs  = pc.get_shs_flow[0].cpu().numpy().astype(np.float32)        # [N, ≥1, 3]
            opac = pc.opacity.cpu().numpy().astype(np.float32).ravel()         # [N]
            scal = pc.scaling.cpu().numpy().astype(np.float32)                 # [N, 3] log-space

            f_dc   = shs[:, 0, :]                                # [N, 3]
            f_rest = shs[:, 1:, :].reshape(len(shs), -1)         # [N, K*3]
            if f_rest.shape[1] < 45:
                pad    = np.zeros((len(f_rest), 45 - f_rest.shape[1]), dtype=np.float32)
                f_rest = np.concatenate([f_rest, pad], axis=1)

            out_path = out_dir / f"keyframe_{i:04d}.ply"
            _write_ply(out_path, xyz, f_dc, f_rest, opac, scal, rot)

            if i % 10 == 0 or i == args.n_keyframes - 1:
                print(f"  [{i + 1:3d}/{args.n_keyframes}] t={t:.3f} → {out_path.name}")

    print(f"Done — {args.n_keyframes} keyframes → {out_dir}")


if __name__ == "__main__":
    main()
