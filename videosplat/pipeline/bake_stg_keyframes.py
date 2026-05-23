#!/usr/bin/env python3
"""
Bake SpacetimeGaussians model → N keyframe PLY snapshots.

Reads the trained PLY (which contains trbf_center, trbf_scale, motion_0..8, omega_0..3)
and evaluates the temporal model at evenly-spaced timestamps in numpy — no STG Python
API needed.

Temporal model (from STG renderer/__init__.py train_ours_lite):
  tforpoly   = t - trbf_center
  xyz_t      = xyz + motion[0:3]*tforpoly + motion[3:6]*tforpoly² + motion[6:9]*tforpoly³
  trbf_dist  = tforpoly / exp(trbf_scale)
  trbf_w     = exp(-trbf_dist²)           # Gaussian temporal visibility weight
  opacity_t  = inv_sigmoid(sigmoid(opacity_raw) * trbf_w)
  rotation_t = normalize(rot + tforpoly * omega)
  features   = f_dc  (no temporal modulation in ours_lite)
  scales     = scale_0..2  (no temporal scaling)

Run with SpacetimeGaussians root on PYTHONPATH (env inherited from parent process):
  python bake_stg_keyframes.py \
      --model_path /path/to/model \
      --n_keyframes 50 \
      --output_dir  /path/to/keyframes
"""

import argparse
from pathlib import Path

import numpy as np


# ── PLY writer ────────────────────────────────────────────────────────────────

def _build_ply_dtype():
    sh_coeffs = [(f"f_dc_{i}", "f4") for i in range(3)]
    sh_coeffs += [(f"f_rest_{i}", "f4") for i in range(45)]
    return np.dtype(
        [("x", "f4"), ("y", "f4"), ("z", "f4"),
         ("nx", "f4"), ("ny", "f4"), ("nz", "f4")]
        + sh_coeffs
        + [("opacity", "f4"),
           ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
           ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4")]
    )


def _write_ply(path: Path, xyz, f_dc, f_rest, opacity, scales, rotations):
    n  = len(xyz)
    dt = _build_ply_dtype()
    v  = np.zeros(n, dtype=dt)
    v["x"], v["y"], v["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    v["nx"] = v["ny"] = v["nz"] = 0.0
    for i in range(3):
        v[f"f_dc_{i}"] = f_dc[:, i]
    for i in range(45):
        v[f"f_rest_{i}"] = f_rest[:, i] if f_rest.shape[1] > i else 0.0
    v["opacity"] = opacity
    v["scale_0"], v["scale_1"], v["scale_2"] = scales[:, 0], scales[:, 1], scales[:, 2]
    v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"] = (
        rotations[:, 0], rotations[:, 1], rotations[:, 2], rotations[:, 3]
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


# ── Temporal evaluation ───────────────────────────────────────────────────────

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))

def _inv_sigmoid(x):
    x = np.clip(x, 1e-6, 1 - 1e-6)
    return np.log(x / (1 - x))


def bake_from_ply(ply_path: Path, n_keyframes: int, out_dir: Path):
    try:
        from plyfile import PlyData
    except ImportError:
        raise ImportError("plyfile is required: pip install plyfile")

    ply = PlyData.read(str(ply_path))
    el  = ply.elements[0]

    xyz       = np.column_stack([el["x"], el["y"], el["z"]]).astype(np.float64)
    trbf_c    = np.array(el["trbf_center"], dtype=np.float64)   # [N] temporal centers
    trbf_s    = np.array(el["trbf_scale"],  dtype=np.float64)   # [N] log-bandwidth
    motion    = np.column_stack([el[f"motion_{i}"] for i in range(9)]).astype(np.float64)  # [N,9]
    f_dc      = np.column_stack([el["f_dc_0"], el["f_dc_1"], el["f_dc_2"]]).astype(np.float32)
    f_rest    = np.zeros((len(xyz), 45), dtype=np.float32)       # lite model has no f_rest
    opacity_r = np.array(el["opacity"], dtype=np.float64)        # pre-sigmoid
    scales    = np.column_stack([el["scale_0"], el["scale_1"], el["scale_2"]]).astype(np.float32)  # log-scales
    rots      = np.column_stack([el["rot_0"], el["rot_1"], el["rot_2"], el["rot_3"]]).astype(np.float64)
    omega     = np.column_stack([el["omega_0"], el["omega_1"], el["omega_2"], el["omega_3"]]).astype(np.float64)

    timestamps = [i / max(n_keyframes - 1, 1) for i in range(n_keyframes)]

    for i, t in enumerate(timestamps):
        # Time offset from each Gaussian's temporal center
        tforpoly = t - trbf_c                                                   # [N]

        # Temporal RBF visibility weight
        trbf_dist   = tforpoly / np.exp(trbf_s)
        trbf_weight = np.exp(-(trbf_dist ** 2)).clip(0.0, 1.0)                 # [N]

        # 3rd-order polynomial position update
        tf1 = tforpoly[:, None]
        xyz_t = (xyz
                 + motion[:, 0:3] * tf1
                 + motion[:, 3:6] * (tf1 ** 2)
                 + motion[:, 6:9] * (tf1 ** 3))

        # Opacity weighted by temporal visibility, stored back as pre-sigmoid value
        eff_opacity = _sigmoid(opacity_r) * trbf_weight
        opacity_t   = _inv_sigmoid(eff_opacity)

        # Rotation with linear angular velocity
        rot_t = rots + tforpoly[:, None] * omega
        norms = np.linalg.norm(rot_t, axis=1, keepdims=True).clip(1e-8)
        rot_t = (rot_t / norms).astype(np.float32)

        out_path = out_dir / f"keyframe_{i:04d}.ply"
        _write_ply(
            out_path,
            xyz=xyz_t.astype(np.float32),
            f_dc=f_dc,
            f_rest=f_rest,
            opacity=opacity_t.astype(np.float32),
            scales=scales,      # already in log-space from PLY
            rotations=rot_t,
        )

        if i % 10 == 0 or i == n_keyframes - 1:
            print(f"  [{i+1:3d}/{n_keyframes}] t={t:.3f} → {out_path.name}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",  required=True)
    parser.add_argument("--n_keyframes", type=int, default=50)
    parser.add_argument("--output_dir",  required=True)
    parser.add_argument("--iteration",   type=int, default=-1)
    args = parser.parse_args()

    model_path = Path(args.model_path)
    out_dir    = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pc_root = model_path / "point_cloud"
    if args.iteration == -1:
        iteration = max(
            int(d.name.split("_")[1])
            for d in pc_root.iterdir()
            if d.name.startswith("iteration_")
        )
    else:
        iteration = args.iteration

    ply_path = pc_root / f"iteration_{iteration}" / "point_cloud.ply"
    print(f"STG checkpoint: iteration {iteration}, PLY: {ply_path}")

    if not ply_path.exists():
        raise FileNotFoundError(f"PLY not found: {ply_path}")

    bake_from_ply(ply_path, args.n_keyframes, out_dir)
    print(f"Done — {args.n_keyframes} keyframes → {out_dir}")


if __name__ == "__main__":
    main()
