#!/usr/bin/env python3
"""Convert a 3DGS / 4DGaussians keyframe PLY to a compact antimatter15 `.splat`.

The `.splat` format (32 bytes/Gaussian) drops spherical-harmonics to SH0 (flat,
view-independent colour), shrinking a ~20-60 MB PLY to ~5-8 MB — small enough to
ship a single interactive frame on a static host (Netlify). The full SH3 + 4D
playback stays in the local viewer (`videosplat view`).

Layout per splat (little-endian): position f32x3 | scale f32x3 (linear) |
rgba u8x4 (SH0 colour + sigmoid opacity) | rotation u8x4 (normalised quat).
This matches antimatter15/splat so @mkkellogg/gaussian-splats-3d loads it directly.

Usage: python scripts/ply_to_splat.py in.ply out.splat [--max-splats N]
"""
import argparse
import numpy as np
from plyfile import PlyData

SH_C0 = 0.28209479177387814  # 0th-order SH → RGB


def ply_to_splat(ply_path: str, out_path: str, max_splats: int = 0) -> int:
    v = PlyData.read(ply_path)["vertex"].data
    xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float32)
    scale = np.exp(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], 1)).astype(np.float32)
    opacity = 1.0 / (1.0 + np.exp(-v["opacity"]))                       # sigmoid
    f_dc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], 1)
    rgb = np.clip(0.5 + SH_C0 * f_dc, 0.0, 1.0)
    rot = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], 1).astype(np.float32)
    rot /= (np.linalg.norm(rot, axis=1, keepdims=True) + 1e-9)

    # importance = opacity * volume; keep the most visually significant splats
    order = np.argsort(-(opacity * scale.prod(1)))
    if max_splats and len(order) > max_splats:
        order = order[:max_splats]
    xyz, scale, rgb, opacity, rot = xyz[order], scale[order], rgb[order], opacity[order], rot[order]

    n = len(xyz)
    buf = np.zeros((n, 32), dtype=np.uint8)
    buf[:, 0:12]  = xyz.view(np.uint8).reshape(n, 12)
    buf[:, 12:24] = scale.view(np.uint8).reshape(n, 12)
    buf[:, 24:27] = (rgb * 255).round().clip(0, 255).astype(np.uint8)
    buf[:, 27]    = (opacity * 255).round().clip(0, 255).astype(np.uint8)
    buf[:, 28:32] = (rot * 128 + 128).round().clip(0, 255).astype(np.uint8)
    buf.tofile(out_path)
    return n


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ply"); ap.add_argument("out")
    ap.add_argument("--max-splats", type=int, default=0, help="cap splat count (0 = all)")
    a = ap.parse_args()
    n = ply_to_splat(a.ply, a.out, a.max_splats)
    import os
    print(f"{a.ply} -> {a.out}: {n:,} splats, {os.path.getsize(a.out)/1e6:.1f} MB")
