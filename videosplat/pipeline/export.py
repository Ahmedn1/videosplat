from __future__ import annotations

"""
Export pipeline: keyframe PLY snapshots → browser-ready .splat files.

For each keyframe PLY produced by train.py's render step, we apply the same
conversion logic:

  PLY vertex attributes → 32-byte .splat record per Gaussian:
    bytes  0-11  position xyz           3 × float32
    bytes 12-23  scale xyz              3 × float32
    bytes 24-27  colour RGBA            4 × uint8
    bytes 28-31  rotation quaternion    4 × uint8

The exported files land in viewer/frames/ as:
  keyframe_0000.splat
  keyframe_0001.splat
  ...

A scene_meta.json is also written with the information the viewer needs
to drive the timeline:
  {
    "n_keyframes": 24,
    "fps": 8.0,
    "label": "scene",
    "n_cameras": 4,
    "cameras": [{"pos": [...], "forward": [...], "up": [...], "name": "..."}]
  }
"""

import json
import struct
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

console = Console()

_SH_C0 = 0.28209479177387814


# ── Public API ──────────────────────────────────────────────────────────────────

def export_keyframes(
    keyframes_dir: Path,
    viewer_dir: Path,
    sparse_dir: Path | None,
    *,
    label: str = "scene",
    n_cameras: int = 1,
    fps: float = 8.0,
    image_height: int | None = None,
    image_width: int | None = None,
    focal_length: float | None = None,
    extra_cameras: list[dict] | None = None,
) -> dict:
    """
    Copy keyframe PLYs to viewer/frames/ and write scene_meta.json.

    Args:
        keyframes_dir:  Directory containing keyframe_XXXX.ply files.
        viewer_dir:     Output directory for the browser viewer assets.
        sparse_dir:     COLMAP sparse/0/ dir for camera poses (optional).
        label:          Label shown in the viewer header.
        n_cameras:      Number of physical cameras used.
        fps:            Playback framerate for the timeline scrubber.
        image_height:   Training image height in pixels (stored for renderer).
        image_width:    Training image width in pixels (stored for renderer).
        focal_length:   Training focal length in pixels (stored for renderer).
        extra_cameras:  Camera pose list [{pos, forward, up, name}] to use when
                        sparse_dir is None or yields no cameras.

    Returns:
        dict with n_keyframes, n_gaussians_per_frame
    """
    import shutil as _shutil
    ply_files = sorted(keyframes_dir.glob("keyframe_*.ply"))
    if not ply_files:
        raise FileNotFoundError(f"No keyframe PLYs found in {keyframes_dir}")

    frames_dir = viewer_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    n_gaussians_list = []

    with Progress(
        SpinnerColumn(), TextColumn("  Exporting"), BarColumn(), TaskProgressColumn(),
        console=console, transient=True,
    ) as prog:
        task = prog.add_task("", total=len(ply_files))

        for ply_path in ply_files:
            dst = frames_dir / ply_path.name
            _shutil.copy2(ply_path, dst)
            n = _count_ply_vertices(dst)
            n_gaussians_list.append(n)
            prog.advance(task)

    cameras = _read_cameras(sparse_dir) if sparse_dir else []
    if not cameras and extra_cameras:
        cameras = extra_cameras

    scene_center, scene_radius = _scene_bounds(ply_files[0])

    meta: dict = {
        "n_keyframes": len(ply_files),
        "fps": fps,
        "label": label,
        "n_cameras": n_cameras,
        "cameras": cameras,
        "scene_center": scene_center,
        "scene_radius": scene_radius,
    }
    if image_height is not None:
        meta["image_height"] = int(image_height)
    if image_width is not None:
        meta["image_width"] = int(image_width)
    if focal_length is not None:
        meta["focal_length"] = round(float(focal_length), 4)

    (viewer_dir / "scene_meta.json").write_text(json.dumps(meta, indent=None))

    avg_g = int(np.mean(n_gaussians_list)) if n_gaussians_list else 0
    console.print(
        f"  [green]Exported {len(ply_files)} keyframes, "
        f"~{avg_g:,} Gaussians each → {frames_dir}[/]"
    )

    return {"n_keyframes": len(ply_files), "n_gaussians_per_frame": avg_g}


def _scene_bounds(ply_path: Path) -> tuple[list[float], float]:
    """Return (scene_center, scene_radius) from a keyframe PLY, filtering outliers."""
    try:
        from plyfile import PlyData
        v = PlyData.read(str(ply_path))["vertex"]
        xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)
        mask = np.ones(len(xyz), dtype=bool)
        for i in range(3):
            lo, hi = np.percentile(xyz[:, i], 5), np.percentile(xyz[:, i], 95)
            mask &= (xyz[:, i] >= lo) & (xyz[:, i] <= hi)
        core = xyz[mask]
        center = core.mean(axis=0)
        radius = float(np.percentile(np.linalg.norm(core - center, axis=1), 95))
        return [round(float(c), 4) for c in center], round(radius, 4)
    except Exception:
        return [0.0, 0.0, 0.0], 5.0


def _count_ply_vertices(ply_path: Path) -> int:
    try:
        from plyfile import PlyData
        return len(PlyData.read(str(ply_path))["vertex"])
    except Exception:
        return 0


# ── PLY → .splat (kept for future use) ──────────────────────────────────────────

def _ply_to_splat(ply_path: Path, splat_path: Path) -> int:
    from plyfile import PlyData

    ply = PlyData.read(str(ply_path))
    v = ply["vertex"]
    n = len(v)

    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)

    scales = np.exp(
        np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=1)
    ).astype(np.float32)
    scales = np.maximum(scales, 1e-6)

    alpha = _sigmoid(np.array(v["opacity"])).astype(np.float32)

    # Cap needle Gaussians (ratio > 30:1)
    s_min = scales.min(axis=1, keepdims=True)
    scales = np.minimum(scales, s_min * 30.0)

    f_dc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1).astype(np.float32)
    rgb = np.clip(_SH_C0 * f_dc + 0.5, 0.0, 1.0)
    rgb_u8 = np.round(rgb * 255).astype(np.uint8)
    alpha_u8 = np.clip(alpha * 255, 0, 255).astype(np.uint8)
    rgba_u8 = np.concatenate([rgb_u8, alpha_u8[:, None]], axis=1)

    rot_raw = np.stack(
        [v["rot_1"], v["rot_2"], v["rot_3"], v["rot_0"]], axis=1  # xyzw order
    ).astype(np.float64)
    norms = np.linalg.norm(rot_raw, axis=1, keepdims=True)
    norms = np.where(norms < 1e-8, 1.0, norms)
    rot_norm = (rot_raw / norms).astype(np.float32)
    rot_u8 = np.clip((rot_norm + 1.0) / 2.0 * 255, 0, 255).astype(np.uint8)

    # Sort by opacity descending
    order = np.argsort(-alpha)
    xyz      = xyz[order]
    scales   = scales[order]
    rgba_u8  = rgba_u8[order]
    rot_u8   = rot_u8[order]

    buf = np.zeros(n, dtype=[
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("sx", "<f4"), ("sy", "<f4"), ("sz", "<f4"),
        ("r", "u1"), ("g", "u1"), ("b", "u1"), ("a", "u1"),
        ("qx", "u1"), ("qy", "u1"), ("qz", "u1"), ("qw", "u1"),
    ])
    buf["x"], buf["y"], buf["z"]     = xyz[:, 0],    xyz[:, 1],    xyz[:, 2]
    buf["sx"], buf["sy"], buf["sz"]  = scales[:, 0], scales[:, 1], scales[:, 2]
    buf["r"], buf["g"], buf["b"], buf["a"] = (
        rgba_u8[:, 0], rgba_u8[:, 1], rgba_u8[:, 2], rgba_u8[:, 3]
    )
    buf["qx"], buf["qy"], buf["qz"], buf["qw"] = (
        rot_u8[:, 0], rot_u8[:, 1], rot_u8[:, 2], rot_u8[:, 3]
    )
    splat_path.write_bytes(buf.tobytes())
    return n


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


# ── Camera pose extraction ───────────────────────────────────────────────────────

def _read_cameras(sparse_dir: Path | None) -> list[dict]:
    if sparse_dir is None:
        return []
    images_bin = sparse_dir / "images.bin"
    if not images_bin.exists():
        return []
    cameras = []
    try:
        from scipy.spatial.transform import Rotation
        with open(images_bin, "rb") as f:
            (n_images,) = struct.unpack("<Q", f.read(8))
            for _ in range(n_images):
                (image_id,) = struct.unpack("<I", f.read(4))
                qw, qx, qy, qz = struct.unpack("<4d", f.read(32))
                tx, ty, tz = struct.unpack("<3d", f.read(24))
                (camera_id,) = struct.unpack("<I", f.read(4))
                name = b""
                while True:
                    ch = f.read(1)
                    if ch == b"\x00":
                        break
                    name += ch
                name = name.decode("utf-8")
                (n_pts,) = struct.unpack("<Q", f.read(8))
                f.read(n_pts * 24)

                R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
                t = np.array([tx, ty, tz])
                pos     = (-R.T @ t).tolist()
                forward = R[2].tolist()
                up      = (-R[1]).tolist()
                cameras.append({"pos": pos, "forward": forward, "up": up, "name": name})
    except Exception as e:
        console.print(f"  [yellow]Could not read COLMAP cameras: {e}[/]")
    cameras.sort(key=lambda c: c["name"])
    return cameras
