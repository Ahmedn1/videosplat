#!/usr/bin/env python3
"""
Orbit video renderer for 4DGaussians models.

Camera sweeps `arc_degrees` starting from the leftmost training camera
while time advances through n_frames timesteps. Both progress simultaneously,
producing a combined camera-orbit + scene-replay video.

Run with 4DGaussians root on PYTHONPATH:
  python render_orbit.py \
      --model_path /path/to/model \
      --source_path /path/to/n3v_scene \
      --arc_degrees 360 \
      --n_frames 50 \
      --fps 24 \
      --width 1280 \
      --height 720 \
      --output /path/to/output.mp4
"""

import argparse
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


# ── ffmpeg discovery ─────────────────────────────────────────────────────────

def _find_ffmpeg(hint: str = "") -> str:
    """Return path to an ffmpeg binary, preferring the explicit hint."""
    import shutil
    if hint:
        return hint
    # System PATH
    found = shutil.which("ffmpeg")
    if found:
        return found
    # imageio_ffmpeg bundled binary (available in the videosplat venv)
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass
    # Common install locations
    for candidate in ["/usr/bin/ffmpeg", "/usr/local/bin/ffmpeg"]:
        if Path(candidate).exists():
            return candidate
    raise FileNotFoundError(
        "ffmpeg not found. Install it with:  sudo apt-get install ffmpeg"
    )


# ── Geometry helpers ──────────────────────────────────────────────────────────

def normalize(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-8)


def lookat_camera_R(orbit_pos: np.ndarray, scene_center: np.ndarray,
                    gravity_down: np.ndarray) -> np.ndarray:
    """
    Build Camera.R (c2w rotation, columns = camera axes in world space).

    4DGaussians/OpenCV convention after the LLFF transform in dataset_readers:
      col 0 = camera X = world RIGHT
      col 1 = camera Y = world DOWN  (gravity_down)
      col 2 = camera Z = world FORWARD (toward scene)
    """
    cam_z = normalize(scene_center - orbit_pos)  # forward, col 2

    # col 1: project gravity_down onto the plane perpendicular to cam_z
    cam_y_raw = gravity_down - np.dot(gravity_down, cam_z) * cam_z
    if np.linalg.norm(cam_y_raw) < 1e-6:
        cam_y_raw = np.array([0.0, 1.0, 0.0])
    cam_y = normalize(cam_y_raw)  # down, col 1

    # col 0: right = cross(down, forward) — right-hand rule for X = Y × Z
    cam_x = np.cross(cam_y, cam_z)  # right, col 0

    return np.stack([cam_x, cam_y, cam_z], axis=-1)  # columns: right, down, forward


# ── Camera object factory ─────────────────────────────────────────────────────

def make_camera(R: np.ndarray, T: np.ndarray,
                fovx: float, fovy: float, width: int, height: int,
                t: float, uid: int):
    from scene.cameras import Camera
    return Camera(
        colmap_id=uid, R=R, T=T,
        FoVx=fovx, FoVy=fovy,
        image=torch.zeros(3, height, width),
        gt_alpha_mask=None,
        image_name=f"orbit_{uid:04d}",
        uid=uid,
        time=t,
    )


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",   required=True)
    parser.add_argument("--scene_meta",   required=True,
                        help="Path to viewer/scene_meta.json (written by export step).")
    parser.add_argument("--arc_degrees",  type=float, default=360.0)
    parser.add_argument("--n_frames",     type=int,   default=50)
    parser.add_argument("--fps",          type=int,   default=24)
    parser.add_argument("--width",        type=int,   default=1280)
    parser.add_argument("--height",       type=int,   default=720)
    parser.add_argument("--output",       required=True)
    parser.add_argument("--iteration",    type=int,   default=-1)
    parser.add_argument("--white_bg",     action="store_true")
    parser.add_argument("--radius_scale", type=float, default=2.5,
                        help="Orbit radius = cloud_radius_p95 * radius_scale.")
    parser.add_argument("--start_pos",   type=float, nargs=3, default=None,
                        metavar=("X", "Y", "Z"),
                        help="Starting camera position in viewer/world coordinates "
                             "(overrides radius_scale; orbit radius derived from distance to scene center).")
    parser.add_argument("--ffmpeg",       default="",
                        help="Path to ffmpeg binary (auto-detected if empty).")
    args = parser.parse_args()

    import json as _json
    model_path  = Path(args.model_path)
    output_path = Path(args.output)
    frames_dir  = output_path.parent / f"_orbit_tmp_{output_path.stem}"
    frames_dir.mkdir(parents=True, exist_ok=True)

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

    ply_path  = pc_root / f"iteration_{iteration}" / "point_cloud.ply"
    deform_dir = pc_root / f"iteration_{iteration}"
    print(f"Checkpoint: iteration {iteration}")

    # ── Load model ────────────────────────────────────────────────────────────
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

    # ── Read intrinsics from scene_meta.json ─────────────────────────────────
    scene_meta = _json.loads(Path(args.scene_meta).read_text())
    H_orig     = float(scene_meta["image_height"])
    W_orig     = float(scene_meta["image_width"])
    focal_orig = float(scene_meta["focal_length"])

    focal_x = focal_orig * (args.width  / W_orig)
    focal_y = focal_orig * (args.height / H_orig)
    fovx = 2 * math.atan(args.width  / (2 * focal_x))
    fovy = 2 * math.atan(args.height / (2 * focal_y))

    # Camera positions from scene_meta (used only for orbit start direction)
    _meta_cams = scene_meta.get("cameras", [])
    cam_positions = (
        np.array([c["pos"] for c in _meta_cams], dtype=float)
        if _meta_cams else np.array([[1.0, 0.0, 0.0]])
    )

    # ── Determine scene geometry from actual Gaussian positions ───────────────
    # Camera positions from scene_meta may differ in scale from the trained
    # Gaussian cloud, so use get_xyz for all orbit geometry calculations.
    xyz_all = gaussians.get_xyz.detach().cpu().numpy()   # (N, 3)

    # Strip outliers (beyond 5th–95th percentile per axis) to get the core cloud.
    mask = np.ones(len(xyz_all), dtype=bool)
    for axis in range(3):
        lo = np.percentile(xyz_all[:, axis], 5)
        hi = np.percentile(xyz_all[:, axis], 95)
        mask &= (xyz_all[:, axis] >= lo) & (xyz_all[:, axis] <= hi)
    xyz_core = xyz_all[mask]

    scene_center = xyz_core.mean(axis=0)

    # ── Orbit plane normal ────────────────────────────────────────────────────
    # The mkkellogg viewer uses cameraUp:[0,-1,0], meaning +Y = world down,
    # -Y = world up, and the Gaussian PLY positions are in the same frame.
    # The horizontal orbit rotates around the Y axis (XZ plane).
    cam_down_avg = np.array([0.0, 1.0, 0.0])   # world down = +Y (viewer convention)
    orbit_normal  = np.array([0.0, -1.0, 0.0])  # world up  = -Y

    # ── Orbit radius + start direction ────────────────────────────────────────
    if args.start_pos is not None:
        # Use the user-supplied viewer position directly.
        start_pos = np.array(args.start_pos, dtype=float)
        diff = start_pos - scene_center
        height_offset = float(np.dot(diff, orbit_normal))
        diff_inplane  = diff - height_offset * orbit_normal
        radius = float(np.linalg.norm(diff_inplane))
        if radius < 1e-4:
            diff_inplane = np.array([1.0, 0.0, 0.0])
            radius = 1.0
        print(f"Start pos (user):  {start_pos.round(3)}")
        print(f"Orbit radius (from start_pos): {radius:.3f},  height offset: {height_offset:.3f}")
    else:
        # Auto: orbit outside the Gaussian cloud.
        xyz_inplane = xyz_core - np.outer(xyz_core @ orbit_normal, orbit_normal)
        center_inplane = scene_center - np.dot(scene_center, orbit_normal) * orbit_normal
        radii_from_center = np.linalg.norm(xyz_inplane - center_inplane, axis=1)
        cloud_radius = float(np.percentile(radii_from_center, 95))
        radius = cloud_radius * args.radius_scale
        xyz_heights = xyz_core @ orbit_normal
        height_offset = float(np.median(xyz_heights) - np.dot(scene_center, orbit_normal))
        print(f"Cloud radius (p95):  {cloud_radius:.3f}  → orbit radius: {radius:.3f}")

    diff_inplane = (
        (np.array(args.start_pos) - scene_center) if args.start_pos is not None
        else (cam_positions[0] - np.dot(cam_positions[0], orbit_normal) * orbit_normal
              - (scene_center - np.dot(scene_center, orbit_normal) * orbit_normal))
    )
    diff_inplane = diff_inplane - np.dot(diff_inplane, orbit_normal) * orbit_normal
    if np.linalg.norm(diff_inplane) < 1e-4:
        diff_inplane = np.array([1.0, 0.0, 0.0])

    orbit_axis1 = normalize(diff_inplane)
    orbit_axis2 = normalize(np.cross(orbit_normal, orbit_axis1))

    arc_rad = math.radians(args.arc_degrees)

    print(f"Gaussians (core): {len(xyz_core):,} / {len(xyz_all):,}")
    print(f"Scene center (Gaussians): {scene_center.round(3)}")
    print(f"Orbit normal (world up): {orbit_normal}  gravity down: {cam_down_avg}")
    print(f"Arc: {args.arc_degrees}°, {args.n_frames} frames @ {args.fps} fps")
    print(f"Output: {output_path}  ({args.width}×{args.height})")

    # ── Pipeline ──────────────────────────────────────────────────────────────
    bg_color   = [1.0, 1.0, 1.0] if args.white_bg else [0.0, 0.0, 0.0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    from gaussian_renderer import render as gs_render

    # ── Render loop ───────────────────────────────────────────────────────────
    for i in range(args.n_frames):
        frac  = i / max(args.n_frames - 1, 1)     # 0 → 1
        t     = frac                               # scene time
        angle = arc_rad * frac                     # camera sweep angle

        orbit_pos = (
            scene_center
            + height_offset * orbit_normal
            + radius * (math.cos(angle) * orbit_axis1 + math.sin(angle) * orbit_axis2)
        )

        R_cam = lookat_camera_R(orbit_pos, scene_center, cam_down_avg)
        T_cam = -(orbit_pos @ R_cam)               # = -R.T @ pos

        cam = make_camera(R_cam, T_cam, fovx, fovy,
                          args.width, args.height, t, uid=i)

        with torch.no_grad():
            pkg   = gs_render(cam, gaussians, hp, background)
            image = pkg["render"]                  # (3, H, W) float [0,1]

        img_np = (image.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(img_np).save(str(frames_dir / f"frame_{i:04d}.png"))

        if i % 10 == 0 or i == args.n_frames - 1:
            print(f"  [{i+1:3d}/{args.n_frames}] t={t:.3f}  angle={math.degrees(angle):.1f}°")

    # ── Encode video ──────────────────────────────────────────────────────────
    ffmpeg_exe = _find_ffmpeg(args.ffmpeg)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg_cmd = [
        ffmpeg_exe, "-y",
        "-framerate", str(args.fps),
        "-i", str(frames_dir / "frame_%04d.png"),
        "-c:v", "libx264",
        "-preset", "slow",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        str(output_path),
    ]
    print("Encoding video…")
    subprocess.run(ffmpeg_cmd, check=True)
    shutil.rmtree(frames_dir)
    print(f"Done → {output_path}")


if __name__ == "__main__":
    main()
