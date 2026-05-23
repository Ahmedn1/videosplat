#!/usr/bin/env python3
"""
Camera-path video renderer for 4DGaussians models.

Reads a camera_path.json produced by the path editor (list of waypoints,
each with a world-space position and look-at target), interpolates a smooth
Catmull-Rom spline through all waypoints, and renders one frame per
interpolated position using the 4DGaussians deformation renderer.

Run with 4DGaussians root on PYTHONPATH:
  python render_path.py \
      --model_path /path/to/model \
      --source_path /path/to/n3v_scene \
      --path_json  /path/to/camera_path.json \
      --n_frames 120 \
      --fps 24 \
      --output /path/to/output.mp4
"""

import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


# ── ffmpeg discovery ──────────────────────────────────────────────────────────

def _find_ffmpeg(hint: str = "") -> str:
    if hint:
        return hint
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass
    for candidate in ["/usr/bin/ffmpeg", "/usr/local/bin/ffmpeg"]:
        if Path(candidate).exists():
            return candidate
    raise FileNotFoundError("ffmpeg not found. Install: sudo apt-get install ffmpeg")


# ── Geometry helpers ──────────────────────────────────────────────────────────

def normalize(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-8)


def catmull_rom_spline(points: np.ndarray, n_samples: int) -> np.ndarray:
    """
    Interpolate a smooth Catmull-Rom spline through `points` (N, D).
    Returns (n_samples, D) array of interpolated positions.
    """
    n = len(points)
    if n == 1:
        return np.tile(points[0], (n_samples, 1))
    if n == 2:
        t = np.linspace(0, 1, n_samples)[:, None]
        return (1 - t) * points[0] + t * points[1]

    # Add phantom endpoints for Catmull-Rom boundary conditions
    pts = np.concatenate([
        [2 * points[0] - points[1]],
        points,
        [2 * points[-1] - points[-2]],
    ], axis=0)

    result = []
    segments = n - 1
    for i_sample in range(n_samples):
        u = i_sample / max(n_samples - 1, 1) * segments
        seg = min(int(u), segments - 1)
        t = u - seg

        p0 = pts[seg]
        p1 = pts[seg + 1]
        p2 = pts[seg + 2]
        p3 = pts[seg + 3]

        # Catmull-Rom formula
        pos = 0.5 * (
            (2 * p1)
            + (-p0 + p2) * t
            + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2
            + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3
        )
        result.append(pos)
    return np.array(result)


def lookat_camera_R(cam_pos: np.ndarray, look_at: np.ndarray,
                    gravity_down: np.ndarray) -> np.ndarray:
    """
    Build c2w rotation R (columns = camera axes in world space):
      col 0 = right, col 1 = down, col 2 = forward.
    """
    cam_z = normalize(look_at - cam_pos)

    cam_y_raw = gravity_down - np.dot(gravity_down, cam_z) * cam_z
    if np.linalg.norm(cam_y_raw) < 1e-6:
        cam_y_raw = np.array([1.0, 0.0, 0.0])
    cam_y = normalize(cam_y_raw)

    cam_x = np.cross(cam_y, cam_z)
    return np.stack([cam_x, cam_y, cam_z], axis=-1)


def make_camera(R, T, fovx, fovy, width, height, t, uid):
    from scene.cameras import Camera
    return Camera(
        colmap_id=uid, R=R, T=T,
        FoVx=fovx, FoVy=fovy,
        image=torch.zeros(3, height, width),
        gt_alpha_mask=None,
        image_name=f"path_{uid:04d}",
        uid=uid,
        time=t,
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",  required=True)
    parser.add_argument("--scene_meta",  required=True,
                        help="Path to viewer/scene_meta.json (written by export step).")
    parser.add_argument("--path_json",   required=True)
    parser.add_argument("--n_frames",    type=int,   default=120)
    parser.add_argument("--fps",         type=int,   default=24)
    parser.add_argument("--width",       type=int,   default=1280)
    parser.add_argument("--height",      type=int,   default=720)
    parser.add_argument("--output",        required=True)
    parser.add_argument("--iteration",    type=int,   default=-1)
    parser.add_argument("--white_bg",     action="store_true")
    parser.add_argument("--forward_shift", type=float, default=0.0,
                        help="Shift camera forward by this many world units along its look direction.")
    parser.add_argument("--ffmpeg",       default="")
    args = parser.parse_args()

    model_path  = Path(args.model_path)
    output_path = Path(args.output)
    frames_dir  = output_path.parent / f"_path_tmp_{output_path.stem}"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # ── Load path JSON ────────────────────────────────────────────────────────
    path_data  = json.loads(Path(args.path_json).read_text())
    waypoints  = path_data["waypoints"]   # [{pos:[x,y,z], lookAt:[x,y,z]}, ...]
    if len(waypoints) < 2:
        raise ValueError("Need at least 2 waypoints in camera_path.json")

    positions = np.array([wp["pos"]    for wp in waypoints], dtype=float)
    look_ats  = np.array([wp["lookAt"] for wp in waypoints], dtype=float)

    print(f"Path: {len(waypoints)} waypoints → {args.n_frames} frames")

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

    # ── Camera intrinsics ─────────────────────────────────────────────────────
    # Prefer FOV exported from the path editor viewer so the renderer matches
    # exactly what the user saw when placing waypoints.
    if "fovY" in path_data:
        fovy_deg  = float(path_data["fovY"])
        fovy      = math.radians(fovy_deg)
        # Derive fovx from the render resolution aspect ratio
        fovx      = 2 * math.atan(math.tan(fovy / 2) * (args.width / args.height))
        print(f"FOV from path editor: {fovy_deg:.1f}° vertical  "
              f"→ fovx={math.degrees(fovx):.1f}°  fovy={fovy_deg:.1f}°")
    else:
        # Fall back to training intrinsics from scene_meta.json
        _smeta     = json.loads(Path(args.scene_meta).read_text())
        H_orig     = float(_smeta["image_height"])
        W_orig     = float(_smeta["image_width"])
        focal_orig = float(_smeta["focal_length"])
        focal_x    = focal_orig * (args.width  / W_orig)
        focal_y    = focal_orig * (args.height / H_orig)
        fovx       = 2 * math.atan(args.width  / (2 * focal_x))
        fovy       = 2 * math.atan(args.height / (2 * focal_y))
        print(f"FOV from training intrinsics: "
              f"fovx={math.degrees(fovx):.1f}°  fovy={math.degrees(fovy):.1f}°")

    # ── Interpolate spline ────────────────────────────────────────────────────
    pos_spline  = catmull_rom_spline(positions, args.n_frames)
    look_spline = catmull_rom_spline(look_ats,  args.n_frames)

    # World down = +Y (mkkellogg/viewer convention: cameraUp = [0,-1,0])
    gravity_down = np.array([0.0, 1.0, 0.0])

    # ── Pipeline ──────────────────────────────────────────────────────────────
    bg_color   = [1.0, 1.0, 1.0] if args.white_bg else [0.0, 0.0, 0.0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    from gaussian_renderer import render as gs_render

    print(f"Rendering {args.n_frames} frames  (forward_shift={args.forward_shift})…")
    for i in range(args.n_frames):
        t       = i / max(args.n_frames - 1, 1)
        cam_pos = pos_spline[i]
        look_at = look_spline[i]


        R_cam = lookat_camera_R(cam_pos, look_at, gravity_down)
        T_cam = -(cam_pos @ R_cam)

        cam = make_camera(R_cam, T_cam, fovx, fovy,
                          args.width, args.height, t, uid=i)

        with torch.no_grad():
            pkg   = gs_render(cam, gaussians, hp, background)
            image = pkg["render"]

        img_np = (image.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(img_np).save(str(frames_dir / f"frame_{i:04d}.png"))

        if i % 20 == 0 or i == args.n_frames - 1:
            print(f"  [{i+1:4d}/{args.n_frames}] t={t:.3f}  "
                  f"pos=({cam_pos[0]:.2f}, {cam_pos[1]:.2f}, {cam_pos[2]:.2f})")

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
