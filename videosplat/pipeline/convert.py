from __future__ import annotations

"""
Format conversion utilities for the videosplat pipeline.

Public functions:

  colmap_to_poses_bounds(sparse_dir, out_path)
      COLMAP binary sparse/0/ → poses_bounds.npy (LLFF format)
      used by the 4DGS training path

  frames_to_videos(cam_frame_dirs, out_dir, fps)
      per-camera frame dirs → cam00.mp4 cam01.mp4 …
      used by the 4DGS training path

  prepare_stg_source(sparse_dir, cam_dirs, out_dir)
      COLMAP sparse/0/ + cam_XX/ frame dirs → STG colmap_N/ structure

  prepare_transforms_source(sparse_dir, cam_dirs, out_dir)
      COLMAP sparse/0/ + cam_XX/ frame dirs → transforms_train.json
      (nerfstudio format; used by Gaussian-Flow and 4D-Rotor)
"""

import json
import shutil
import struct
import subprocess
from pathlib import Path

import numpy as np
from rich.console import Console

console = Console()


# ── Public API ──────────────────────────────────────────────────────────────────

def colmap_to_poses_bounds(sparse_dir: Path, out_path: Path) -> int:
    """
    Convert COLMAP sparse/0/ binary output → poses_bounds.npy (LLFF format).

    Groups images by camera prefix (cam_00_frame_*.jpg → cam_00), takes the
    median pose for each physical camera, and packs into (N_cams, 17) LLFF array.

    Returns the number of cameras written.
    """
    cameras_bin  = sparse_dir / "cameras.bin"
    images_bin   = sparse_dir / "images.bin"
    points3d_bin = sparse_dir / "points3D.bin"

    if not cameras_bin.exists() or not images_bin.exists():
        raise FileNotFoundError(f"COLMAP binaries not found in {sparse_dir}")

    cameras  = _read_cameras_bin(cameras_bin)
    images   = _read_images_bin(images_bin)
    points3d = _read_points3d_bin(points3d_bin) if points3d_bin.exists() else {}

    # Group images by physical camera (prefix before "_frame_")
    cam_groups: dict[str, list[dict]] = {}
    for img in images.values():
        key = img["name"].split("_frame_")[0]  # "cam_00"
        cam_groups.setdefault(key, []).append(img)

    # Collect sparse point XYZ for near/far computation.
    # Filter long-tail outlier triangulations (background reflections, stray
    # ceiling points, etc.) using the camera cluster as the anchor: any point
    # farther than K × camera_radius from the camera centroid is dropped.
    # Without this filter, a few outlier points 100+ units away poison the
    # depth percentile inside _compute_bounds and drag `near` to ~0, which
    # breaks 4DGS's depth-aware densification heuristics. Same mechanism as
    # the init point cloud filter in train.py:_ensure_init_pointcloud.
    pts_xyz_all = np.array([p["xyz"] for p in points3d.values()]) if points3d else None
    pts_xyz: np.ndarray | None = pts_xyz_all
    if pts_xyz_all is not None and len(pts_xyz_all) > 0:
        cam_centers_for_filter = []
        for cam_name in sorted(cam_groups.keys()):
            imgs = cam_groups[cam_name]
            ts = []
            for img in imgs:
                R_w2c = _quat_to_rotmat(img["qw"], img["qx"], img["qy"], img["qz"])
                t_w2c = np.array([img["tx"], img["ty"], img["tz"]])
                ts.append(-R_w2c.T @ t_w2c)
            cam_centers_for_filter.append(np.median(np.stack(ts), axis=0))
        cam_centers_for_filter = np.stack(cam_centers_for_filter)
        centroid = cam_centers_for_filter.mean(axis=0)
        radius   = float(np.linalg.norm(cam_centers_for_filter - centroid, axis=1).max())
        keep_r   = 3.0 * radius
        dist     = np.linalg.norm(pts_xyz_all - centroid, axis=1)
        inlier   = dist < keep_r
        pts_xyz  = pts_xyz_all[inlier]
        n_drop   = len(pts_xyz_all) - len(pts_xyz)
        if n_drop > 0:
            console.print(
                f"  [dim]bounds: {len(pts_xyz_all):,} sparse pts → {len(pts_xyz):,} "
                f"inliers within {keep_r:.2f} of camera centroid (dropped {n_drop:,} "
                f"outliers)[/]"
            )

    poses_list  = []
    bounds_list = []

    for cam_name in sorted(cam_groups.keys()):
        imgs = cam_groups[cam_name]

        # Compute R_c2w and cam_pos for each frame, then take median
        R_c2w_list = []
        pos_list   = []
        cam_id_used = imgs[0]["camera_id"]

        for img in imgs:
            R_w2c = _quat_to_rotmat(img["qw"], img["qx"], img["qy"], img["qz"])
            t_w2c = np.array([img["tx"], img["ty"], img["tz"]])
            R_c2w = R_w2c.T
            cam_pos = -R_c2w @ t_w2c
            R_c2w_list.append(R_c2w)
            pos_list.append(cam_pos)

        # Median over frames (cameras are static; this filters COLMAP jitter)
        R_c2w = np.median(np.stack(R_c2w_list), axis=0)
        # Re-orthogonalize via SVD
        U, _, Vt = np.linalg.svd(R_c2w)
        R_c2w = U @ Vt
        cam_pos = np.median(np.stack(pos_list), axis=0)

        # COLMAP OpenCV c2w cols: [right, down, forward]
        # LLFF poses_bounds.npy cols (Mildenhall convention, what every
        # downstream LLFF reader expects):  [down, right, back]
        #
        # Why this matters: 4DGaussians' dynerf reader takes the LLFF poses
        # and applies `[col1, -col0, col2]` to recover NeRF/OpenGL
        # `[right, up, back]` on load. If we write `[right, up, back]`
        # directly (a previous bug here), 4DGS interprets it as LLFF and
        # the swap rotates every camera ~90° about the back axis — making
        # train PSNR look fine (memorizing each pose's own views) while
        # test PSNR collapses to ~10 because cross-view geometry is wrong.
        # An hour of debugging coffee_martini quality led us here.
        R_llff = np.zeros_like(R_c2w)
        R_llff[:, 0] = R_c2w[:, 1]    # LLFF col 0 = down  (= OpenCV col 1)
        R_llff[:, 1] = R_c2w[:, 0]    # LLFF col 1 = right (= OpenCV col 0)
        R_llff[:, 2] = -R_c2w[:, 2]   # LLFF col 2 = back  (= -OpenCV forward)

        # Intrinsics from COLMAP camera
        cam = cameras.get(cam_id_used, list(cameras.values())[0])
        H, W = int(cam["height"]), int(cam["width"])
        focal = _focal_from_params(cam["model"], cam["params"])

        # 3×5 pose: [R_llff | pos | [H; W; focal]]
        pose35 = np.zeros((3, 5))
        pose35[:, :3] = R_llff
        pose35[:, 3]  = cam_pos
        pose35[0, 4]  = H
        pose35[1, 4]  = W
        pose35[2, 4]  = focal
        poses_list.append(pose35)

        # Near/far from sparse 3D points projected into this camera
        near, far = _compute_bounds(R_c2w, cam_pos, pts_xyz)
        bounds_list.append([near, far])

    N = len(poses_list)
    poses_flat = np.array(poses_list).reshape(N, 15)   # (N, 15)
    bounds     = np.array(bounds_list)                  # (N, 2)
    poses_bounds = np.concatenate([poses_flat, bounds], axis=1)  # (N, 17)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out_path), poses_bounds)
    console.print(f"  [dim]poses_bounds.npy: {N} cameras → {out_path}[/]")
    return N


def frames_to_videos(
    cam_frame_dirs: list[Path],
    out_dir: Path,
    fps: float = 10.0,
    ffmpeg_exe: str = "",
) -> list[Path]:
    """
    Encode per-camera frame directories → cam00.mp4, cam01.mp4, … in out_dir.

    cam_frame_dirs[i] must contain frame_XXXXXX.jpg files (sync.py output).
    Returns list of created MP4 paths.
    """
    ffmpeg = _find_ffmpeg(ffmpeg_exe)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_paths = []

    for idx, cam_dir in enumerate(sorted(cam_frame_dirs)):
        out_mp4 = out_dir / f"cam{idx:02d}.mp4"
        if out_mp4.exists():
            out_paths.append(out_mp4)
            continue

        frames = sorted(cam_dir.glob("frame_*.jpg"))
        if not frames:
            console.print(f"  [yellow]No frames in {cam_dir} — skipping[/]")
            continue

        cmd = [
            ffmpeg, "-y",
            "-framerate", str(fps),
            "-i", str(cam_dir / "frame_%06d.jpg"),
            "-c:v", "libx264",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            str(out_mp4),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        out_paths.append(out_mp4)
        console.print(f"  [dim]cam{idx:02d}.mp4: {len(frames)} frames[/]")

    console.print(f"  [green]{len(out_paths)} camera videos written → {out_dir}[/]")
    return out_paths


# ── COLMAP binary readers ────────────────────────────────────────────────────

def _read_cameras_bin(path: Path) -> dict:
    cameras = {}
    _NUM_PARAMS = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8}
    _MODEL_NAMES = {0: "SIMPLE_PINHOLE", 1: "PINHOLE", 2: "SIMPLE_RADIAL",
                    3: "RADIAL", 4: "OPENCV", 5: "OPENCV_FISHEYE"}
    with open(path, "rb") as f:
        num = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num):
            cam_id   = struct.unpack("<I", f.read(4))[0]
            model_id = struct.unpack("<i", f.read(4))[0]
            w        = struct.unpack("<Q", f.read(8))[0]
            h        = struct.unpack("<Q", f.read(8))[0]
            n_params = _NUM_PARAMS.get(model_id, 4)
            params   = list(struct.unpack(f"<{n_params}d", f.read(8 * n_params)))
            cameras[cam_id] = {
                "model":  _MODEL_NAMES.get(model_id, "PINHOLE"),
                "width":  w,
                "height": h,
                "params": params,
            }
    return cameras


def _read_images_bin(path: Path) -> dict:
    images = {}
    with open(path, "rb") as f:
        num = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num):
            img_id         = struct.unpack("<I",  f.read(4))[0]
            qw, qx, qy, qz = struct.unpack("<4d", f.read(32))
            tx, ty, tz     = struct.unpack("<3d", f.read(24))
            cam_id         = struct.unpack("<I",  f.read(4))[0]
            name = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name += c
            name     = name.decode("utf-8")
            num_pts  = struct.unpack("<Q", f.read(8))[0]
            f.read(num_pts * 24)  # skip 2D points (x, y, point3D_id each 8 bytes)
            images[img_id] = {
                "image_id":  img_id,
                "qw": qw, "qx": qx, "qy": qy, "qz": qz,
                "tx": tx, "ty": ty, "tz": tz,
                "camera_id": cam_id,
                "name":      name,
            }
    return images


def _read_points3d_bin(path: Path) -> dict:
    points = {}
    with open(path, "rb") as f:
        num = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num):
            pt_id     = struct.unpack("<Q",  f.read(8))[0]
            xyz       = np.array(struct.unpack("<3d", f.read(24)))
            f.read(3)   # rgb (3 bytes)
            f.read(8)   # error (double)
            track_len = struct.unpack("<Q", f.read(8))[0]
            f.read(track_len * 8)  # skip track elements (image_id + point2D_idx each 4 bytes)
            points[pt_id] = {"xyz": xyz}
    return points


# ── COLMAP binary writers ─────────────────────────────────────────────────────

def _write_cameras_bin(path: Path, cameras: list[dict]) -> None:
    """Write COLMAP cameras.bin.  Each entry: {camera_id, model, width, height, params}."""
    _MODEL_IDS = {"SIMPLE_PINHOLE": 0, "PINHOLE": 1, "SIMPLE_RADIAL": 2,
                  "RADIAL": 3, "OPENCV": 4}
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(cameras)))
        for cam in cameras:
            model_id = _MODEL_IDS.get(cam["model"], 1)
            params   = cam["params"]
            f.write(struct.pack("<I",            cam["camera_id"]))
            f.write(struct.pack("<i",            model_id))
            f.write(struct.pack("<Q",            cam["width"]))
            f.write(struct.pack("<Q",            cam["height"]))
            f.write(struct.pack(f"<{len(params)}d", *params))


def _write_images_bin(path: Path, images: list[dict]) -> None:
    """Write COLMAP images.bin.  Each entry: {image_id, qw,qx,qy,qz, tx,ty,tz, camera_id, name}."""
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(images)))
        for img in images:
            f.write(struct.pack("<I",  img["image_id"]))
            f.write(struct.pack("<4d", img["qw"], img["qx"], img["qy"], img["qz"]))
            f.write(struct.pack("<3d", img["tx"],  img["ty"],  img["tz"]))
            f.write(struct.pack("<I",  img["camera_id"]))
            f.write(img["name"].encode() + b"\x00")
            f.write(struct.pack("<Q", 0))   # 0 point observations


def _write_points3d_bin_empty(path: Path) -> None:
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", 0))


def _subsample_points3d_bin(src: Path, dst: Path, max_points: int) -> int:
    """
    Read COLMAP points3D.bin, randomly subsample to max_points, write to dst.
    Assumes track_length == 0 for all points (as written by our pipeline).
    Returns actual number of points written.
    """
    with open(src, "rb") as f:
        n_total = struct.unpack("<Q", f.read(8))[0]
        if n_total == 0:
            _write_points3d_bin_empty(dst)
            return 0

        # Each point (track_len=0): 8+24+3+8+8 = 51 bytes
        data = f.read()

    # Parse all records as a structured numpy array for fast sampling.
    # Use raw bytes then index — safe because track_len=0 means fixed stride.
    RECORD_BYTES = 51
    expected = n_total * RECORD_BYTES
    if len(data) != expected:
        # Fall back: just copy if layout doesn't match expectation
        import shutil as _sh
        _sh.copy2(src, dst)
        return n_total

    if n_total <= max_points:
        import shutil as _sh
        _sh.copy2(src, dst)
        return n_total

    rng = np.random.default_rng(42)
    indices = rng.choice(n_total, size=max_points, replace=False)
    indices.sort()

    records = np.frombuffer(data, dtype=np.uint8).reshape(n_total, RECORD_BYTES)
    sampled = records[indices]

    with open(dst, "wb") as f:
        f.write(struct.pack("<Q", max_points))
        for new_id, row in enumerate(sampled):
            f.write(struct.pack("<Q", new_id + 1))   # re-index from 1
            f.write(row[8:].tobytes())               # xyz(24) + rgb(3) + err(8) + track(8)
    return max_points


# ── Math helpers ──────────────────────────────────────────────────────────────

def _quat_to_rotmat(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    from scipy.spatial.transform import Rotation
    return Rotation.from_quat([qx, qy, qz, qw]).as_matrix()


def _focal_from_params(model: str, params: list[float]) -> float:
    if model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL"):
        return params[0]          # f
    if model in ("PINHOLE", "OPENCV"):
        return (params[0] + params[1]) / 2   # (fx + fy) / 2
    return params[0]


def _compute_bounds(
    R_c2w: np.ndarray,
    cam_pos: np.ndarray,
    pts_xyz: np.ndarray | None,
    default_near: float = 0.05,
    default_far:  float = 100.0,
) -> tuple[float, float]:
    """Project 3D points into camera frame; return near/far depth bounds.

    Uses the 5th/95th percentile rather than 0.5th/99.5th because the
    long tail at very small depth is contaminated by spurious sparse
    features (lens flare, sensor glare, dust) that triangulate to ~5cm
    from the camera but aren't real scene geometry. Tighter percentile
    keeps `near` close to actual scene content (~5-10 m for typical
    indoor multi-cam rigs), matching what reference N3V calibrations
    use and what 4DGS expects for its frustum / depth sampling.
    """
    if pts_xyz is None or len(pts_xyz) == 0:
        return default_near, default_far

    R_w2c  = R_c2w.T
    t_w2c  = -R_w2c @ cam_pos
    depths = (R_w2c @ pts_xyz.T)[2] + t_w2c[2]   # Z in OpenCV = forward depth
    depths = depths[depths > 0]

    if len(depths) == 0:
        return default_near, default_far

    near = max(float(np.percentile(depths,  5.0)), default_near)
    far  = float(np.percentile(depths, 95.0))
    return near, far


# ── STG source preparation ────────────────────────────────────────────────────

def prepare_stg_source(
    sparse_dir: Path,
    cam_dirs: list[Path],
    out_dir: Path,
    *,
    max_init_points: int = 20_000,
) -> Path:
    """
    Build the colmap_N/ directory structure SpacetimeGaussians expects.

    STG's colmap loader works as follows:
      - Read colmap_0/sparse/0/images.bin → N entries, one per CAMERA (no frame index)
        with names like "cam_00.jpg", "cam_01.jpg", …
      - For each temporal step T, load images from colmap_T/images/ using the SAME
        camera names: colmap_T/images/cam_00.jpg = frame T of camera 0.

    Layout produced:
        out_dir/
            colmap_T/ (T = 0 … n_frames-1)
                sparse/0/
                    cameras.bin  → symlink to master cameras.bin
                    images.bin   → N entries with camera-only names (cam_NN.jpg)
                    points3D.bin → empty
                images/
                    cam_00.jpg   → symlink to frame T of camera 0
                    cam_01.jpg   → …
            poses_bounds.npy  → LLFF near/far bounds (read from master sparse/)

    Returns out_dir (pass as source_path to train_stg).
    """
    import re as _re
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect frame paths per camera, sorted by frame index
    sorted_cam_dirs = sorted(cam_dirs)
    cam_frames: list[list[Path]] = []
    for cam_dir in sorted_cam_dirs:
        frames = sorted(cam_dir.glob("frame_*.jpg"))
        if frames:
            cam_frames.append(frames)

    if not cam_frames:
        raise FileNotFoundError(f"No frame_*.jpg files found under {cam_dirs}")

    n_frames = min(len(f) for f in cam_frames)

    # Read master images.bin → extract one pose record per physical camera.
    # Names may be "cam_00_frame_000000.jpg" or "cam_00.jpg" (camera-only).
    # Group by camera prefix (everything before the first "_frame_" or before ".").
    _PREFIX = _re.compile(r"^(.+?)(?:_frame_\d+)?\.(?:jpg|png)$")
    master_images = _read_images_bin(sparse_dir / "images.bin")
    cam_pose_by_key: dict[str, dict] = {}
    for rec in sorted(master_images.values(), key=lambda r: r["image_id"]):
        m = _PREFIX.match(rec["name"])
        key = m.group(1) if m else rec["name"].rsplit(".", 1)[0]
        cam_pose_by_key.setdefault(key, rec)   # keep first occurrence per camera

    # Pair each cam_dir with its pose by name — handles non-contiguous subsets correctly.
    # cam_dir.name ("cam_05") must match a key in cam_pose_by_key ("cam_05").
    cam_with_poses: list[tuple[Path, list[Path], dict]] = []
    for cam_dir, frames in zip(sorted_cam_dirs, cam_frames):
        pose = cam_pose_by_key.get(cam_dir.name)
        if pose is not None:
            cam_with_poses.append((cam_dir, frames, pose))

    n = len(cam_with_poses)
    console.print(f"  [dim]Building STG source: {n} cameras × {n_frames} frames → {out_dir}[/]")

    # Build a subsampled points3D.bin so STG's knn() cdist stays within GPU memory.
    # Full 194K-point cloud → N²×4 bytes ≈ 140 GB; 20K → 1.6 GB (safe on 16 GB GPU).
    master_pts_src = (sparse_dir / "points3D.bin").resolve()
    stg_pts = out_dir / "points3D_stg.bin"
    if not stg_pts.exists():
        if master_pts_src.exists() and master_pts_src.stat().st_size > 8:
            n_written = _subsample_points3d_bin(master_pts_src, stg_pts, max_init_points)
            console.print(f"  [dim]Init points: {n_written:,} (capped from full cloud for GPU)[/]")
        else:
            _write_points3d_bin_empty(stg_pts)
    master_pts = stg_pts.resolve()

    for t in range(n_frames):
        col_dir = out_dir / f"colmap_{t}"
        img_dir = col_dir / "images"
        img_dir.mkdir(parents=True, exist_ok=True)

        frame_sparse = col_dir / "sparse" / "0"
        frame_sparse.mkdir(parents=True, exist_ok=True)

        # cameras.bin: symlink to master (shared intrinsics)
        cam_link = frame_sparse / "cameras.bin"
        if not cam_link.exists():
            cam_link.symlink_to((sparse_dir / "cameras.bin").resolve())

        # points3D.bin: symlink to the subsampled cloud built above
        pts_path = frame_sparse / "points3D.bin"
        if not pts_path.exists():
            pts_path.symlink_to(master_pts)

        # images.bin: one entry per camera with camera-only name ("cam_00.jpg")
        # Pose matched by cam_dir.name so non-contiguous subsets work correctly.
        frame_records = []
        for cam_idx, (cam_dir, frames, pose) in enumerate(cam_with_poses):
            frame_records.append({
                "image_id":  cam_idx + 1,
                "qw": pose["qw"], "qx": pose["qx"],
                "qy": pose["qy"], "qz": pose["qz"],
                "tx": pose["tx"], "ty": pose["ty"], "tz": pose["tz"],
                "camera_id": pose["camera_id"],
                "name": f"{cam_dir.name}.jpg",   # "cam_00.jpg"
            })
        _write_images_bin(frame_sparse / "images.bin", frame_records)

        # images/cam_NN.jpg → symlink to actual frame T image for that camera
        for cam_dir, frames, _ in cam_with_poses:
            src = frames[t]
            dst = img_dir / f"{cam_dir.name}.jpg"
            if not dst.exists():
                dst.symlink_to(src.resolve())

    # poses_bounds.npy — near/far depth bounds; STG expects it one level above colmap_0/.
    # Use colmap_0/sparse/0 (15 training cameras, camera-only names) so the viewer orbit
    # only visits positions the model was actually trained on — not all 27 in the master.
    poses_dst = out_dir / "poses_bounds.npy"
    if not poses_dst.exists():
        colmap_0_sparse = out_dir / "colmap_0" / "sparse" / "0"
        colmap_to_poses_bounds(colmap_0_sparse, poses_dst)

    console.print(f"  [green]STG source ready: {n_frames} colmap_N/ dirs → {out_dir}[/]")
    return out_dir


# ── transforms_train.json source preparation (Gaussian-Flow + 4D-Rotor) ────────

def prepare_transforms_source(
    sparse_dir: Path,
    cam_dirs: list[Path],
    out_dir: Path,
) -> Path:
    """
    Build a nerfstudio-style transforms_train.json from COLMAP output.

    Reads camera intrinsics from sparse_dir/cameras.bin and per-image
    extrinsics from sparse_dir/images.bin.  Associates images with
    (camera, frame) pairs by parsing the image names written by calibrate.py
    (format: cam_{NN}_frame_{NNNNNN}.jpg).

    Layout produced:
        out_dir/
            transforms_train.json
            transforms_test.json   (every 8th frame held out)
            images/
                cam_00/ → symlinks to original frame files
                cam_01/ → ...
                …

    Returns out_dir (pass this as source_path to train_gflow / train_4drotor).
    """
    cameras_bin = sparse_dir / "cameras.bin"
    images_bin  = sparse_dir / "images.bin"
    if not cameras_bin.exists() or not images_bin.exists():
        raise FileNotFoundError(f"COLMAP binaries not found in {sparse_dir}")

    cameras = _read_cameras_bin(cameras_bin)
    images  = _read_images_bin(images_bin)

    # Group COLMAP image records by (cam_prefix, frame_idx)
    # image name format from calibrate.py: cam_00_frame_000000.jpg
    import re
    _PAT = re.compile(r"^(cam_\d+)_frame_(\d+)\.")
    grouped: dict[str, dict[int, dict]] = {}   # cam_prefix → {frame_idx: image_record}
    for rec in images.values():
        m = _PAT.match(rec["name"])
        if not m:
            continue
        cam_prefix  = m.group(1)
        frame_idx   = int(m.group(2))
        grouped.setdefault(cam_prefix, {})[frame_idx] = rec

    if not grouped:
        raise ValueError(
            f"No images matching cam_NN_frame_NNNNNN.jpg found in {images_bin}. "
            "Run calibrate first."
        )

    cam_prefixes = sorted(grouped.keys())
    all_frame_idxs = sorted({fi for cam in grouped.values() for fi in cam.keys()})
    n_frames = len(all_frame_idxs)

    # Pick intrinsics from first camera record
    first_cam_rec = cameras[next(iter(images.values()))["camera_id"]]
    W       = first_cam_rec["width"]
    H       = first_cam_rec["height"]
    params  = first_cam_rec["params"]
    model   = first_cam_rec["model"]
    focal   = _focal_from_params(model, params)
    fl_x    = params[0] if model in ("PINHOLE", "OPENCV") else focal
    fl_y    = params[1] if model in ("PINHOLE", "OPENCV") else focal
    cx      = params[2] if len(params) > 2 else W / 2.0
    cy      = params[3] if len(params) > 3 else H / 2.0

    # Build images/ symlink tree
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Map from cam_prefix to source cam_dir
    cam_dir_map: dict[str, Path] = {}
    for cam_dir in sorted(cam_dirs):
        prefix = cam_dir.name  # e.g. cam_00
        cam_dir_map[prefix] = cam_dir

    for cam_prefix in cam_prefixes:
        link_dir = images_dir / cam_prefix
        link_dir.mkdir(exist_ok=True)
        src_cam  = cam_dir_map.get(cam_prefix)
        if src_cam is None:
            continue
        frame_src_dir = src_cam / "images" if (src_cam / "images").exists() else src_cam
        for frame_path in sorted(frame_src_dir.glob("frame_*.jpg")):
            link = link_dir / frame_path.name
            if not link.exists():
                link.symlink_to(frame_path.resolve())

    # Build frames list
    frames_all: list[dict] = []
    for fi, frame_idx in enumerate(all_frame_idxs):
        t = fi / max(n_frames - 1, 1)
        for cam_prefix in cam_prefixes:
            rec = grouped[cam_prefix].get(frame_idx)
            if rec is None:
                continue
            R = _quat_to_rotmat(rec["qw"], rec["qx"], rec["qy"], rec["qz"])
            t_vec = np.array([rec["tx"], rec["ty"], rec["tz"]])
            # Convert w2c (OpenCV) → c2w (column-major 4×4)
            c2w = np.eye(4)
            c2w[:3, :3] = R.T
            c2w[:3,  3] = -(R.T @ t_vec)
            # OpenCV → OpenGL coordinate flip (nerfstudio convention)
            c2w[:3, 1] *= -1
            c2w[:3, 2] *= -1
            frames_all.append({
                "file_path": f"images/{cam_prefix}/frame_{frame_idx:06d}.jpg",
                "transform_matrix": c2w.tolist(),
                "time": t,
            })

    # Split train/test (every 8th frame held out for test)
    frames_train = [f for i, f in enumerate(frames_all) if i % 8 != 0]
    frames_test  = [f for i, f in enumerate(frames_all) if i % 8 == 0]

    base = {
        "w": W, "h": H,
        "fl_x": fl_x, "fl_y": fl_y,
        "cx": cx, "cy": cy,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "transforms_train.json").write_text(
        json.dumps({**base, "frames": frames_train}, indent=None)
    )
    (out_dir / "transforms_test.json").write_text(
        json.dumps({**base, "frames": frames_test}, indent=None)
    )

    console.print(
        f"  [green]transforms_train.json: {len(frames_train)} frames "
        f"({len(cam_prefixes)} cameras × {n_frames} steps) → {out_dir}[/]"
    )
    return out_dir


# ── Pointrix Gaussian-Flow source preparation ────────────────────────────────────

def prepare_gflow_pointrix_source(
    sparse_dir: Path,
    cam_dirs: list[Path],
    out_dir: Path,
    *,
    iterations: int = 30_000,
    model_path: Path,
    scale: float = 0.5,
) -> tuple[Path, Path]:
    """
    Build a Pointrix Gaussian-Flow source directory from COLMAP output.

    Reads sparse/0/ and cam_XX/ frame dirs then produces:
        out_dir/
            sparse/0/
                cameras.bin   (copied from sparse_dir)
                images.bin    (rewritten: frame_{T*n_cams+c+1:06d}.jpg names)
                points3D.bin  (symlinked / empty)
            images/
                frame_000001.jpg → symlink to actual frame
                ...
        gflow_config.yaml

    Images use frame-major ordering so all cameras at the same real
    time step T have nearly-equal time values (≈ T/n_frames).

    Returns (out_dir, config_yaml_path).
    """
    import re

    cameras_bin = sparse_dir / "cameras.bin"
    images_bin  = sparse_dir / "images.bin"
    if not cameras_bin.exists() or not images_bin.exists():
        raise FileNotFoundError(f"COLMAP binaries not found in {sparse_dir}")

    images_map = _read_images_bin(images_bin)

    _PAT = re.compile(r"^(cam_\d+)_frame_(\d+)\.")
    grouped: dict[str, dict[int, dict]] = {}
    for rec in images_map.values():
        m = _PAT.match(rec["name"])
        if not m:
            continue
        grouped.setdefault(m.group(1), {})[int(m.group(2))] = rec

    if not grouped:
        raise ValueError(
            f"No images matching cam_NN_frame_NNNNNN.jpg in {images_bin}. "
            "Run calibration first."
        )

    cam_prefixes = sorted(grouped.keys())
    frame_idxs   = sorted({fi for cam in grouped.values() for fi in cam.keys()})
    n_cams   = len(cam_prefixes)
    cam_dir_map: dict[str, Path] = {d.name: d for d in sorted(cam_dirs)}

    pointrix_sparse = out_dir / "sparse" / "0"
    pointrix_sparse.mkdir(parents=True, exist_ok=True)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(cameras_bin, pointrix_sparse / "cameras.bin")

    dst_pts = pointrix_sparse / "points3D.bin"
    if not dst_pts.exists():
        src_pts = sparse_dir / "points3D.bin"
        if src_pts.exists():
            dst_pts.symlink_to(src_pts.resolve())
        else:
            _write_points3d_bin_empty(dst_pts)

    new_images: list[dict] = []
    new_img_id = 1

    for T_idx, frame_idx in enumerate(frame_idxs):
        for c_idx, cam_prefix in enumerate(cam_prefixes):
            rec = grouped[cam_prefix].get(frame_idx)
            if rec is None:
                continue
            global_seq = T_idx * n_cams + c_idx + 1   # 1-indexed
            new_name = f"frame_{global_seq:06d}.jpg"

            new_images.append({
                "image_id":  new_img_id,
                "qw": rec["qw"], "qx": rec["qx"],
                "qy": rec["qy"], "qz": rec["qz"],
                "tx": rec["tx"], "ty": rec["ty"], "tz": rec["tz"],
                "camera_id": rec["camera_id"],
                "name": new_name,
            })
            new_img_id += 1

            dst_link = images_dir / new_name
            if not dst_link.exists():
                cam_dir = cam_dir_map.get(cam_prefix)
                if cam_dir is not None:
                    src_frame = cam_dir / f"frame_{frame_idx:06d}.jpg"
                    if src_frame.exists():
                        dst_link.symlink_to(src_frame.resolve())

    _write_images_bin(pointrix_sparse / "images.bin", new_images)

    console.print(
        f"  [dim]Pointrix GFlow source: {n_cams} cams × {len(frame_idxs)} frames "
        f"({len(new_images)} images) → {out_dir}[/]"
    )

    # Check how many sparse points are available for Pointrix init
    dst_pts = pointrix_sparse / "points3D.bin"
    n_sparse = 0
    if dst_pts.exists():
        with open(dst_pts, "rb") as _f:
            raw = _f.read(8)
            if len(raw) == 8:
                import struct as _struct
                n_sparse = _struct.unpack("<Q", raw)[0]

    config_path = out_dir / "gflow_config.yaml"
    if n_sparse == 0:
        console.print(
            "  [yellow]No sparse points — Pointrix will use random initialisation "
            "(100K pts). Expect slower convergence.[/]"
        )
    _write_gflow_pointrix_config(
        config_path,
        data_path=out_dir,
        exp_dir=model_path,
        iterations=iterations,
        scale=scale,
        has_sparse_pts=(n_sparse > 0),
    )

    return out_dir, config_path


def _write_gflow_pointrix_config(
    config_path: Path,
    *,
    data_path: Path,
    exp_dir: Path,
    iterations: int,
    scale: float = 0.5,
    has_sparse_pts: bool = True,
) -> None:
    cfg = f"""\
name: "gflow_train"
exp_dir: "{exp_dir.resolve()}"
use_timestamp: false

trainer:
  output_path: "{exp_dir.resolve()}"
  max_steps: {iterations}
  val_interval: {max(iterations // 3, 10000)}
  training: true
  enable_gui: false

  spatial_lr_scale: true
  model:
    name: GaussianFlow
    lambda_ssim: 0.2
    point_cloud:
      point_cloud_type: "GaussianFlowPointCloud"
      trainable: true
      unwarp_prefix: "point_cloud"
      max_sh_degree: 3
      pos_traj_dim: 3
      rot_traj_dim: 3
      feat_traj_dim: 3
      rescale_value: 0.7
      offset_value: 0.1
      initializer:
        init_type: "{'colmap' if has_sparse_pts else 'random'}"
        feat_dim: 3
        num_points: 100000
        radius: 5.0
    camera_model:
      name: TimeCameraModel
      enable_training: false
    renderer:
      name: "GaussianFlowRender"
      render_depth: false
      update_sh_iter: 1000
      max_sh_degree: 3

  controller:
    name: GFDensificationController
    normalize_grad: false
    control_module: "point_cloud"
    densify_start_iter: 500
    densify_stop_iter: 15000
    prune_interval: 100
    duplicate_interval: 100
    opacity_reset_interval: 30001
    densify_grad_threshold: 0.0002
    min_opacity: 0.001
    max_points: 120000
    optimizer_name: "optimizer_1"

  gui:
    name: GaussianFlowGUI
    viewer_port: 9012

  optimizer:
    optimizer_1:
      type: BaseOptimizer
      name: Adam
      args:
        eps: 1.0e-15
      extra_cfg:
        backward: false
      params:
        point_cloud.position:
          lr: 0.000016
        point_cloud.pos_params:
          lr: 0.00005
        point_cloud.features:
          lr: 0.0025
        point_cloud.features_rest:
          lr: 0.000125
        point_cloud.feat_params:
          lr: 0.0025
        point_cloud.scaling:
          lr: 0.005
        point_cloud.rotation:
          lr: 0.001
        point_cloud.rot_params:
          lr: 0.005
        point_cloud.opacity:
          lr: 0.05
        point_cloud.time_center:
          lr: 0.001

  scheduler:
    name: "ExponLRScheduler"
    params:
      point_cloud.position:
        init:  0.000016
        final: 0.00000016
        max_steps: {iterations}
      point_cloud.pos_params:
        init:  0.00005
        final: 0.0000005
        max_steps: {iterations}

  datapipeline:
    data_set: "CustomDataset"
    shuffle: true
    batch_size: 1
    num_workers: 0
    dataset:
      data_path: "{data_path.resolve()}"
      cached_observed_data: true
      scale: {scale}
      white_bg: false
      observed_data_dirs_dict:
        image: images

  writer:
    writer_type: "TensorboardWriter"

  hooks:
    LogHook:
      name: LogHook
    CheckPointHook:
      name: CheckPointHook

  exporter:
    exporter_1:
      type: MetricExporter
"""
    config_path.write_text(cfg)


# ── ffmpeg discovery (mirrors render_orbit.py) ─────────────────────────────────

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
