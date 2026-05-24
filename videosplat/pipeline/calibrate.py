from __future__ import annotations

"""
Camera calibration for multi-camera video capture.

Two backends:

  1. COLMAP (default) — runs COLMAP feature extraction + exhaustive matching
     across all cameras simultaneously, yielding a single sparse/0/ reconstruction
     that contains all camera poses in a common world frame.

  2. MASt3R — learned dense-matching alternative for low-texture / repetitive
     scenes where COLMAP fails. Optional pycolmap-based refinement pass.

Output: <out_dir>/sparse/0/{cameras,images,points3D}.bin  (COLMAP standard)
"""

import json
import struct
from pathlib import Path

import numpy as np
from rich.console import Console

console = Console()

# Default number of frames per camera for the static-rig COLMAP path. The rig
# geometry is constant across frames, so a small evenly-spread sample is enough;
# more frames add temporal feature tracks (better-constrained poses) but also
# re-introduce moving foreground. Overridable via calibrate(calib_frames=...).
_COLMAP_RIG_FRAMES = 15


# ── Public API ──────────────────────────────────────────────────────────────────

def calibrate(
    cam_dirs: list[Path],
    out_dir: Path,
    *,
    method: str = "colmap",
    camera_model: str = "SIMPLE_PINHOLE",
    fps: float = 10.0,
    static_cameras: bool = False,
    calib_frames: int | None = None,
    mast3r_dir: Path | None = None,
    mast3r_niter: int = 500,
    mast3r_refine: bool = False,
    mast3r_image_size: int = 512,
) -> dict:
    """
    Estimate camera intrinsics + extrinsics for a multi-camera rig.

    Args:
        cam_dirs:            List of per-camera frame directories (cam_00/, ...).
        out_dir:             Workspace root; sparse/0/ is created here.
        method:              'colmap' (default) or 'mast3r'.
        camera_model:        COLMAP camera model (only used in 'colmap' mode).
        static_cameras:      MASt3R: cameras are fixed; sample a few frames per
                             camera and replicate the averaged pose to all frames.
        calib_frames:        MASt3R: frames per camera to sample. Static default: 3.
                             Moving default: None (use all frames).
        mast3r_dir:          Path to the MASt3R repo clone.
        mast3r_niter:        MASt3R global-alignment iterations.
        mast3r_refine:       If True, refine MASt3R poses+intrinsics via pycolmap
                             (SIFT extract → match → triangulate + BA). Fails
                             soft (keeps MASt3R output) if too few features.
                             Default False because empirically refinement HURTS
                             low-texture scenes — SIFT finds too few matches
                             and BA pulls poses to a worse local minimum than
                             MASt3R's dense-matching prior. Only enable when
                             SIFT consistently finds 100+ matches per pair
                             (textured outdoor / object-centric scenes).

    Returns:
        dict with n_cameras, n_registered, n_points3D, sparse_dir
    """
    sparse_dir = out_dir / "sparse" / "0"
    sparse_dir.mkdir(parents=True, exist_ok=True)

    if method == "mast3r":
        meta = _run_mast3r(
            cam_dirs, out_dir, sparse_dir,
            static_cameras=static_cameras,
            calib_frames=calib_frames,
            mast3r_dir=mast3r_dir,
            mast3r_niter=mast3r_niter,
            mast3r_refine=mast3r_refine,
            mast3r_image_size=mast3r_image_size,
        )
    else:
        meta = _run_colmap(cam_dirs, out_dir, sparse_dir, camera_model, calib_frames)

    # Convert COLMAP-format output → poses_bounds.npy + cam*.mp4 so train.py works
    from videosplat.pipeline.convert import colmap_to_poses_bounds, frames_to_videos
    try:
        colmap_to_poses_bounds(sparse_dir, out_dir / "poses_bounds.npy")
    except Exception as e:
        console.print(f"  [yellow]poses_bounds.npy generation failed: {e}[/]")

    try:
        frames_to_videos(cam_dirs, out_dir, fps=fps)
    except Exception as e:
        console.print(f"  [yellow]frames_to_videos failed: {e}[/]")

    return meta


# ── MASt3R calibration ──────────────────────────────────────────────────────────

def _run_mast3r(
    cam_dirs: list[Path],
    out_dir: Path,
    sparse_dir: Path,
    *,
    static_cameras: bool = False,
    calib_frames: int | None = None,
    mast3r_dir: Path | None = None,
    mast3r_niter: int = 500,
    mast3r_refine: bool = False,
    mast3r_image_size: int = 512,
) -> dict:
    import sys
    import torch

    mast3r_root = mast3r_dir or (Path.home() / "mast3r")
    for p in [str(mast3r_root), str(mast3r_root / "dust3r")]:
        if p not in sys.path:
            sys.path.insert(0, p)

    from mast3r.model import AsymmetricMASt3R
    from mast3r.image_pairs import make_pairs
    from dust3r.inference import inference
    from dust3r.utils.image import load_images
    from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

    device = "cuda" if torch.cuda.is_available() else "cpu"
    console.print(f"  [dim]MASt3R: device={device}[/]")

    ckpt_dir = mast3r_root / "checkpoints"
    ckpts = sorted(ckpt_dir.glob("*.pth"))
    if not ckpts:
        raise FileNotFoundError(f"No .pth checkpoint in {ckpt_dir}. Download from https://github.com/naver/mast3r")
    ckpt = ckpts[0]
    console.print(f"  [dim]MASt3R checkpoint: {ckpt.name}[/]")

    model = AsymmetricMASt3R.from_pretrained(str(ckpt)).to(device).eval()

    orig_w, orig_h = _infer_image_size(cam_dirs)
    n_cams = len(cam_dirs)

    per_cam_frames: list[list[Path]] = [sorted(d.glob("frame_*.jpg")) for d in sorted(cam_dirs)]

    # ── Collect image paths and build pairs ─────────────────────────────────────
    if static_cameras:
        # Cameras don't move but we sample N evenly-spaced frames per camera so
        # MASt3R can average out single-frame pose noise (later, in cam_groups).
        n_sample = calib_frames if calib_frames is not None else 3
        image_paths: list[str] = []
        frame_map: list[tuple[int, int]] = []
        for ci, frames in enumerate(per_cam_frames):
            if not frames:
                continue
            idxs = _evenly_spaced(len(frames), n_sample)
            for fi in idxs:
                image_paths.append(str(frames[fi]))
                frame_map.append((ci, fi))
        n_imgs = len(image_paths)
        images = load_images(image_paths, size=mast3r_image_size, verbose=False)

        if n_sample == 1:
            # Complete graph — only n_cams images.
            pairs = make_pairs(images, scene_graph="complete", symmetrize=True)
        else:
            # Sparse graph: within-camera complete (temporal pairs that constrain
            # the per-camera averaged pose) + cross-camera complete at ONE
            # representative temporal step (cross-camera structure). This keeps
            # pair count near n_cams^2 instead of (n_cams*n_sample)^2.
            # Image i in `images` corresponds to frame_map[i] = (cam, frame_idx).
            # Within each camera, the N samples are contiguous: base = ci*n_sample.
            pair_indices: list[tuple[int, int]] = []
            for ci in range(n_cams):
                base = ci * n_sample
                for a in range(n_sample):
                    for b in range(a + 1, n_sample):
                        pair_indices.append((base + a, base + b))
            mid = n_sample // 2
            for ca in range(n_cams):
                for cb in range(ca + 1, n_cams):
                    pair_indices.append((ca * n_sample + mid, cb * n_sample + mid))
            pair_indices += [(j, i) for i, j in pair_indices]  # symmetrize
            pairs = [(images[i], images[j]) for i, j in pair_indices]

        console.print(
            f"  [dim]MASt3R static: {n_sample} frame(s)/cam × {n_cams} cams"
            f" = {n_imgs} images, {len(pairs):,} pairs"
            f"{' (complete)' if n_sample == 1 else ' (sparse: within-cam + cross-cam at mid frame)'}[/]"
        )

    elif calib_frames is not None:
        n_sample = calib_frames
        image_paths = []
        frame_map = []
        for ci, frames in enumerate(per_cam_frames):
            if not frames:
                continue
            idxs = _evenly_spaced(len(frames), n_sample)
            for fi in idxs:
                image_paths.append(str(frames[fi]))
                frame_map.append((ci, fi))
        console.print(
            f"  [dim]MASt3R moving (sampled): {n_sample} frames/cam × {n_cams} cams"
            f" = {len(image_paths)} images, complete graph[/]"
        )
        images = load_images(image_paths, size=mast3r_image_size, verbose=False)
        pairs = make_pairs(images, scene_graph="complete", symmetrize=True)

    else:
        # Moving cameras — all frames, hybrid scene graph
        image_paths = []
        for frames in per_cam_frames:
            image_paths.extend(str(f) for f in frames)
        n_frames = len(per_cam_frames[0]) if per_cam_frames else 0
        console.print(
            f"  [dim]MASt3R moving (all): {n_cams} cams × {n_frames} frames"
            f" = {len(image_paths)} images[/]"
        )
        images = load_images(image_paths, size=mast3r_image_size, verbose=False)

        # Within-camera swin-5 + cross-camera pairs at K keyframes
        pairs_idx: list[tuple[int, int]] = []
        for ci in range(n_cams):
            for fi in range(n_frames):
                for delta in range(1, 6):
                    if fi + delta < n_frames:
                        pairs_idx.append((ci * n_frames + fi, ci * n_frames + fi + delta))
        K = max(2, n_frames // 10)
        kf_idxs = _evenly_spaced(n_frames, K)
        for kf in kf_idxs:
            for ca in range(n_cams):
                for cb in range(ca + 1, n_cams):
                    pairs_idx.append((ca * n_frames + kf, cb * n_frames + kf))
        pairs_idx += [(j, i) for i, j in pairs_idx]  # symmetrize
        pairs = [(images[i], images[j]) for i, j in pairs_idx]
        console.print(f"  [dim]Scene graph: {len(pairs):,} pairs[/]")
        frame_map = None  # all frames in flat cam-major order

    # ── Inference ───────────────────────────────────────────────────────────────
    console.print(f"  [dim]MASt3R inference ({len(pairs):,} pairs)…[/]")
    output = _mast3r_chunked_inference(pairs, model, device)

    # Free the model now — it is not needed for global alignment and holds ~2.7 GB.
    del model
    torch.cuda.empty_cache()

    # ── Global alignment ────────────────────────────────────────────────────────
    def _align(out, dev, mode, niter):
        scene = global_aligner(out, dev, mode=mode, verbose=False)
        scene.compute_global_alignment(init="mst", niter=niter, schedule="cosine", lr=0.01)
        return scene

    # Pick an initial optimizer that fits in memory. PointCloudOptimizer stores
    # the full _stacked_pred_i/j tensors on GPU and OOMs above ~900 pairs on
    # typical 8-16 GB laptop GPUs. Threshold deliberately conservative — the
    # cost of trying-and-failing is ~30s plus a fall to slow CPU alignment that
    # gives meaningfully worse poses.
    n_pairs = len(pairs)
    if n_pairs > 900:
        ladder = [
            (device, GlobalAlignerMode.ModularPointCloudOptimizer, "GPU Modular"),
            ("cpu",  GlobalAlignerMode.PointCloudOptimizer,        "CPU PointCloud"),
        ]
    else:
        ladder = [
            (device, GlobalAlignerMode.PointCloudOptimizer,        "GPU PointCloud"),
            (device, GlobalAlignerMode.ModularPointCloudOptimizer, "GPU Modular"),
            ("cpu",  GlobalAlignerMode.PointCloudOptimizer,        "CPU PointCloud"),
        ]

    console.print(
        f"  [dim]MASt3R global alignment (niter={mast3r_niter}, "
        f"start={ladder[0][2]}, pairs={n_pairs:,})…[/]"
    )
    scene = None
    last_label = ladder[-1][2]
    for attempt, (dev, mode, label) in enumerate(ladder):
        try:
            if attempt > 0:
                console.print(f"  [yellow]OOM — retrying with {label} optimizer[/]")
                scene = None
                torch.cuda.empty_cache()
            scene = _align(output, dev, mode, niter=mast3r_niter)
            break
        except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:
            if "out of memory" not in str(exc).lower() and not isinstance(exc, torch.cuda.OutOfMemoryError):
                raise
            if label == last_label:
                raise RuntimeError("MASt3R global alignment OOM on every fallback") from exc

    # ── Extract poses, intrinsics, and dense cloud ──────────────────────────────
    with torch.no_grad():
        im_poses   = scene.get_im_poses().cpu().numpy()        # [N, 4, 4] c2w
        focals     = scene.get_focals().cpu().numpy().flatten()  # [N]
        pps        = scene.get_principal_points().cpu().numpy()  # [N, 2]
        pts3d_list = scene.get_pts3d()
        conf_list  = scene.get_conf()

    # Adaptive threshold: CPU optimizer produces lower scores than GPU.
    xyz_all = np.zeros((0, 3), dtype=np.float64)
    rgb_all = np.zeros((0, 3), dtype=np.uint8)
    conf_thr_used = 1.5
    for conf_thr in (1.5, 1.0, 0.5, 0.1):
        all_xyz: list[np.ndarray] = []
        all_rgb: list[np.ndarray] = []
        for img_dict, pts, conf in zip(images, pts3d_list, conf_list):
            mask  = conf.detach().cpu().numpy().reshape(-1) > conf_thr
            xyz   = pts.detach().cpu().numpy().reshape(-1, 3)[mask]
            img_t = img_dict["img"]  # [1, 3, H, W] in [-1, 1]
            rgb   = ((img_t[0].permute(1, 2, 0).cpu().numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
            all_xyz.append(xyz)
            all_rgb.append(rgb.reshape(-1, 3)[mask])
        xyz_all = np.concatenate(all_xyz, axis=0) if all_xyz else np.zeros((0, 3))
        rgb_all = np.concatenate(all_rgb, axis=0) if all_rgb else np.zeros((0, 3), dtype=np.uint8)
        conf_thr_used = conf_thr
        if len(xyz_all) > 0:
            break
    console.print(f"  [dim]Dense cloud: {len(xyz_all):,} points (conf > {conf_thr_used})[/]")

    # ── Scene-scale normalisation ────────────────────────────────────────────────
    # 4DGS computes cameras_extent (and many densification thresholds) from the
    # camera-center spread. MASt3R's metric output can land anywhere, so rescale
    # so the bounding radius is 1.0 — same regime COLMAP usually lives in.
    TARGET_RADIUS = 1.0
    centers = im_poses[:, :3, 3]
    center_mean = centers.mean(axis=0)
    distances = np.linalg.norm(centers - center_mean, axis=1)
    scene_radius = float(distances.max())
    if scene_radius > 1e-6:
        scale_factor = TARGET_RADIUS / scene_radius
        im_poses[:, :3, 3] *= scale_factor
        if len(xyz_all):
            xyz_all = xyz_all * scale_factor
        console.print(
            f"  [dim]Scene normalised: radius {scene_radius:.3f} → "
            f"{TARGET_RADIUS:.3f} (scale {scale_factor:.4f})[/]"
        )
    else:
        console.print(f"  [yellow]Degenerate scene radius ({scene_radius:.3e}); skipping rescale[/]")

    # Sanity check: camera-to-point-cloud distance percentiles. Cameras should
    # sit within roughly the same volume as the points.
    if len(xyz_all):
        rescaled_centers = im_poses[:, :3, 3]
        # subsample points for cheaper percentile compute
        pts_sample = xyz_all[np.random.default_rng(0).choice(
            len(xyz_all), size=min(5000, len(xyz_all)), replace=False
        )]
        # distance from each camera to the nearest few points (median ~ scene scale)
        d_min = np.linalg.norm(
            pts_sample[None, :, :] - rescaled_centers[:, None, :], axis=-1
        ).min(axis=1)
        p10, p50, p90 = np.percentile(d_min, [10, 50, 90])
        console.print(
            f"  [dim]Camera→points (post-rescale): p10={p10:.3f} p50={p50:.3f} p90={p90:.3f}[/]"
        )

    # Scale from MASt3R 512px-space to original resolution (intrinsics only —
    # focals/pps live in pixel space, untouched by the world-space rescale).
    mast3r_h, mast3r_w = int(images[0]["true_shape"][0][0]), int(images[0]["true_shape"][0][1])
    scale = max(orig_h, orig_w) / max(mast3r_h, mast3r_w)

    med_focal = float(np.median(focals)) * scale
    med_cx    = float(np.median(pps[:, 0])) * scale
    med_cy    = float(np.median(pps[:, 1])) * scale

    # ── Optional pycolmap refinement (SIFT extract → match → triangulate + BA) ──
    # Refines per-sampled-image poses + intrinsics + sparse points using SIFT
    # matches that MASt3R alone may have missed. Only runs in modes where the
    # sampled image set is small enough for exhaustive matching (static, or
    # moving with --calib-frames sampling). Fail-soft.
    refinement_eligible = static_cameras or (calib_frames is not None)
    if mast3r_refine and refinement_eligible:
        refine_work = out_dir / "_mast3r_refine"
        refined = _refine_mast3r_with_colmap(
            sampled_image_paths=image_paths,
            sampled_im_poses=im_poses,
            focal_init=med_focal,
            cx_init=med_cx,
            cy_init=med_cy,
            orig_w=orig_w,
            orig_h=orig_h,
            work_dir=refine_work,
        )
        if refined is not None:
            # Take refined poses + intrinsics — these are the high-value output
            # of BA. Keep MASt3R's dense xyz_all/rgb_all for 4DGS init though:
            # SIFT triangulation on low-texture scenes (basketball dome) yields
            # too few points (e.g., 325) for a useful 4DGS initialisation.
            # Refinement only nudges poses slightly so the dense cloud stays
            # ~aligned in the same gauge.
            im_poses  = refined["im_poses"]
            med_focal = refined["focal"]
            med_cx    = refined["cx"]
            med_cy    = refined["cy"]
            console.print(
                f"  [bold green]✓ Refinement applied — refined poses + intrinsics. "
                f"Keeping MASt3R dense cloud ({len(xyz_all):,} pts) for init.[/]"
            )
        else:
            console.print(
                "  [bold yellow]× Refinement returned no result — using MASt3R output as-is.[/]"
            )
    elif mast3r_refine and not refinement_eligible:
        console.print(
            "  [dim]Refinement skipped: --calib-frames not set in moving mode "
            "(would require exhaustive matching on all frames).[/]"
        )

    console.print(
        f"  [dim]Intrinsics: focal={med_focal:.1f}px  cx={med_cx:.1f}  cy={med_cy:.1f}"
        f"  (original {orig_w}×{orig_h})[/]"
    )

    # ── Write cameras.bin ────────────────────────────────────────────────────────
    _write_cameras_bin(sparse_dir / "cameras.bin", {
        1: {"camera_id": 1, "model": "PINHOLE", "width": orig_w, "height": orig_h,
            "params": [med_focal, med_focal, med_cx, med_cy]},
    })

    # ── Write images.bin ─────────────────────────────────────────────────────────
    from scipy.spatial.transform import Rotation as _Rot

    colmap_images: dict[int, dict] = {}
    img_id = 0

    def _c2w_to_colmap(c2w: np.ndarray) -> dict:
        """Convert 4×4 c2w → COLMAP quaternion + translation."""
        R_c2w = c2w[:3, :3]
        t_c2w = c2w[:3, 3]
        R_w2c = R_c2w.T
        t_w2c = -R_w2c @ t_c2w
        q = _rotation_to_quaternion(R_w2c)
        return {"qw": q[0], "qx": q[1], "qy": q[2], "qz": q[3],
                "tx": t_w2c[0], "ty": t_w2c[1], "tz": t_w2c[2]}

    if static_cameras:
        # Average N sampled poses per physical camera, then replicate to all frames
        cam_groups: dict[int, list[int]] = {}
        for mi, (ci, _) in enumerate(frame_map):
            cam_groups.setdefault(ci, []).append(mi)

        avg_c2w: dict[int, np.ndarray] = {}
        for ci, mis in cam_groups.items():
            c2ws = im_poses[mis]
            avg_rot = _Rot.from_matrix(c2ws[:, :3, :3]).mean().as_matrix()
            avg_t   = c2ws[:, :3, 3].mean(axis=0)
            m = np.eye(4)
            m[:3, :3] = avg_rot
            m[:3, 3]  = avg_t
            avg_c2w[ci] = m

        for ci, frames in enumerate(per_cam_frames):
            pose = _c2w_to_colmap(avg_c2w.get(ci, np.eye(4)))
            for frame_path in frames:
                img_id += 1
                colmap_images[img_id] = {
                    "image_id": img_id,
                    **pose,
                    "camera_id": 1,
                    "name": f"{frame_path.parent.name}_{frame_path.name}",
                }

    elif calib_frames is not None:
        # Moving sampled: each sampled frame gets its own MASt3R pose
        for mi, (ci, fi) in enumerate(frame_map):
            pose = _c2w_to_colmap(im_poses[mi])
            frame_path = per_cam_frames[ci][fi]
            img_id += 1
            colmap_images[img_id] = {
                "image_id": img_id,
                **pose,
                "camera_id": 1,
                "name": f"{frame_path.parent.name}_{frame_path.name}",
            }

    else:
        # Moving all frames: flat cam-major order matches images list
        mi = 0
        for ci, frames in enumerate(per_cam_frames):
            for frame_path in frames:
                pose = _c2w_to_colmap(im_poses[mi])
                img_id += 1
                colmap_images[img_id] = {
                    "image_id": img_id,
                    **pose,
                    "camera_id": 1,
                    "name": f"{frame_path.parent.name}_{frame_path.name}",
                }
                mi += 1

    _write_images_bin(sparse_dir / "images.bin", colmap_images)

    _write_points3d_bin_from_xyz_rgb(sparse_dir / "points3D.bin", xyz_all, rgb_all)

    n_written = len(colmap_images)
    console.print(f"  [green]MASt3R calibration done — {n_written} image entries → {sparse_dir}[/]")

    return {
        "n_cameras": n_cams,
        "n_registered": n_written,
        "n_points3D": min(len(xyz_all), 50_000),
        "pct_registered": 100.0,
        "sparse_dir": sparse_dir,
    }


def _mast3r_chunked_inference(
    pairs: list,
    model,
    device: str,
    chunk_size: int = 64,
) -> dict:
    """
    Run MASt3R/DUSt3R inference in chunks and discard descriptor tensors
    immediately after each chunk.

    global_aligner only needs pts3d + conf from pred1/pred2.  The desc
    and desc_conf tensors (24 channels × H × W per image) account for
    ~75 % of inference RAM — stripping them keeps total CPU memory at
    ~4 MB/pair (pts3d+conf only) rather than ~17 MB/pair.
    """
    import torch
    from dust3r.inference import inference

    accumulated: list[dict] = []
    n_chunks = (len(pairs) + chunk_size - 1) // chunk_size

    for ci in range(n_chunks):
        chunk = pairs[ci * chunk_size : (ci + 1) * chunk_size]
        out = inference(chunk, model, device, batch_size=1, verbose=False)

        # Move to CPU and drop large descriptor maps right away.
        # Some values (e.g. true_shape) are plain lists, not tensors.
        import torch as _torch
        def _to_cpu(v):
            return v.cpu() if isinstance(v, _torch.Tensor) else v

        stripped: dict = {}
        for view_key in ("view1", "view2"):
            stripped[view_key] = {k: _to_cpu(v) for k, v in out[view_key].items()}
        for pred_key in ("pred1", "pred2"):
            stripped[pred_key] = {
                k: _to_cpu(v)
                for k, v in out[pred_key].items()
                if k not in ("desc", "desc_conf")
            }
        accumulated.append(stripped)
        del out
        torch.cuda.empty_cache()

        if n_chunks > 1:
            console.print(
                f"  [dim]  chunk {ci + 1}/{n_chunks} done "
                f"({min((ci + 1) * chunk_size, len(pairs))}/{len(pairs)} pairs)[/]"
            )

    # Concatenate chunks along the pair dimension.
    # Non-tensor values (e.g. true_shape lists) are concatenated as plain lists.
    import torch as _torch
    def _cat(lst: list[dict]) -> dict:
        out: dict = {}
        for k in lst[0]:
            vals = [d[k] for d in lst]
            if isinstance(vals[0], _torch.Tensor):
                out[k] = _torch.cat(vals, dim=0)
            else:
                # list-of-lists → flat list
                merged = []
                for v in vals:
                    merged.extend(v if isinstance(v, list) else [v])
                out[k] = merged
        return out

    return {
        "view1": _cat([r["view1"] for r in accumulated]),
        "view2": _cat([r["view2"] for r in accumulated]),
        "pred1": _cat([r["pred1"] for r in accumulated]),
        "pred2": _cat([r["pred2"] for r in accumulated]),
    }


def _evenly_spaced(n: int, k: int) -> list[int]:
    """Return k evenly-spaced indices in [0, n-1], deduplicated."""
    if k >= n:
        return list(range(n))
    idxs = [round(i * (n - 1) / max(k - 1, 1)) for i in range(k)]
    seen: set[int] = set()
    return [x for x in idxs if not (x in seen or seen.add(x))]  # type: ignore[func-returns-value]


def _write_points3d_bin_from_xyz_rgb(
    path: Path,
    xyz: np.ndarray,
    rgb: np.ndarray,
    max_points: int = 50_000,
) -> None:
    """Write COLMAP points3D.bin from xyz [N,3] float64 + rgb [N,3] uint8."""
    n = len(xyz)
    if n > max_points:
        idx = np.random.default_rng(42).choice(n, size=max_points, replace=False)
        xyz, rgb = xyz[idx], rgb[idx]
        n = max_points
    rgb_u8 = rgb.clip(0, 255).astype(np.uint8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", n))
        for i in range(n):
            f.write(struct.pack("<Q", i + 1))
            f.write(struct.pack("<3d", *xyz[i].astype(np.float64)))
            f.write(rgb_u8[i].tobytes())
            f.write(struct.pack("<d", 0.0))  # reprojection error
            f.write(struct.pack("<Q", 0))    # track length = 0


# ── pycolmap refinement of MASt3R output ───────────────────────────────────────

def _refine_mast3r_with_colmap(
    sampled_image_paths: list[str],
    sampled_im_poses: np.ndarray,           # [N, 4, 4] c2w
    focal_init: float, cx_init: float, cy_init: float,
    orig_w: int, orig_h: int,
    work_dir: Path,
) -> dict | None:
    """
    Refine MASt3R poses + intrinsics by running pycolmap SIFT extract → match →
    triangulate_points (with MASt3R poses as priors) → BA.

    Returns a dict with refined poses/intrinsics/points, or None if refinement
    fails or yields too few features (fail-soft — caller keeps MASt3R output).
    """
    import shutil
    try:
        import pycolmap
    except ImportError:
        console.print("  [yellow]pycolmap not available — skipping refinement[/]")
        return None

    work_dir.mkdir(parents=True, exist_ok=True)

    # Flat symlinked image dir using the COLMAP image-name convention:
    # name = <cam_dir>_<frame_file>, e.g. cam_00_frame_000012.jpg
    images_flat = work_dir / "images"
    if images_flat.exists():
        shutil.rmtree(images_flat)
    images_flat.mkdir()
    flat_names: list[str] = []
    for src_path in sampled_image_paths:
        src = Path(src_path)
        name = f"{src.parent.name}_{src.name}"
        dst = images_flat / name
        if dst.exists():
            dst.unlink()
        dst.symlink_to(src.resolve())
        flat_names.append(name)

    # SIFT extract + exhaustive match (cameras are described later, in memory)
    db = work_dir / "database.db"
    if db.exists():
        db.unlink()
    try:
        console.print(f"  [dim]COLMAP refine: SIFT on {len(flat_names)} images…[/]")
        # Force PINHOLE — must match the in-memory Reconstruction we build
        # below. pycolmap's default is SIMPLE_RADIAL which triggers a
        # camera-model-mismatch check failure inside triangulate_points.
        reader_options = pycolmap.ImageReaderOptions()
        reader_options.camera_model = "PINHOLE"
        pycolmap.extract_features(
            database_path=str(db),
            image_path=str(images_flat),
            camera_mode=pycolmap.CameraMode.SINGLE,
            reader_options=reader_options,
        )
        pycolmap.match_exhaustive(database_path=str(db))
    except Exception as exc:
        console.print(f"  [yellow]COLMAP feature/matching failed ({exc}) — skipping refinement[/]")
        return None

    # Read the database-assigned image_ids and build rec using THOSE IDs from
    # the start. Building with our own ids and then calling
    # `transcribe_image_ids_to_database` is broken in pycolmap 4.0.4: that
    # function remaps `image.image_id` + `image.frame_id` but does NOT update
    # the corresponding `frame.data_ids`, so the resulting `Frame`s reference
    # the wrong images. The downstream `triangulate_points` then trips its
    # `existing_frame.DataIds() == frame.DataIds()` check (the v3/v5/v6
    # failure mode). Using DB-aligned ids from the start avoids the call
    # entirely; trivial frames built in this loop have `frame_id = image_id`
    # = DB's image_id, matching the DB's auto-created frames exactly.
    try:
        db_handle = pycolmap.Database.open(str(db))
        name_to_db_id = {img.name: img.image_id for img in db_handle.read_all_images()}
        db_handle.close()
    except Exception as exc:
        console.print(
            f"  [yellow]Could not read image_ids from database ({exc}); "
            "refinement aborted[/]"
        )
        return None

    rec = pycolmap.Reconstruction()
    cam = pycolmap.Camera.create_from_model_name(
        camera_id=1, model_name="PINHOLE",
        focal_length=focal_init, width=orig_w, height=orig_h,
    )
    # create_from_model_name sets params = [f, f, w/2, h/2]; honour MASt3R's
    # actual principal point (often centered, but not guaranteed).
    cam.params = [focal_init, focal_init, cx_init, cy_init]
    rec.add_camera_with_trivial_rig(cam)

    missing_in_db: list[str] = []
    for c2w, name in zip(sampled_im_poses, flat_names):
        db_image_id = name_to_db_id.get(name)
        if db_image_id is None:
            missing_in_db.append(name)
            continue
        R_c2w = c2w[:3, :3]
        t_c2w = c2w[:3, 3]
        R_w2c = R_c2w.T
        t_w2c = -R_w2c @ t_c2w
        cam_from_world = pycolmap.Rigid3d(
            np.hstack([R_w2c, t_w2c[:, None]]).astype(np.float64)
        )
        img = pycolmap.Image(name=name, camera_id=1, image_id=db_image_id)
        rec.add_image_with_trivial_frame(img, cam_from_world)

    if missing_in_db:
        console.print(
            f"  [yellow]{len(missing_in_db)} sampled images missing from DB "
            f"(first: {missing_in_db[0]}); refinement aborted[/]"
        )
        return None

    # Triangulate with MASt3R poses + BA (refine intrinsics on). Read state
    # directly from the in-memory Reconstruction — pycolmap mutates `rec`.
    output_dir = work_dir / "output"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir()
    try:
        pycolmap.triangulate_points(
            reconstruction=rec,
            database_path=str(db),
            image_path=str(images_flat),
            output_path=str(output_dir),
            refine_intrinsics=True,
        )
    except Exception as exc:
        console.print(f"  [yellow]COLMAP triangulate_points failed ({exc}) — skipping refinement[/]")
        return None

    if rec.num_points3D() < 100:
        console.print(
            f"  [yellow]COLMAP refine triangulated only {rec.num_points3D()} points — "
            "low-texture scene; keeping MASt3R output[/]"
        )
        return None

    # `triangulate_points` keeps camera poses fixed by design — it only
    # triangulates new 3D points from given poses and (with refine_intrinsics)
    # tweaks focal length. To actually refine the extrinsics we need a
    # separate `bundle_adjustment` pass. Default `BundleAdjustmentOptions`
    # has `refine_rig_from_world=True` + `refine_focal_length=True` +
    # `refine_points3D=True`, which jointly optimises poses + intrinsics +
    # points against the SIFT correspondences. This is what actually closes
    # the train/test PSNR gap caused by pose noise.
    try:
        ba_opts = pycolmap.BundleAdjustmentOptions()
        ba_opts.print_summary = False
        pycolmap.bundle_adjustment(rec, ba_opts)
    except Exception as exc:
        console.print(
            f"  [yellow]Bundle adjustment failed ({exc}); using triangulate-only output[/]"
        )

    # Build name → refined c2w lookup using pycolmap's Rigid3d.
    # `image.cam_from_world` is a method in pycolmap 4.x (not an attribute) —
    # it walks Image → Frame → rig_from_world to assemble the world→cam pose.
    refined_c2w_by_name: dict[str, np.ndarray] = {}
    for image in rec.images.values():
        if not image.has_pose:
            continue   # image wasn't registered; skip (keep MASt3R pose)
        c2w_rigid = image.cam_from_world().inverse()
        m = np.eye(4)
        m[:3, :4] = c2w_rigid.matrix()
        refined_c2w_by_name[image.name] = m

    refined_im_poses = np.zeros_like(sampled_im_poses)
    n_registered = 0
    for i, name in enumerate(flat_names):
        if name in refined_c2w_by_name:
            refined_im_poses[i] = refined_c2w_by_name[name]
            n_registered += 1
        else:
            # Image wasn't registered after BA — keep MASt3R pose
            refined_im_poses[i] = sampled_im_poses[i]

    if n_registered < len(flat_names) // 2:
        console.print(
            f"  [yellow]Only {n_registered}/{len(flat_names)} images registered "
            "after refinement; keeping MASt3R output[/]"
        )
        return None

    # Refined intrinsics (single shared PINHOLE camera)
    refined_cam = next(iter(rec.cameras.values()))
    refined_focal = float(refined_cam.mean_focal_length())
    refined_cx    = float(refined_cam.principal_point_x)
    refined_cy    = float(refined_cam.principal_point_y)

    # Diagnostic: how much did poses move? Small shift = MASt3R dense cloud
    # stays usable as-is. Large shift = caller may want to discard it.
    pose_drift = np.linalg.norm(
        refined_im_poses[:, :3, 3] - sampled_im_poses[:, :3, 3], axis=1
    )
    drift_p50, drift_p90 = float(np.median(pose_drift)), float(np.percentile(pose_drift, 90))

    console.print(
        f"  [green]COLMAP refine: {n_registered}/{len(flat_names)} imgs registered, "
        f"{rec.num_points3D():,} SIFT points triangulated, "
        f"focal {focal_init:.1f}→{refined_focal:.1f}px, "
        f"pose drift p50={drift_p50:.4f} p90={drift_p90:.4f}[/]"
    )
    return {
        "im_poses":  refined_im_poses,
        "focal":     refined_focal,
        "cx":        refined_cx,
        "cy":        refined_cy,
    }


# ── COLMAP multi-camera ─────────────────────────────────────────────────────────

def _run_colmap(
    cam_dirs: list[Path],
    out_dir: Path,
    sparse_dir: Path,
    camera_model: str,
    calib_frames: int | None = None,
) -> dict:
    try:
        import pycolmap
    except ImportError:
        raise ImportError(
            "pycolmap is required.\n"
            "Install it: pip install pycolmap"
        )

    # Flatten frames into a single images/ directory, prefixed by cam ID, so
    # COLMAP sees all cameras in one pass while the filename prefix encodes
    # which physical camera each frame came from.
    #
    # For a static synchronised rig the geometry is identical in every frame,
    # so we calibrate on a small subset per camera (default: 1 representative
    # middle frame). This is both far faster — exhaustive matching is O(N²) in
    # image count, so 18 images vs 1800 is seconds vs many hours — and cleaner:
    # extra frames add redundant near-identical views plus moving foreground
    # (people/objects) that produce spurious cross-camera matches and dilute the
    # static scene structure COLMAP should be triangulating. Pass calib_frames
    # to sample more (e.g. for slowly-moving rigs); None ⇒ 1 frame/camera.
    n_per_cam = calib_frames if calib_frames is not None else _COLMAP_RIG_FRAMES

    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    for cam_dir in cam_dirs:
        frames = sorted(cam_dir.glob("frame_*.jpg"))
        if not frames:
            continue
        if n_per_cam >= len(frames):
            selected = frames
        elif n_per_cam == 1:
            selected = [frames[len(frames) // 2]]   # middle: representative static view
        else:
            selected = [frames[i] for i in _evenly_spaced(len(frames), n_per_cam)]
        for frame in selected:
            # Name: cam_00_frame_000000.jpg  →  unambiguous per-camera identity
            dst = images_dir / f"{cam_dir.name}_{frame.name}"
            if not dst.exists():
                dst.symlink_to(frame.resolve())

    n_images = len(list(images_dir.glob("*.jpg")))
    console.print(f"  [dim]COLMAP workspace: {n_images} images from {len(cam_dirs)} cameras[/]")

    workspace = out_dir / "colmap_workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    db_path = workspace / "database.db"

    # Feature extraction
    console.print("  [dim]COLMAP: feature extraction…[/]")
    reader_opts = pycolmap.ImageReaderOptions()
    reader_opts.camera_model = camera_model
    pycolmap.extract_features(
        database_path=db_path,
        image_path=images_dir,
        camera_mode=pycolmap.CameraMode.SINGLE,  # one intrinsic model shared
        reader_options=reader_opts,
    )

    # Exhaustive matching — necessary because frames from different cameras
    # have no temporal neighbourhood relationship.
    console.print("  [dim]COLMAP: exhaustive matching…[/]")
    pycolmap.match_exhaustive(database_path=db_path)

    # Incremental reconstruction
    console.print("  [dim]COLMAP: incremental mapping…[/]")
    reconstructions = pycolmap.incremental_mapping(
        database_path=db_path,
        image_path=images_dir,
        output_path=sparse_dir.parent,
    )

    if not reconstructions:
        raise RuntimeError(
            "COLMAP failed to reconstruct any cameras.\n"
            "Try --camera-model SIMPLE_RADIAL or ensure sufficient visual overlap."
        )

    rec = reconstructions[0]
    n_reg = len(rec.images)
    n_pts = len(rec.points3D)
    pct = 100.0 * n_reg / n_images

    console.print(
        f"  [{'green' if pct >= 70 else 'yellow'}]"
        f"Registered {n_reg}/{n_images} images ({pct:.0f}%), {n_pts:,} sparse points[/]"
    )

    return {
        "n_cameras": len(cam_dirs),
        "n_registered": n_reg,
        "n_points3D": n_pts,
        "pct_registered": pct,
        "sparse_dir": sparse_dir,
        "images_dir": images_dir,
    }


# ── COLMAP binary writers ────────────────────────────────────────────────────────

def _write_cameras_bin(path: Path, cameras: dict) -> None:
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(cameras)))
        for cam in cameras.values():
            model_id = {"SIMPLE_PINHOLE": 0, "PINHOLE": 1, "SIMPLE_RADIAL": 2, "RADIAL": 3}.get(
                cam["model"], 1
            )
            params = cam["params"]
            f.write(struct.pack("<IiQQ", cam["camera_id"], model_id, cam["width"], cam["height"]))
            for p in params:
                f.write(struct.pack("<d", p))


def _write_images_bin(path: Path, images: dict) -> None:
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(images)))
        for img in images.values():
            f.write(struct.pack("<I", img["image_id"]))
            f.write(struct.pack("<4d", img["qw"], img["qx"], img["qy"], img["qz"]))
            f.write(struct.pack("<3d", img["tx"], img["ty"], img["tz"]))
            f.write(struct.pack("<I", img["camera_id"]))
            f.write(img["name"].encode("utf-8") + b"\x00")
            f.write(struct.pack("<Q", 0))  # 0 keypoints


def _write_points3d_bin(path: Path) -> None:
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", 0))


# ── Math helpers ─────────────────────────────────────────────────────────────────

def _rotation_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Convert 3×3 rotation matrix to (w, x, y, z) quaternion."""
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(R).as_quat()[[3, 0, 1, 2]]  # xyzw → wxyz


def _infer_image_size(cam_dirs: list[Path]) -> tuple[int, int]:
    for cam_dir in cam_dirs:
        frames = list(cam_dir.glob("*.jpg"))
        if frames:
            import cv2
            img = cv2.imread(str(frames[0]))
            if img is not None:
                h, w = img.shape[:2]
                return w, h
    return 1280, 720  # fallback
