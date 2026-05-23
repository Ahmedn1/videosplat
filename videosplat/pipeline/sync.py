from __future__ import annotations

"""
Multi-camera frame extraction and synchronization.

Two input modes:
  1. Video files — extract frames from each camera's MP4/MOV, then align
     by a shared audio event (e.g. a hand-clap at the start of each take).
  2. Pre-extracted directories — validate that per-camera frame dirs exist
     and are consistent; skip extraction entirely.

Output layout:
  <out_dir>/
    cam_00/frame_000000.jpg
    cam_01/frame_000000.jpg
    ...

Frame indices are common across cameras after synchronization, i.e.
cam_00/frame_000042.jpg and cam_01/frame_000042.jpg represent the same
instant in time.
"""

import math
import re
import shutil
from pathlib import Path

import cv2
import numpy as np
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

console = Console()

# ── Public API ──────────────────────────────────────────────────────────────────

def extract_and_sync(
    source: Path,
    out_dir: Path,
    *,
    target_fps: float = 10.0,
    max_dim: int = 1280,
    audio_sync: bool = True,
    pre_extracted: bool = False,
) -> dict:
    """
    Main entry point.

    Args:
        source:         Either a directory of .mp4/.mov files (one per camera)
                        or a directory of per-camera subdirs (pre-extracted).
        out_dir:        Output root; cam_XX/ subdirs are created here.
        target_fps:     Frame extraction rate (applies only when extracting from video).
        max_dim:        Resize longest edge to this pixel count.
        audio_sync:     If True, attempt audio-based synchronization. Falls back
                        to frame-count alignment if librosa is not installed.
        pre_extracted:  If True, treat source as already-extracted cam_XX/ dirs.

    Returns:
        dict with keys: n_cameras, n_frames, fps, cam_dirs
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    if pre_extracted:
        return _validate_pre_extracted(source, out_dir)

    video_files = _find_videos(source)
    if not video_files:
        raise FileNotFoundError(f"No video files found in {source}")

    console.print(f"  Found {len(video_files)} camera video(s)")

    # Extract frames from each camera into temporary per-cam dirs
    raw_dirs: list[Path] = []
    raw_fps_list: list[float] = []
    for idx, vf in enumerate(sorted(video_files)):
        cam_raw = out_dir / f"_raw_cam_{idx:02d}"
        fps = _extract_frames(vf, cam_raw, target_fps=target_fps, max_dim=max_dim, cam_id=idx)
        raw_dirs.append(cam_raw)
        raw_fps_list.append(fps)

    # Synchronize (find common start frame per camera)
    if len(video_files) > 1 and audio_sync:
        offsets = _audio_sync(video_files, raw_dirs, target_fps)
    else:
        offsets = [0] * len(video_files)

    # Apply offsets and copy to final cam_XX/ dirs
    cam_dirs = _apply_offsets(raw_dirs, offsets, out_dir)

    # Clean up raw temp dirs
    for d in raw_dirs:
        shutil.rmtree(d, ignore_errors=True)

    n_frames = min(len(list(d.glob("*.jpg"))) for d in cam_dirs)
    console.print(
        f"  [green]Sync complete — {len(cam_dirs)} cameras, {n_frames} common frames[/]"
    )

    return {
        "n_cameras": len(cam_dirs),
        "n_frames": n_frames,
        "fps": target_fps,
        "cam_dirs": cam_dirs,
    }


# ── Frame extraction ─────────────────────────────────────────────────────────────

def _extract_frames(
    video_path: Path,
    out_dir: Path,
    *,
    target_fps: float,
    max_dim: int,
    cam_id: int,
) -> float:
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    stride = max(1, round(src_fps / target_fps))

    console.print(
        f"  cam_{cam_id:02d}: {video_path.name}  "
        f"{src_fps:.1f} fps, {total} frames → extracting every {stride}"
    )

    frame_idx = 0
    saved = 0

    with Progress(
        SpinnerColumn(), TextColumn(f"  cam_{cam_id:02d}"), BarColumn(), TaskProgressColumn(),
        console=console, transient=True,
    ) as prog:
        task = prog.add_task("", total=total // stride)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % stride == 0:
                frame = _resize(frame, max_dim)
                fname = out_dir / f"frame_{saved:06d}.jpg"
                cv2.imwrite(str(fname), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
                saved += 1
                prog.advance(task)

            frame_idx += 1

    cap.release()
    console.print(f"  [dim]cam_{cam_id:02d}: {saved} frames extracted → {out_dir}[/]")
    return src_fps


def _resize(frame: np.ndarray, max_dim: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if max(h, w) <= max_dim:
        return frame
    scale = max_dim / max(h, w)
    new_w = round(w * scale / 2) * 2
    new_h = round(h * scale / 2) * 2
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


# ── Audio synchronization ────────────────────────────────────────────────────────

def _visual_sync(cam_dirs: list[Path]) -> list[int]:
    """
    Find per-camera temporal offsets by cross-correlating frame-difference
    motion signals. Works for any scene with common motion events (ball, jump).

    Falls back to zero offsets if frames cannot be read.
    """
    try:
        from scipy.signal import correlate as _correlate
    except ImportError:
        console.print("  [yellow]scipy not installed — skipping visual sync.[/]")
        return [0] * len(cam_dirs)

    signals = []
    for cam_dir in cam_dirs:
        frames = sorted(cam_dir.glob("frame_*.jpg"))
        if len(frames) < 2:
            signals.append(np.array([0.0]))
            continue
        diffs = []
        prev = None
        for fp in frames:
            img = cv2.imread(str(fp), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = img.astype(np.float32)
            if prev is not None:
                diffs.append(float(np.mean(np.abs(img - prev))))
            prev = img
        signals.append(np.array(diffs) if diffs else np.array([0.0]))

    # Cross-correlate each camera's signal against camera 0
    ref  = signals[0]
    lags = [0]
    for sig in signals[1:]:
        n    = max(len(ref), len(sig))
        corr = _correlate(sig, ref, mode="full")
        lag  = int(np.argmax(corr)) - (len(ref) - 1)
        lags.append(lag)

    # Convert lags to frame offsets: positive lag = this camera is behind ref
    # We trim cameras that are AHEAD of the slowest-starting camera
    min_lag = min(lags)
    offsets = [lag - min_lag for lag in lags]   # all ≥ 0

    console.print(f"  [dim]Visual sync offsets (frames to skip): {offsets}[/]")
    return offsets


def _audio_sync(video_files: list[Path], raw_dirs: list[Path], target_fps: float) -> list[int]:
    """
    Find the sample offset of a shared audio transient (e.g. clap) in each
    video and return per-camera frame offsets relative to the latest start.

    Falls back to zero offsets if librosa/soundfile are not installed or if
    audio extraction fails.
    """
    try:
        import librosa
        import soundfile  # noqa: F401 — checked for availability
    except ImportError:
        console.print(
            "  [yellow]librosa/soundfile not installed — falling back to visual sync.[/]\n"
            "  Install with: pip install 'splat[audio]'"
        )
        return _visual_sync(raw_dirs)

    audios: list[np.ndarray] = []
    sr_ref: int | None = None

    for vf in video_files:
        try:
            y, sr = librosa.load(str(vf), sr=22050, mono=True)
            if sr_ref is None:
                sr_ref = sr
            audios.append(y)
        except Exception as e:
            console.print(f"  [yellow]Audio load failed for {vf.name}: {e} — falling back to visual sync.[/]")
            return _visual_sync(raw_dirs)

    # Detect transient (clap) position in each audio stream via onset strength
    clap_samples = []
    for y in audios:
        onset_env = librosa.onset.onset_strength(y=y, sr=sr_ref)
        peak = int(np.argmax(onset_env))
        # Convert onset frame → sample
        hop = 512
        clap_samples.append(peak * hop)

    # Frame offsets: relative to latest clap (all cameras start from the clap)
    latest = max(clap_samples)
    frame_offsets = []
    for cs in clap_samples:
        # How many frames before the clap to skip in this camera's extracted sequence
        # clap_samples[i] is in audio samples; target_fps cancels with src_fps
        # so we approximate: offset_in_frames ≈ (latest - cs) / sr * target_fps
        # Since we've already extracted at target_fps, this is a rough approximation.
        frame_offsets.append(round((latest - cs) / sr_ref * target_fps))

    console.print(f"  [dim]Audio sync offsets (frames to skip): {frame_offsets}[/]")
    return frame_offsets


# ── Offset application ───────────────────────────────────────────────────────────

def _apply_offsets(raw_dirs: list[Path], offsets: list[int], out_dir: Path) -> list[Path]:
    """
    Trim each camera's frames by its offset and copy to cam_XX/ dirs.
    All cameras end up with the same number of frames.
    """
    cam_dirs = []
    trimmed: list[list[Path]] = []

    for raw_dir, off in zip(raw_dirs, offsets):
        frames = sorted(raw_dir.glob("frame_*.jpg"))
        trimmed.append(frames[max(0, off):])

    n_common = min(len(t) for t in trimmed)

    for idx, (raw_dir, frames) in enumerate(zip(raw_dirs, trimmed)):
        cam_dir = out_dir / f"cam_{idx:02d}"
        cam_dir.mkdir(parents=True, exist_ok=True)
        for new_idx, src in enumerate(frames[:n_common]):
            dst = cam_dir / f"frame_{new_idx:06d}.jpg"
            shutil.copy2(src, dst)
        cam_dirs.append(cam_dir)

    return cam_dirs


# ── Pre-extracted validation ─────────────────────────────────────────────────────

def _validate_pre_extracted(source: Path, out_dir: Path) -> dict:
    cam_dirs = sorted(source.glob("cam_*"))
    if not cam_dirs:
        raise FileNotFoundError(f"No cam_XX/ directories found in {source}")

    counts = {d: len(list(d.glob("*.jpg"))) for d in cam_dirs}
    min_frames = min(counts.values())

    console.print(
        f"  Pre-extracted: {len(cam_dirs)} cameras, "
        f"{min_frames} common frames"
    )

    # If source == out_dir, we're done; otherwise symlink/copy
    if source.resolve() != out_dir.resolve():
        for cam_dir in cam_dirs:
            dst = out_dir / cam_dir.name
            if not dst.exists():
                dst.symlink_to(cam_dir.resolve())

    return {
        "n_cameras": len(cam_dirs),
        "n_frames": min_frames,
        "fps": None,  # unknown when pre-extracted
        "cam_dirs": [out_dir / d.name for d in cam_dirs],
    }


# ── Helpers ──────────────────────────────────────────────────────────────────────

def _find_videos(directory: Path) -> list[Path]:
    exts = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    return [f for f in sorted(directory.iterdir()) if f.suffix.lower() in exts]


def video_info(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return {"fps": fps, "width": w, "height": h, "n_frames": n, "duration_sec": n / fps if fps else 0}
