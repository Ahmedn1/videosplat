from __future__ import annotations

"""
4DGS training: multi-camera frames + COLMAP poses → trained 4DGaussians model.

Wraps 4DGaussians/train.py as a subprocess (standalone subprocess pattern)
to keep the rasterizer build isolated from the CLI's Python environment.

After training, a separate bake step evaluates the deformation network at
N evenly-spaced timestamps and writes one standard 3DGS PLY per keyframe.
Each PLY contains full SH3 colour coefficients (f_dc + f_rest), compatible
with the mkkellogg browser viewer.

4DGaussians data layout (dynerf mode, auto-detected from poses_bounds.npy):
  <source>/
    cam00.mp4, cam01.mp4, …   ← raw N3V video files
    poses_bounds.npy           ← NeRF-style camera intrinsics + extrinsics
    points3D_downsample2.ply   ← sparse init point cloud (generated if missing)
"""

import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

from rich.console import Console

console = Console()


# ── Public API ──────────────────────────────────────────────────────────────────

def train_4dgs(
    source_path: Path,
    model_path: Path,
    stg_dir: Path,          # named stg_dir for CLI compat; points to 4DGaussians root
    *,
    iterations: int = 14_000,
    n_keyframes: int = 50,
    max_cameras: int | None = None,
    downsample: int = 4,
    stg_config: Path | None = None,   # unused for 4DGaussians (no mmcv config needed)
    train_python: str | None = None,
    extra_args: list[str] | None = None,
) -> Path:
    """
    Run 4DGaussians training then bake keyframe PLY snapshots.

    Args:
        source_path:  Scene directory (N3V dynerf: contains cam*.mp4 + poses_bounds.npy).
        model_path:   Output directory for the trained model.
        stg_dir:      Root of the cloned 4DGaussians repo.
        iterations:   Training iterations (14k default ≈ 25 min; 30k ≈ 60 min).
        n_keyframes:  Number of time steps to bake after training.
        max_cameras:  If set, select this many cameras using farthest-point sampling.
        train_python: Python executable with 4DGaussians dependencies.
        extra_args:   Extra flags forwarded to train.py.

    Returns:
        model_path (for chaining)
    """
    backend_dir = stg_dir  # alias for clarity
    train_script = backend_dir / "train.py"
    if not train_script.exists():
        raise FileNotFoundError(
            f"4DGaussians/train.py not found at {train_script}.\n"
            "Clone: git clone https://github.com/hustvl/4DGaussians\n"
            "Set path: videosplat config --backend-dir /path/to/4DGaussians"
        )

    model_path.mkdir(parents=True, exist_ok=True)
    python_exe = train_python or sys.executable

    # Generate points3D_downsample2.ply at the *original* source path before
    # the camera subset is built — subset creation symlinks the ply from here.
    _ensure_init_pointcloud(source_path, backend_dir, python_exe)

    # Optionally create a camera-subset directory with diverse viewpoints
    if max_cameras is not None:
        source_path = _prepare_camera_subset(source_path, model_path, max_cameras)

    cmd = [
        python_exe, str(train_script),
        "--source_path", str(source_path.resolve()),
        "--model_path",  str(model_path.resolve()),
        "--iterations",  str(iterations),
        "--save_iterations", str(iterations),
        "--expname", source_path.name,
    ]
    if extra_args:
        cmd.extend(extra_args)

    env = _build_env(backend_dir, downsample=downsample)

    console.print(
        f"  [dim]Training 4DGaussians: {iterations:,} iterations "
        f"(source: {source_path.name}, downsample: {downsample}×)[/]"
    )

    t0 = time.time()
    result = subprocess.run(cmd, env=env, cwd=str(backend_dir))
    elapsed = time.time() - t0

    if result.returncode != 0:
        raise RuntimeError(
            f"4DGaussians training failed (exit code {result.returncode}).\n"
            "Check the output above for CUDA / memory errors."
        )

    console.print(f"  [green]Training complete in {elapsed / 60:.1f} min[/]")

    config = {
        "source_path": str(source_path),
        "iterations": iterations,
        "n_keyframes": n_keyframes,
        "train_time_min": round(elapsed / 60, 1),
    }
    (model_path / "splat_train_config.json").write_text(json.dumps(config, indent=2))

    # Bake keyframe PLY snapshots using deformation network
    console.print(f"  Baking {n_keyframes} keyframe snapshots…")
    render_keyframes(
        model_path=model_path,
        backend_dir=backend_dir,
        source_path=source_path,
        n_keyframes=n_keyframes,
        iterations=iterations,
        python_exe=python_exe,
        env=env,  # already has DYNERF_DOWNSAMPLE set
    )

    return model_path


# ── Keyframe baking ─────────────────────────────────────────────────────────────

def render_keyframes(
    model_path: Path,
    backend_dir: Path,
    source_path: Path,
    n_keyframes: int,
    iterations: int,
    python_exe: str,
    env: dict,
    configs: str | None = None,
    prune_opacity: float = 0.05,
    prune_scale_mult: float = 5.0,
    prune_dist_mult: float = 3.0,
) -> list[Path]:
    """
    Run bake_4dgs_keyframes.py as a subprocess with 4DGaussians on PYTHONPATH.
    Generates N PLY files in standard 3DGS format with full SH3 colour.
    `configs`: 4DGaussians config the model was trained with (required for the
    casual/nerfies path so the baked deformation-net architecture matches).
    """
    keyframes_dir = model_path / "keyframes"
    keyframes_dir.mkdir(parents=True, exist_ok=True)

    bake_script = Path(__file__).parent / "bake_4dgs_keyframes.py"

    cmd = [
        python_exe, str(bake_script),
        "--model_path",  str(model_path.resolve()),
        "--source_path", str(source_path.resolve()),
        "--n_keyframes", str(n_keyframes),
        "--output_dir",  str(keyframes_dir.resolve()),
        "--iteration",   str(iterations),
    ]
    if configs:
        cmd += ["--configs", str(configs)]
    cmd += ["--prune-opacity", str(prune_opacity),
            "--prune-scale-mult", str(prune_scale_mult),
            "--prune-dist-mult", str(prune_dist_mult)]

    result = subprocess.run(cmd, env=env, cwd=str(backend_dir))
    if result.returncode != 0:
        raise RuntimeError(
            f"Keyframe baking failed (exit code {result.returncode}).\n"
            "Check the output above."
        )

    paths = sorted(keyframes_dir.glob("keyframe_*.ply"))
    console.print(f"  [green]{len(paths)} keyframe PLYs → {keyframes_dir}[/]")
    return paths


def train_nerfies(
    source_path: Path,      # nerfies dataset dir (dataset.json + camera/ + rgb/2x + points3D_downsample2.ply)
    model_path: Path,
    backend_dir: Path,      # 4DGaussians root
    *,
    configs: str,           # e.g. <backend>/arguments/hypernerf/default.py
    iterations: int = 14_000,
    n_keyframes: int = 100,
    train_python: str | None = None,
    extra_args: list[str] | None = None,
    prune_opacity: float = 0.05,
    prune_scale_mult: float = 5.0,
    prune_dist_mult: float = 3.0,
    opt_overrides: dict | None = None,
) -> Path:
    """Train 4DGaussians in nerfies/HyperNeRF mode (casual moving/heterogeneous
    cams), then bake keyframes. Auto-detected as nerfies via dataset.json (and the
    absence of a COLMAP sparse/). Requires the mmengine+mmcv shim for --configs.

    opt_overrides: dict merged into the config's OptimizationParams (e.g.
    {"opacity_reset_interval": 3000, "lambda_dssim": 0.2}) — re-enable anti-floater
    regularisation that HyperNeRF's default disables for object-centric scenes."""
    train_script = backend_dir / "train.py"
    if not train_script.exists():
        raise FileNotFoundError(f"4DGaussians/train.py not found at {train_script}.")
    model_path.mkdir(parents=True, exist_ok=True)
    python_exe = train_python or sys.executable
    # 4DGaussians' --configs OVERRIDES CLI args, so a plain --iterations is ignored.
    # Materialise an effective config that inherits the chosen one but honours the
    # requested iteration count (so --iterations actually takes effect).
    eff_cfg = configs
    try:
        from mmengine import Config
        c = Config.fromfile(str(configs))
        op = dict(c.get("OptimizationParams", {}))
        op["iterations"] = int(iterations)
        op["coarse_iterations"] = min(int(op.get("coarse_iterations", 3000)), max(500, int(iterations) // 4))
        if opt_overrides:
            op.update(opt_overrides)
            console.print(f"  [dim]OptimizationParams overrides: {opt_overrides}[/]")
        c.OptimizationParams = op
        eff_cfg = str(model_path / "_effective_config.py")
        c.dump(eff_cfg)
    except Exception as e:
        console.print(f"  [yellow]could not override iterations in config ({e}); using {Path(configs).name} as-is[/]")
    cmd = [
        python_exe, str(train_script),
        "--source_path", str(source_path.resolve()),
        "--model_path",  str(model_path.resolve()),
        "--configs",     str(eff_cfg),
        "--iterations",  str(iterations),
        "--save_iterations", str(iterations),
        "--expname", source_path.name,
    ]
    if extra_args:
        cmd.extend(extra_args)
    env = _build_env(backend_dir, downsample=1)   # nerfies ignores DYNERF_DOWNSAMPLE
    console.print(f"  [dim]Training 4DGaussians (nerfies): {iterations:,} iters, configs={Path(configs).name}[/]")
    t0 = time.time()
    if subprocess.run(cmd, env=env, cwd=str(backend_dir)).returncode != 0:
        raise RuntimeError("4DGaussians nerfies training failed — check output above.")
    console.print(f"  [green]Training complete in {(time.time()-t0)/60:.1f} min[/]")
    console.print(f"  Baking {n_keyframes} keyframe snapshots…")
    render_keyframes(model_path=model_path, backend_dir=backend_dir, source_path=source_path,
                     n_keyframes=n_keyframes, iterations=iterations, python_exe=python_exe,
                     env=env, configs=configs, prune_opacity=prune_opacity,
                     prune_scale_mult=prune_scale_mult, prune_dist_mult=prune_dist_mult)
    return model_path


# ── Init point cloud ─────────────────────────────────────────────────────────────

def _ensure_init_pointcloud(source_path: Path, backend_dir: Path, python_exe: str) -> None:
    """
    4DGaussians' dynerf loader requires points3D_downsample2.ply.
    Generate it from the COLMAP sparse points if not already present,
    filtering long-tail outlier triangulations that would blow up the
    deformation network's working AABB.
    """
    target = source_path / "points3D_downsample2.ply"
    if target.exists():
        return

    # Look for sparse points in either layout:
    #   STG/COLMAP:  source_path/colmap_0/sparse/0/points3D.bin
    #   4DGS/MASt3R: source_path/sparse/0/points3D.bin (written by calibrate.py)
    for candidate in (
        source_path / "sparse" / "0" / "points3D.bin",
        source_path / "colmap_0" / "sparse" / "0" / "points3D.bin",
    ):
        if candidate.exists():
            bin_path = candidate
            break
    else:
        console.print(
            f"  [yellow]No COLMAP sparse points found and points3D_downsample2.ply missing. "
            "4DGaussians will fail to load the scene.[/]"
        )
        return

    # Read camera centers from poses_bounds.npy (LLFF format: (N, 17) →
    # first 15 are flattened 3x5 pose [R|t|hwf], translation = col 3).
    # We need centroid + max-radius to bound the init cloud to inliers.
    import numpy as np
    poses_path = source_path / "poses_bounds.npy"
    cam_center: list[float] | None = None
    keep_radius: float | None = None
    if poses_path.exists():
        try:
            pb = np.load(str(poses_path))
            cam_pos = pb[:, :-2].reshape(-1, 3, 5)[:, :, 3]  # (N, 3)
            centroid = cam_pos.mean(axis=0)
            radius   = float(np.linalg.norm(cam_pos - centroid, axis=1).max())
            # Keep points within 3× camera radius of the camera cluster.
            # The cameras themselves are within 1×; legitimate scene
            # content out to ~3× covers walls, ceiling, surroundings.
            # Points beyond that are reconstruction outliers (background
            # through windows, glare, stray feature triangulations) that
            # would distort 4DGS's deformation-net AABB.
            cam_center  = centroid.tolist()
            keep_radius = 3.0 * radius
        except Exception as exc:
            console.print(f"  [yellow]Could not read camera centroid ({exc}); filter disabled[/]")

    console.print("  [dim]Generating points3D_downsample2.ply from COLMAP sparse…[/]")
    bin_abs = bin_path.resolve()
    target_abs = target.resolve()

    # Build subprocess script. Two branches: with-filter (we know camera
    # extent) and without (no poses_bounds.npy, fall back to random N).
    if cam_center is not None and keep_radius is not None:
        cx, cy, cz = cam_center
        script = (
            "import sys; sys.path.insert(0, '.');"
            "from scene.colmap_loader import read_points3D_binary;"
            "from scene.dataset_readers import storePly;"
            "import numpy as np;"
            f"xyz, rgb, _ = read_points3D_binary('{bin_abs}');"
            f"center = np.array([{cx!r}, {cy!r}, {cz!r}]);"
            f"keep_r = {keep_radius!r};"
            "n_total = len(xyz);"
            "dist = np.linalg.norm(xyz - center, axis=1);"
            "inlier = dist < keep_r;"
            "xyz, rgb = xyz[inlier], rgb[inlier];"
            "n_inlier = len(xyz);"
            "n_target = min(n_inlier, 100_000);"
            "n_target = max(1, n_target);"
            "rng = np.random.default_rng(0);"
            "idx = rng.choice(n_inlier, size=n_target, replace=False) if n_inlier > 0 else np.zeros(0, dtype=int);"
            f"storePly('{target_abs}', xyz[idx], rgb[idx]);"
            f"print(f'init cloud: {{n_total}} → {{n_inlier}} inliers within {{keep_r:.2f}} of camera centroid → {{n_target}} written to {target_abs}')"
        )
    else:
        # No poses_bounds.npy — keep the historical random-half behaviour,
        # but cap at 100K so we don't ship an unhinged number of points.
        script = (
            "import sys; sys.path.insert(0, '.');"
            "from scene.colmap_loader import read_points3D_binary;"
            "from scene.dataset_readers import storePly;"
            "import numpy as np;"
            f"xyz, rgb, _ = read_points3D_binary('{bin_abs}');"
            "n_total = len(xyz);"
            "n_target = min(max(1, n_total // 2), 100_000);"
            "rng = np.random.default_rng(0);"
            "idx = rng.choice(n_total, size=n_target, replace=False);"
            f"storePly('{target_abs}', xyz[idx], rgb[idx]);"
            f"print(f'init cloud: {{n_total}} → {{n_target}} (no camera bounds; filter disabled) → {target_abs}')"
        )
    subprocess.run([python_exe, "-c", script], cwd=str(backend_dir), check=True)


# ── Camera subset ───────────────────────────────────────────────────────────────

def _prepare_camera_subset(source_path: Path, model_path: Path, max_cameras: int) -> Path:
    """
    Select `max_cameras` diverse cameras from the N3V scene using farthest-point
    sampling on positions extracted from poses_bounds.npy, then create a subset
    directory with symlinks so 4DGaussians trains on only those cameras.

    Returns the subset directory path (or source_path unchanged if max_cameras
    >= total available cameras).
    """
    import numpy as np
    import shutil

    poses_bounds = source_path / "poses_bounds.npy"
    if not poses_bounds.exists():
        console.print("[yellow]poses_bounds.npy not found — using all cameras[/]")
        return source_path

    pb = np.load(str(poses_bounds))            # (N_cams, 17)
    n_total = pb.shape[0]

    if max_cameras >= n_total:
        console.print(f"  [dim]max_cameras={max_cameras} ≥ total {n_total} — using all[/]")
        return source_path

    # Camera centre: column 3 of each 3×4 c2w block (positions 3, 8, 13 in the 15-value c2w)
    c2w = pb[:, :15].reshape(n_total, 3, 5)
    positions = c2w[:, :3, 3]                  # (N, 3) camera centres in world space

    # Farthest-point sampling
    selected = [0]
    min_dists = np.full(n_total, np.inf)
    for _ in range(max_cameras - 1):
        last = positions[selected[-1]]
        dists = np.linalg.norm(positions - last, axis=1)
        min_dists = np.minimum(min_dists, dists)
        min_dists[selected] = -1              # exclude already-selected
        selected.append(int(np.argmax(min_dists)))

    selected.sort()
    console.print(
        f"  [dim]Camera subset (farthest-point, {max_cameras}/{n_total}): "
        f"cam{selected}[/]".replace("[", "cam_ids=[")
    )

    # Build subset directory
    subset_dir = model_path / "cam_subset"
    if subset_dir.exists():
        shutil.rmtree(subset_dir)
    subset_dir.mkdir(parents=True)

    # Symlink selected cam*.mp4 files
    mp4s = sorted(source_path.glob("cam*.mp4"))
    for idx in selected:
        if idx < len(mp4s):
            src = mp4s[idx]
            (subset_dir / src.name).symlink_to(src.resolve())
        # Also symlink cam*/images/ cache if it exists
        cam_images = source_path / f"cam{idx:02d}" / "images"
        if cam_images.exists():
            cam_dir = subset_dir / f"cam{idx:02d}"
            cam_dir.mkdir(exist_ok=True)
            (cam_dir / "images").symlink_to(cam_images.resolve())

    # Write subset poses_bounds.npy
    np.save(str(subset_dir / "poses_bounds.npy"), pb[selected])

    # Symlink init point cloud
    init_ply = source_path / "points3D_downsample2.ply"
    if init_ply.exists():
        (subset_dir / init_ply.name).symlink_to(init_ply.resolve())

    return subset_dir


# ── Helpers ─────────────────────────────────────────────────────────────────────

def find_ply(model_path: Path, iteration: int) -> Path:
    return model_path / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"


def count_gaussians(ply_path: Path) -> int:
    try:
        from plyfile import PlyData
        return len(PlyData.read(str(ply_path))["vertex"])
    except Exception:
        return -1


def _build_env(backend_dir: Path, downsample: int = 4) -> dict:
    env = dict(os.environ)

    if "CUDA_HOME" not in env:
        for candidate in [
            Path.home() / "cuda-home",
            Path("/usr/local/cuda"),
            Path("/usr/local/cuda-12"),
            Path("/usr/local/cuda-11"),
        ]:
            if (candidate / "bin" / "nvcc").exists():
                env["CUDA_HOME"] = str(candidate)
                break

    cuda_home = env.get("CUDA_HOME", "")
    if cuda_home:
        env["PATH"] = f"{cuda_home}/bin:" + env.get("PATH", "")

    if Path("/usr/bin/gcc-12").exists():
        env.setdefault("CC", "/usr/bin/gcc-12")
        env.setdefault("CXX", "/usr/bin/g++-12")

    pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{backend_dir}:{pythonpath}" if pythonpath else str(backend_dir)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env["DYNERF_DOWNSAMPLE"] = str(float(downsample))

    return env


# Backwards-compat alias
train = train_4dgs


# ── SpacetimeGaussians training ───────────────────────────────────────────────

def train_stg(
    source_path: Path,
    model_path: Path,
    stg_dir: Path,
    *,
    iterations: int = 20_000,
    n_keyframes: int = 50,
    downsample: int = 2,
    stg_config: Path | None = None,
    train_python: str | None = None,
    extra_args: list[str] | None = None,
) -> Path:
    """
    Run SpacetimeGaussians training then bake keyframe PLY snapshots.

    source_path must contain colmap_0/ (produced by prepare_stg_source()).
    STG trains with --source_path pointing to colmap_0 inside source_path.
    """
    train_script = stg_dir / "train.py"
    if not train_script.exists():
        raise FileNotFoundError(
            f"SpacetimeGaussians/train.py not found at {train_script}.\n"
            "Clone: git clone https://github.com/oppo-us-research/SpacetimeGaussians\n"
            "Set:   videosplat config --stg-dir /path/to/SpacetimeGaussians"
        )

    model_path.mkdir(parents=True, exist_ok=True)
    python_exe = train_python or sys.executable

    # Auto-detect or generate STG config
    config_path = stg_config or _get_stg_config(stg_dir, source_path, n_keyframes, downsample)

    cmd = [
        python_exe, str(train_script),
        "--config",         str(config_path),
        "--model_path",     str(model_path.resolve()),
        "--source_path",    str((source_path / "colmap_0").resolve()),
        # Pass explicitly so helper3dg.py's "CLI != default → keep CLI" logic wins over config
        "--iterations",     str(iterations),
        "--resolution",     str(downsample),
        "--save_iterations", str(iterations),
    ]
    if extra_args:
        cmd.extend(extra_args)

    env = _build_env(stg_dir, downsample=1)  # STG doesn't use DYNERF_DOWNSAMPLE
    env["PYTHONPATH"] = f"{stg_dir}:{stg_dir / 'thirdparty' / 'gaussian_splatting'}:" \
                        + env.get("PYTHONPATH", "")

    console.print(
        f"  [dim]Training SpacetimeGaussians: {iterations:,} iterations "
        f"(source: {source_path.name}, downsample: {downsample}×)[/]"
    )
    if _vram_gb() < 20:
        console.print(
            "  [yellow]Warning: STG lite recommends 24 GB VRAM. "
            f"Detected ~{_vram_gb():.0f} GB — training may OOM. "
            "Consider --downsample 2 or fewer cameras.[/]"
        )

    t0 = time.time()
    result = subprocess.run(cmd, env=env, cwd=str(stg_dir))
    elapsed = time.time() - t0

    if result.returncode != 0:
        raise RuntimeError(f"STG training failed (exit code {result.returncode}).")

    console.print(f"  [green]STG training complete in {elapsed / 60:.1f} min[/]")

    # Bake keyframe PLY snapshots
    console.print(f"  Baking {n_keyframes} STG keyframe snapshots…")
    bake_script = Path(__file__).parent / "bake_stg_keyframes.py"
    keyframes_dir = model_path / "keyframes"
    keyframes_dir.mkdir(parents=True, exist_ok=True)

    bake_cmd = [
        python_exe, str(bake_script),
        "--model_path",  str(model_path.resolve()),
        "--n_keyframes", str(n_keyframes),
        "--output_dir",  str(keyframes_dir.resolve()),
    ]
    result2 = subprocess.run(bake_cmd, env=env, cwd=str(stg_dir))
    if result2.returncode != 0:
        raise RuntimeError(f"STG keyframe baking failed (exit code {result2.returncode}).")

    return model_path


def _get_stg_config(stg_dir: Path, source_path: Path, n_keyframes: int, downsample: int) -> Path:
    """Return an STG config path: use existing lite config or write a minimal one."""
    # Try to find an existing lite config
    lite_dir = stg_dir / "configs" / "n3d_lite"
    if lite_dir.exists():
        existing = sorted(lite_dir.glob("*.json"))
        if existing:
            # Use first available config as template; return it directly
            return existing[0]

    # Write a minimal config under model's parent
    cfg = {
        "model":          "ours_lite",
        "resolution":     downsample,
        "duration":       n_keyframes,
        "preprocesspoints": 3,
    }
    cfg_path = source_path / "_stg_config.json"
    import json as _json
    cfg_path.write_text(_json.dumps(cfg, indent=2))
    return cfg_path


def _vram_gb() -> float:
    """Return available GPU VRAM in GB, or 0 if undetectable."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / 1e9
    except Exception:
        pass
    return 0.0


# ── Gaussian-Flow training ─────────────────────────────────────────────────────

def train_gflow(
    source_path: Path,
    model_path: Path,
    gflow_dir: Path,
    *,
    iterations: int = 30_000,
    n_keyframes: int = 50,
    downsample: int = 2,
    train_python: str | None = None,
    extra_args: list[str] | None = None,
    gflow_config: "Path | None" = None,
) -> Path:
    """
    Run Gaussian-Flow training then bake keyframes.

    Two modes:
      • gflow_config=None  → original 4DGS-style Gaussian-Flow (train.py)
      • gflow_config=Path  → Pointrix Gaussian-Flow (launch.py); source_path is
                             the data dir built by prepare_gflow_pointrix_source()
    """
    train_script  = gflow_dir / "train.py"
    launch_script = gflow_dir / "launch.py"

    # ── Pointrix path ──────────────────────────────────────────────────────────
    if gflow_config is not None:
        if not launch_script.exists():
            raise FileNotFoundError(
                f"Gaussian-Flow/launch.py not found at {launch_script}.\n"
                "Clone: git clone https://github.com/NJU-3DV/Gaussian-Flow\n"
                "Set:   videosplat config --gflow-dir /path/to/Gaussian-Flow"
            )

        model_path.mkdir(parents=True, exist_ok=True)
        python_exe = train_python or sys.executable
        env = _build_env(gflow_dir, downsample=1)

        cmd = [
            python_exe, str(launch_script),
            "--config", str(gflow_config.resolve()),
        ]
        if extra_args:
            cmd.extend(extra_args)

        console.print(
            f"  [dim]Training Gaussian-Flow (Pointrix): {iterations:,} iterations "
            f"(source: {source_path.name})[/]"
        )

        def _raise_nofile():
            # 1550+ images opened simultaneously by Pointrix DataLoader threads
            soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
            target = max(65536, soft)
            resource.setrlimit(resource.RLIMIT_NOFILE, (min(target, hard), hard))

        # Opt-in CUDA sync mode for debugging CUDA faults (slows training ~2x).
        # Default off. Set VIDEOSPLAT_CUDA_SYNC=1 to enable.
        if os.environ.get("VIDEOSPLAT_CUDA_SYNC", "0") == "1":
            env["CUDA_LAUNCH_BLOCKING"] = "1"

        t0 = time.time()
        result = subprocess.run(cmd, env=env, cwd=str(gflow_dir), preexec_fn=_raise_nofile)
        elapsed = time.time() - t0

        if result.returncode != 0:
            raise RuntimeError(
                f"Gaussian-Flow (Pointrix) training failed (exit {result.returncode})."
            )

        console.print(f"  [green]Gaussian-Flow training complete in {elapsed / 60:.1f} min[/]")

        # Find the latest checkpoint saved by DefaultTrainer
        ckpts = sorted(
            model_path.glob("chkpnt*.pth"),
            key=lambda p: int(p.stem.replace("chkpnt", "") or 0),
        )
        if not ckpts:
            raise FileNotFoundError(f"No checkpoint (chkpnt*.pth) found in {model_path}")
        checkpoint = ckpts[-1]
        console.print(f"  [dim]Checkpoint: {checkpoint.name}[/]")

        # Bake keyframes via the Pointrix-specific script
        console.print(f"  Baking {n_keyframes} Pointrix keyframe snapshots…")
        bake_script   = Path(__file__).parent / "bake_gflow_pointrix_keyframes.py"
        keyframes_dir = model_path / "keyframes"
        keyframes_dir.mkdir(parents=True, exist_ok=True)

        bake_cmd = [
            python_exe, str(bake_script),
            "--checkpoint",  str(checkpoint.resolve()),
            "--config",      str(gflow_config.resolve()),
            "--gflow_dir",   str(gflow_dir.resolve()),
            "--n_keyframes", str(n_keyframes),
            "--output_dir",  str(keyframes_dir.resolve()),
        ]
        result2 = subprocess.run(bake_cmd, env=env, cwd=str(gflow_dir), preexec_fn=_raise_nofile)
        if result2.returncode != 0:
            raise RuntimeError(
                f"Pointrix keyframe baking failed (exit {result2.returncode})."
            )

        return model_path

    # ── Original 4DGS-style path ───────────────────────────────────────────────
    if not train_script.exists() and launch_script.exists():
        raise FileNotFoundError(
            f"The Gaussian-Flow repo at {gflow_dir} uses the Pointrix framework\n"
            "(launch.py present, train.py absent). Pass gflow_config= to use it,\n"
            "or switch to the original implementation:\n"
            "  git clone -b d3dgs_poly https://github.com/Linyou/D3DGS ~/GaussianFlow_orig\n"
            "  videosplat config --gflow-dir ~/GaussianFlow_orig"
        )

    if not train_script.exists():
        raise FileNotFoundError(
            f"Gaussian-Flow/train.py not found at {train_script}.\n"
            "Clone: git clone -b d3dgs_poly https://github.com/Linyou/D3DGS ~/GaussianFlow_orig\n"
            "Set:   videosplat config --gflow-dir /path/to/GaussianFlow_orig"
        )

    model_path.mkdir(parents=True, exist_ok=True)
    python_exe = train_python or sys.executable

    cmd = [
        python_exe, str(train_script),
        "--source_path",     str(source_path.resolve()),
        "--model_path",      str(model_path.resolve()),
        "--iterations",      str(iterations),
        "--save_iterations", str(iterations),
    ]
    if extra_args:
        cmd.extend(extra_args)

    env = _build_env(gflow_dir, downsample=downsample)

    console.print(
        f"  [dim]Training Gaussian-Flow: {iterations:,} iterations "
        f"(source: {source_path.name})[/]"
    )

    t0 = time.time()
    result = subprocess.run(cmd, env=env, cwd=str(gflow_dir))
    elapsed = time.time() - t0

    if result.returncode != 0:
        raise RuntimeError(f"Gaussian-Flow training failed (exit code {result.returncode}).")

    console.print(f"  [green]Gaussian-Flow training complete in {elapsed / 60:.1f} min[/]")

    console.print(f"  Baking {n_keyframes} Gaussian-Flow keyframe snapshots…")
    bake_script = Path(__file__).parent / "bake_gflow_keyframes.py"
    keyframes_dir = model_path / "keyframes"
    keyframes_dir.mkdir(parents=True, exist_ok=True)

    bake_cmd = [
        python_exe, str(bake_script),
        "--model_path",  str(model_path.resolve()),
        "--source_path", str(source_path.resolve()),
        "--n_keyframes", str(n_keyframes),
        "--output_dir",  str(keyframes_dir.resolve()),
    ]
    result2 = subprocess.run(bake_cmd, env=env, cwd=str(gflow_dir))
    if result2.returncode != 0:
        raise RuntimeError(f"Gaussian-Flow baking failed (exit code {result2.returncode}).")

    return model_path


# ── 4D-Rotor Gaussians training ────────────────────────────────────────────────

def train_4drotor(
    source_path: Path,
    model_path: Path,
    rotor_dir: Path,
    *,
    iterations: int = 30_000,
    n_keyframes: int = 50,
    train_python: str | None = None,
    extra_args: list[str] | None = None,
) -> Path:
    """
    Run 4D-Rotor Gaussians training via nerfstudio (ns-train splatfacto-big).

    source_path must contain transforms_train.json (from prepare_transforms_source()).
    Requires nerfstudio installed: pip install nerfstudio.

    Baking produces a single exported PLY (static snapshot); full temporal
    keyframe baking from nerfstudio is deferred to a follow-up implementation.
    """
    import shutil as _shutil
    ns_train = _shutil.which("ns-train")
    if not ns_train:
        raise FileNotFoundError(
            "ns-train not found on PATH. Install nerfstudio:\n"
            "  pip install nerfstudio\n"
            "  pip install -e /path/to/4D-Rotor-Gaussians"
        )

    model_path.mkdir(parents=True, exist_ok=True)

    cmd = [
        ns_train, "splatfacto-big",
        "--data",               str(source_path.resolve()),
        "--output-dir",         str(model_path.resolve()),
        "--max_num_iterations", str(iterations),
        "--vis",                "wandb" if _shutil.which("wandb") else "viewer",
    ]
    if extra_args:
        cmd.extend(extra_args)

    env = dict(os.environ)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if rotor_dir.exists():
        pp = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{rotor_dir}:{pp}" if pp else str(rotor_dir)

    console.print(
        f"  [dim]Training 4D-Rotor Gaussians (ns-train splatfacto-big): "
        f"{iterations:,} iterations[/]"
    )

    t0 = time.time()
    result = subprocess.run(cmd, env=env)
    elapsed = time.time() - t0

    if result.returncode != 0:
        raise RuntimeError(f"4D-Rotor training failed (exit code {result.returncode}).")

    console.print(f"  [green]4D-Rotor training complete in {elapsed / 60:.1f} min[/]")

    # Export PLY from nerfstudio output
    ns_export = _shutil.which("ns-export")
    if ns_export:
        # Find latest nerfstudio config
        configs = sorted(model_path.rglob("config.yml"))
        if configs:
            export_dir = model_path / "keyframes"
            export_dir.mkdir(exist_ok=True)
            export_cmd = [
                ns_export, "gaussian-splat",
                "--load-config",  str(configs[-1]),
                "--output-dir",   str(export_dir),
            ]
            result2 = subprocess.run(export_cmd, env=env)
            if result2.returncode == 0:
                # Rename exported PLY to keyframe_0000.ply (single static frame)
                exported = list(export_dir.glob("*.ply"))
                if exported:
                    exported[0].rename(export_dir / "keyframe_0000.ply")
                    console.print(
                        f"  [yellow]4D-Rotor: exported 1 static PLY. "
                        "Temporal keyframe baking not yet supported.[/]"
                    )
            else:
                console.print("  [yellow]ns-export failed — check nerfstudio output above.[/]")
    else:
        console.print("  [yellow]ns-export not found — skipping PLY export.[/]")

    return model_path
