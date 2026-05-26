"""
videosplat CLI

Commands:
  run           <videos_dir>     Full pipeline: camera videos → training → viewer
                                 --algo 4dgs|stg|gaussian-flow|4d-rotor
  prep          <videos_dir>     Preprocessing only: extract frames + run COLMAP
  render-video  <out_dir>        Orbit video renderer
  render-path   <out_dir>        Camera-path video renderer
  view          <out_dir>        Open an existing scene in the browser
  config                         Show / update backend algorithm paths
"""

import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

app = typer.Typer(
    name="videosplat",
    help="Free-viewpoint video replay via 4D Gaussian Splatting.",
    add_completion=False,
)
console = Console()


# ── preprocess ────────────────────────────────────────────────────────────────

@app.command()
def preprocess(
    scene: Annotated[Path, typer.Argument(
        help="Path to a raw N3V scene directory (containing cam00.mp4 … camNN.mp4)."
    )],
    start_frame: Annotated[int, typer.Option("--start-frame", help="First frame to process (default 0)")] = 0,
    end_frame: Annotated[int, typer.Option("--end-frame", help="Last frame to process (default 50; full scene is 300)")] = 50,
    downscale: Annotated[int, typer.Option("--downscale", help="Spatial downscale factor (default 1 = full res)")] = 1,
) -> None:
    """Preprocess a raw Neural 3D Video scene: extract frames and run per-frame COLMAP."""
    from videosplat.config import get_stg_dir, validate

    issues = validate()
    if issues:
        for issue in issues:
            console.print(f"[red]{issue}[/]")
        raise typer.Exit(1)

    if not scene.exists():
        console.print(f"[red]Scene not found: {scene}[/]")
        raise typer.Exit(1)

    stg_dir = get_stg_dir()
    pre_script = stg_dir / "script" / "pre_n3d.py"
    if not pre_script.exists():
        console.print(f"[red]Preprocessing script not found: {pre_script}[/]")
        raise typer.Exit(1)

    console.print(
        Panel(
            f"Scene:       [bold]{scene}[/]\n"
            f"Frames:      {start_frame} → {end_frame}  ({end_frame - start_frame} frames)\n"
            f"Downscale:   {downscale}×\n"
            f"Script:      {pre_script}",
            title="[bold cyan]VideoSplat preprocess[/]",
        )
    )

    cmd = [
        sys.executable, str(pre_script),
        "--videopath", str(scene.resolve()),
        "--startframe", str(start_frame),
        "--endframe", str(end_frame),
        "--downscale", str(downscale),
    ]

    # pre_n3d.py uses sys.path.append(".") — must run from STG root.
    # Inject colmap binary into PATH if it's not already on PATH.
    env = _env_with_colmap()
    result = subprocess.run(cmd, cwd=str(stg_dir), env=env)
    if result.returncode != 0:
        console.print("[red]Preprocessing failed — check output above.[/]")
        raise typer.Exit(result.returncode)

    console.print(f"[bold green]Preprocessing complete.[/] Scene ready at [cyan]{scene}[/]")


# ── run ────────────────────────────────────────────────────────────────────────

_ALGOS = ("4dgs", "stg", "gaussian-flow", "4d-rotor")

@app.command()
def run(
    source: Annotated[Path, typer.Argument(
        help="Directory of camera MP4/MOV files (or pre-extracted cam_XX/ dirs)."
    )],
    algo: Annotated[str, typer.Option("--algo",
        help=f"Training algorithm: {', '.join(_ALGOS)}")] = "4dgs",
    name: Annotated[str, typer.Option("--name", "-n", help="Scene label shown in the viewer")] = "",
    output: Annotated[Optional[Path], typer.Option("--output", "-o")] = None,
    label: Annotated[str, typer.Option("--label")] = "scene",
    pre_extracted: Annotated[bool, typer.Option("--pre-extracted")] = False,
    target_fps: Annotated[float, typer.Option("--fps")] = 10.0,
    max_dim: Annotated[int, typer.Option("--max-dim")] = 1280,
    audio_sync: Annotated[bool, typer.Option("--audio-sync/--no-audio-sync")] = True,
    camera_model: Annotated[str, typer.Option("--camera-model")] = "SIMPLE_PINHOLE",
    iterations: Annotated[int, typer.Option("--iterations")] = 14_000,
    n_keyframes: Annotated[int, typer.Option("--keyframes")] = 50,
    replay_fps: Annotated[float, typer.Option("--replay-fps")] = 8.0,
    max_cameras: Annotated[Optional[int], typer.Option("--max-cameras")] = None,
    downsample: Annotated[int, typer.Option("--downsample")] = 4,
    stg_config: Annotated[Optional[Path], typer.Option("--stg-config")] = None,
    train_python: Annotated[Optional[str], typer.Option("--train-python")] = None,
    extra_args: Annotated[Optional[str], typer.Option("--extra-args",
        help="Extra flags forwarded verbatim to the training script.")] = None,
    calib_method: Annotated[str, typer.Option("--calib-method",
        help="Calibration backend: colmap (default) or mast3r.")] = "colmap",
    static_cameras: Annotated[bool, typer.Option("--static-cameras",
        help="Cameras are fixed (MASt3R: sample a few frames, replicate pose to all).")] = False,
    calib_frames: Annotated[Optional[int], typer.Option("--calib-frames",
        help="Frames/camera for MASt3R calibration. Static default: 3. Moving default: all.")] = None,
    mast3r_niter: Annotated[int, typer.Option("--mast3r-niter",
        help="MASt3R global-alignment iterations (higher = better convergence, ~30s extra per 200 iters).")] = 500,
    mast3r_refine: Annotated[bool, typer.Option("--mast3r-refine/--no-mast3r-refine",
        help="Refine MASt3R poses + intrinsics with pycolmap (SIFT extract → match → "
             "triangulate + BA). Default OFF — empirically hurts on low-texture rigs "
             "(e.g. white domes, painted studios) where SIFT finds too few matches to "
             "outperform MASt3R's dense-matching prior. Enable on textured outdoor / "
             "object-centric scenes where SIFT consistently finds 100+ matches/pair.")] = False,
    mast3r_image_size: Annotated[int, typer.Option("--mast3r-image-size",
        help="Image size for MASt3R inference. Lower (e.g. 384) cuts GPU memory ~44%% so "
             "the Modular optimizer can fit; may give slightly lower pose quality. Default: 512.")] = 512,
    skip_sync: Annotated[bool, typer.Option("--skip-sync")] = False,
    skip_calibrate: Annotated[bool, typer.Option("--skip-calibrate")] = False,
    skip_train: Annotated[bool, typer.Option("--skip-train")] = False,
    no_view: Annotated[bool, typer.Option("--no-view")] = False,
) -> None:
    """Run the full pipeline: camera videos → 4D Gaussian Splatting → free-viewpoint replay."""
    if algo not in _ALGOS:
        console.print(f"[red]Unknown algo '{algo}'. Choose from: {', '.join(_ALGOS)}[/]")
        raise typer.Exit(1)
    if not source.exists():
        console.print(f"[red]Source not found: {source}[/]")
        raise typer.Exit(1)

    from videosplat.config import validate, get_backend_dir, get_stg_dir, get_gflow_dir, get_rotor_dir

    issues = validate(algo)
    if issues and not skip_train:
        for issue in issues:
            console.print(f"[red]{issue}[/]")
        raise typer.Exit(1)

    out_dir    = output or source.parent / f"{source.name}_splat4d"
    out_dir.mkdir(parents=True, exist_ok=True)
    scene_name = name or source.name.replace("_", " ").title()

    _print_header(scene_name, source, out_dir, iterations, algo=algo)

    # ── Step 1: Frame sync + extraction ───────────────────────────────────────
    sync_dir = out_dir / "sync"
    if not skip_sync:
        console.rule("[bold]Step 1/4  Frame Extraction & Sync")
        from videosplat.pipeline.sync import extract_and_sync
        sync_meta = extract_and_sync(
            source=source,
            out_dir=sync_dir,
            target_fps=target_fps,
            max_dim=max_dim,
            audio_sync=audio_sync,
            pre_extracted=pre_extracted,
        )
        console.print(
            f"  [green]✓ {sync_meta['n_cameras']} cameras × "
            f"{sync_meta['n_frames']} frames[/]"
        )
    else:
        console.print("[dim]Skipping sync — reusing existing sync/[/]")

    cam_dirs = sorted(sync_dir.glob("cam_*"))
    if not cam_dirs:
        console.print(f"[red]No cam_XX/ directories found in {sync_dir}[/]")
        raise typer.Exit(1)

    # ── Step 2: Camera calibration ────────────────────────────────────────────
    if not skip_calibrate:
        _method_label = calib_method.upper()
        console.rule(f"[bold]Step 2/4  Camera Calibration ({_method_label})")
        from videosplat.pipeline.calibrate import calibrate
        from videosplat.config import get_mast3r_dir
        cal_meta = calibrate(
            cam_dirs=cam_dirs,
            out_dir=out_dir,
            method=calib_method,
            camera_model=camera_model,
            fps=target_fps,
            static_cameras=static_cameras,
            calib_frames=calib_frames,
            mast3r_dir=get_mast3r_dir() if calib_method == "mast3r" else None,
            mast3r_niter=mast3r_niter,
            mast3r_refine=mast3r_refine and calib_method == "mast3r",
            mast3r_image_size=mast3r_image_size,
        )
        pct = cal_meta.get("pct_registered", 100)
        colour = "green" if pct >= 70 else "yellow"
        console.print(
            f"  [{colour}]Registered {cal_meta['n_registered']} images, "
            f"{cal_meta['n_points3D']:,} sparse points[/]"
        )
    else:
        console.print("[dim]Skipping calibration — reusing existing sparse/[/]")

    sparse_dir = out_dir / "sparse" / "0"
    if not sparse_dir.exists():
        console.print("[red]sparse/0/ not found — run calibration first[/]")
        raise typer.Exit(1)

    # ── Step 3: Algo-specific source prep + training ──────────────────────────
    model_dir = out_dir / "model"
    if not skip_train:
        console.rule(f"[bold]Step 3/4  Training ({algo})")
        _extra = extra_args.split() if extra_args else None

        if algo == "4dgs":
            from videosplat.pipeline.convert import frames_to_videos
            from videosplat.pipeline.train import train_4dgs, _build_env
            # 4DGS needs cam*.mp4 + poses_bounds.npy — calibrate.py already wrote these
            train_4dgs(
                source_path=out_dir,
                model_path=model_dir,
                stg_dir=get_backend_dir(),
                iterations=iterations,
                n_keyframes=n_keyframes,
                max_cameras=max_cameras,
                downsample=downsample,
                train_python=train_python,
                extra_args=_extra,
            )

        elif algo == "stg":
            from videosplat.pipeline.convert import prepare_stg_source
            from videosplat.pipeline.train import train_stg
            _stg_cam_dirs = list(cam_dirs)
            if max_cameras and max_cameras < len(_stg_cam_dirs):
                _stg_cam_dirs = sorted(_stg_cam_dirs)[:max_cameras]
                console.print(f"  [dim]max-cameras: using {len(_stg_cam_dirs)} cameras[/]")
            stg_source = prepare_stg_source(sparse_dir, _stg_cam_dirs, out_dir / "source_stg")
            train_stg(
                source_path=stg_source,
                model_path=model_dir,
                stg_dir=get_stg_dir(),
                iterations=iterations,
                n_keyframes=n_keyframes,
                downsample=downsample,
                stg_config=stg_config,
                train_python=train_python,
                extra_args=_extra,
            )

        elif algo == "gaussian-flow":
            from videosplat.pipeline.train import train_gflow
            _gflow_dir = get_gflow_dir()
            _pointrix  = (_gflow_dir / "launch.py").exists() and not (_gflow_dir / "train.py").exists()

            if _pointrix:
                from videosplat.pipeline.convert import prepare_gflow_pointrix_source
                gflow_source, _gflow_cfg = prepare_gflow_pointrix_source(
                    sparse_dir, cam_dirs, out_dir / "source_gflow",
                    iterations=iterations,
                    model_path=model_dir,
                    scale=round(1.0 / downsample, 4),
                )
            else:
                from videosplat.pipeline.convert import prepare_transforms_source
                gflow_source = prepare_transforms_source(sparse_dir, cam_dirs, out_dir / "source_gflow")
                _gflow_cfg = None

            train_gflow(
                source_path=gflow_source,
                model_path=model_dir,
                gflow_dir=_gflow_dir,
                iterations=iterations,
                n_keyframes=n_keyframes,
                downsample=downsample,
                train_python=train_python,
                extra_args=_extra,
                gflow_config=_gflow_cfg,
            )

        elif algo == "4d-rotor":
            from videosplat.pipeline.convert import prepare_transforms_source
            from videosplat.pipeline.train import train_4drotor
            rotor_source = prepare_transforms_source(sparse_dir, cam_dirs, out_dir / "source_rotor")
            train_4drotor(
                source_path=rotor_source,
                model_path=model_dir,
                rotor_dir=get_rotor_dir(),
                iterations=iterations,
                n_keyframes=n_keyframes,
                train_python=train_python,
                extra_args=_extra,
            )
    else:
        console.print("[dim]Skipping training — reusing existing model/[/]")

    keyframes_dir = model_dir / "keyframes"
    if not keyframes_dir.exists() or not list(keyframes_dir.glob("*.ply")):
        console.print(f"[red]No keyframe PLYs found in {keyframes_dir}[/]")
        raise typer.Exit(1)

    # ── Step 4: Export to viewer ───────────────────────────────────────────────
    console.rule("[bold]Step 4/4  Export to viewer")
    viewer_dir = out_dir / "viewer"
    _install_viewer(viewer_dir, scene_name)

    # Read intrinsics from COLMAP.
    # For STG: use colmap_0/sparse/0 which has only the N training cameras (camera-only
    # names). The master sparse/0 has N_cams × N_frames entries which produce a broken
    # 1350-camera orbit in the viewer.
    from videosplat.pipeline.export import export_keyframes
    _export_sparse = sparse_dir
    if algo == "stg":
        _stg_col0 = out_dir / "source_stg" / "colmap_0" / "sparse" / "0"
        if _stg_col0.exists():
            _export_sparse = _stg_col0
    _focal, _H, _W = _intrinsics_from_colmap(_export_sparse)
    exp_meta = export_keyframes(
        keyframes_dir=keyframes_dir,
        viewer_dir=viewer_dir,
        sparse_dir=_export_sparse,
        label=label,
        n_cameras=len(cam_dirs),
        fps=replay_fps,
        image_height=_H,
        image_width=_W,
        focal_length=_focal,
    )

    console.print(
        Panel(
            f"[bold green]Done![/]  {exp_meta['n_keyframes']} keyframes × "
            f"~{exp_meta['n_gaussians_per_frame']:,} Gaussians\n\n"
            f"Algo:   [cyan]{algo}[/]\n"
            f"Output: [cyan]{out_dir}[/]\n"
            f"Viewer: [cyan]{viewer_dir / 'index.html'}[/]\n\n"
            f"To reopen: [bold]videosplat view {out_dir}[/]",
            title=f"[bold]{scene_name}",
        )
    )

    if not no_view:
        _open_viewer(viewer_dir)


# ── prep ───────────────────────────────────────────────────────────────────────

@app.command()
def prep(
    source: Annotated[Path, typer.Argument(
        help="Directory of camera MP4/MOV files (or pre-extracted cam_XX/ dirs).")],
    output: Annotated[Optional[Path], typer.Option("--output", "-o")] = None,
    target_fps: Annotated[float, typer.Option("--fps")] = 10.0,
    max_dim: Annotated[int, typer.Option("--max-dim")] = 1280,
    audio_sync: Annotated[bool, typer.Option("--audio-sync/--no-audio-sync")] = True,
    pre_extracted: Annotated[bool, typer.Option("--pre-extracted")] = False,
    camera_model: Annotated[str, typer.Option("--camera-model")] = "SIMPLE_PINHOLE",
) -> None:
    """Run only frame extraction + COLMAP calibration (no training). Useful for inspecting results before committing to a full training run."""
    if not source.exists():
        console.print(f"[red]Source not found: {source}[/]")
        raise typer.Exit(1)

    out_dir = output or source.parent / f"{source.name}_prep"
    out_dir.mkdir(parents=True, exist_ok=True)

    console.rule("[bold]Step 1/2  Frame Extraction & Sync")
    from videosplat.pipeline.sync import extract_and_sync
    sync_meta = extract_and_sync(
        source=source,
        out_dir=out_dir / "sync",
        target_fps=target_fps,
        max_dim=max_dim,
        audio_sync=audio_sync,
        pre_extracted=pre_extracted,
    )
    cam_dirs = sorted((out_dir / "sync").glob("cam_*"))
    console.print(f"  [green]✓ {sync_meta['n_cameras']} cameras × {sync_meta['n_frames']} frames[/]")

    console.rule("[bold]Step 2/2  Camera Calibration (COLMAP)")
    from videosplat.pipeline.calibrate import calibrate
    cal_meta = calibrate(
        cam_dirs=cam_dirs,
        out_dir=out_dir,
        method="colmap",
        camera_model=camera_model,
        fps=target_fps,
    )
    pct = cal_meta.get("pct_registered", 100)
    colour = "green" if pct >= 70 else "yellow"
    console.print(
        f"  [{colour}]Registered {cal_meta['n_registered']} / "
        f"{cal_meta.get('n_images', '?')} images, "
        f"{cal_meta['n_points3D']:,} sparse points[/]"
    )

    focal, H, W = _intrinsics_from_colmap(out_dir / "sparse" / "0")
    console.print(
        Panel(
            f"Cameras:      {sync_meta['n_cameras']}\n"
            f"Frames:       {sync_meta['n_frames']}\n"
            f"Registered:   {pct:.0f}%\n"
            f"Intrinsics:   {W}×{H}  focal={focal:.1f} px\n\n"
            f"sparse/0/:    [cyan]{out_dir / 'sparse' / '0'}[/]\n"
            f"cam_XX/:      [cyan]{out_dir / 'sync'}[/]",
            title="[bold cyan]Preprocessing complete[/]",
        )
    )


# ── render-video ──────────────────────────────────────────────────────────────

@app.command(name="render-video")
def render_video(
    output_dir: Annotated[Path, typer.Argument(
        help="videosplat output directory (contains model/ and viewer/).")],
    arc: Annotated[float, typer.Option("--arc",
        help="Camera sweep arc in degrees (e.g. 180, 360).")] = 360.0,
    n_frames: Annotated[int, typer.Option("--frames",
        help="Number of frames in the output video.")] = 50,
    fps: Annotated[int, typer.Option("--fps")] = 24,
    width: Annotated[int, typer.Option("--width")] = 1280,
    height: Annotated[int, typer.Option("--height")] = 720,
    iterations: Annotated[int, typer.Option("--iterations")] = -1,
    white_bg: Annotated[bool, typer.Option("--white-bg")] = False,
    video_out: Annotated[Optional[Path], typer.Option("--save-file", "--output", "-o",
        help="Output MP4 path (default: <output_dir>/orbit_video.mp4).")] = None,
    radius_scale: Annotated[float, typer.Option("--radius-scale",
        help="Orbit radius = cloud_radius_p95 * radius_scale (increase to move camera further out).")] = 2.5,
    start_pos: Annotated[Optional[str], typer.Option("--start-pos",
        help="Starting camera position as 'X,Y,Z' (copy from viewer HUD). Overrides --radius-scale.")] = None,
    train_python: Annotated[Optional[str], typer.Option("--train-python")] = None,
) -> None:
    """Render an orbit video: camera sweeps arc_degrees while the scene plays."""
    from videosplat.config import get_backend_dir
    from videosplat.pipeline.train import _build_env

    model_path  = output_dir / "model"
    scene_meta  = output_dir / "viewer" / "scene_meta.json"
    if not model_path.exists():
        console.print(f"[red]Model not found at {model_path}[/]")
        raise typer.Exit(1)
    if not scene_meta.exists():
        console.print(f"[red]scene_meta.json not found — run the full pipeline first.[/]")
        raise typer.Exit(1)

    backend_dir = get_backend_dir()
    python_exe  = train_python or sys.executable
    out_mp4     = video_out or (output_dir / "orbit_video.mp4")

    render_script = Path(__file__).parent / "pipeline" / "render_orbit.py"
    env = _build_env(backend_dir)

    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        ffmpeg_exe = shutil.which("ffmpeg") or "ffmpeg"

    cmd = [
        python_exe, str(render_script),
        "--model_path",  str(model_path.resolve()),
        "--scene_meta",  str(scene_meta.resolve()),
        "--arc_degrees", str(arc),
        "--n_frames",    str(n_frames),
        "--fps",         str(fps),
        "--width",       str(width),
        "--height",      str(height),
        "--output",      str(out_mp4.resolve()),
    ]
    if iterations != -1:
        cmd += ["--iteration", str(iterations)]
    if white_bg:
        cmd.append("--white_bg")
    cmd += ["--radius_scale", str(radius_scale)]
    if start_pos:
        x, y, z = [s.strip() for s in start_pos.split(",")]
        cmd += ["--start_pos", x, y, z]
    cmd += ["--ffmpeg", ffmpeg_exe]

    console.print(
        f"  [dim]Rendering {n_frames}-frame orbit video "
        f"({arc}° arc, {width}×{height} @ {fps} fps)…[/]"
    )
    result = subprocess.run(cmd, env=env, cwd=str(backend_dir))
    if result.returncode != 0:
        console.print("[red]Render failed — check output above.[/]")
        raise typer.Exit(1)

    console.print(f"[green]Video saved → {out_mp4}[/]")


# ── path-editor ───────────────────────────────────────────────────────────────

@app.command(name="path-editor")
def path_editor(
    output_dir: Annotated[Path, typer.Argument(
        help="videosplat output directory (must contain viewer/ with scene_meta.json).")],
    port: Annotated[int, typer.Option("--port")] = 8081,
) -> None:
    """Open the camera path editor in the browser for a scene."""
    import shutil as _shutil

    viewer_dir = output_dir / "viewer"
    if not viewer_dir.exists():
        console.print(f"[red]Viewer dir not found: {viewer_dir}[/]")
        raise typer.Exit(1)

    # Copy / refresh path_editor.html into the viewer dir
    template = Path(__file__).parent / "viewer" / "static" / "path_editor.html"
    if not template.exists():
        console.print(f"[red]path_editor.html not found at {template}[/]")
        raise typer.Exit(1)

    scene_name = output_dir.name
    content = template.read_text().replace("{{SCENE_NAME}}", scene_name)
    dest = viewer_dir / "path_editor.html"
    dest.write_text(content)

    from videosplat.viewer.serve import serve as _serve
    console.print(f"[bold cyan]Path editor → [/]http://localhost:{port}/path_editor.html")
    console.print("Navigate to positions, add waypoints, then [bold]Export path.json[/].")
    console.print("Place the downloaded JSON next to your output dir, then run:")
    console.print(f"  [dim]videosplat render-path <source> {output_dir} --path camera_path.json[/]")

    import os, http.server, socket, threading, time, webbrowser

    os.chdir(viewer_dir)
    for candidate in range(port, port + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("localhost", candidate)) != 0:
                port = candidate
                break

    class _Handler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, fmt, *a): pass
        def end_headers(self):
            self.send_header("Cross-Origin-Opener-Policy", "same-origin")
            self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
            if self.path.split("?")[0].endswith((".html", ".js")):
                self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            super().end_headers()

    def _open():
        time.sleep(0.5)
        webbrowser.open(f"http://localhost:{port}/path_editor.html")
    threading.Thread(target=_open, daemon=True).start()

    with http.server.HTTPServer(("", port), _Handler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            console.print("\n[dim]Editor stopped.[/]")


# ── render-path ───────────────────────────────────────────────────────────────

@app.command(name="render-path")
def render_path_cmd(
    output_dir: Annotated[Path, typer.Argument(
        help="videosplat output directory (contains model/ and viewer/).")],
    path_json: Annotated[Path, typer.Option("--path",
        help="camera_path.json exported from the path editor.")] = Path("camera_path.json"),
    n_frames: Annotated[int, typer.Option("--frames")] = 120,
    fps: Annotated[int, typer.Option("--fps")] = 24,
    width: Annotated[int, typer.Option("--width")] = 1280,
    height: Annotated[int, typer.Option("--height")] = 720,
    iterations: Annotated[int, typer.Option("--iterations")] = -1,
    white_bg: Annotated[bool, typer.Option("--white-bg")] = False,
    video_out: Annotated[Optional[Path], typer.Option("--save-file", "--output", "-o")] = None,
    forward_shift: Annotated[float, typer.Option("--forward-shift",
        help="Shift camera forward by this many world units (compensates viewer/renderer FOV mismatch).")] = 0.0,
    configs: Annotated[Optional[str], typer.Option("--configs",
        help="4DGaussians config the model was trained with — REQUIRED for casual/nerfies models (e.g. arguments/hypernerf/default.py).")] = None,
    train_python: Annotated[Optional[str], typer.Option("--train-python")] = None,
) -> None:
    """Render a video along a camera path exported from the path editor."""
    from videosplat.config import get_backend_dir
    from videosplat.pipeline.train import _build_env

    model_path = output_dir / "model"
    scene_meta = output_dir / "viewer" / "scene_meta.json"
    if not model_path.exists():
        console.print(f"[red]Model not found at {model_path}[/]")
        raise typer.Exit(1)
    if not path_json.exists():
        console.print(f"[red]Path JSON not found: {path_json}[/]")
        console.print("Export it from:  videosplat path-editor <output_dir>")
        raise typer.Exit(1)
    if not scene_meta.exists():
        console.print(f"[red]scene_meta.json not found — run the full pipeline first.[/]")
        raise typer.Exit(1)

    backend_dir = get_backend_dir()
    python_exe  = train_python or sys.executable
    out_mp4     = video_out or (output_dir / "path_video.mp4")

    render_script = Path(__file__).parent / "pipeline" / "render_path.py"
    env = _build_env(backend_dir)

    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        ffmpeg_exe = shutil.which("ffmpeg") or "ffmpeg"

    cmd = [
        python_exe, str(render_script),
        "--model_path",  str(model_path.resolve()),
        "--scene_meta",  str(scene_meta.resolve()),
        "--path_json",   str(path_json.resolve()),
        "--n_frames",    str(n_frames),
        "--fps",         str(fps),
        "--width",       str(width),
        "--height",      str(height),
        "--output",      str(out_mp4.resolve()),
        "--ffmpeg",      ffmpeg_exe,
    ]
    if iterations != -1:
        cmd += ["--iteration", str(iterations)]
    if configs:
        cmd += ["--configs", str(configs)]
    if white_bg:
        cmd.append("--white_bg")
    if forward_shift != 0.0:
        cmd += ["--forward_shift", str(forward_shift)]

    console.print(
        f"  [dim]Rendering {n_frames}-frame path video "
        f"({width}×{height} @ {fps} fps)…[/]"
    )
    result = subprocess.run(cmd, env=env, cwd=str(backend_dir))
    if result.returncode != 0:
        console.print("[red]Render failed — check output above.[/]")
        raise typer.Exit(1)

    console.print(f"[green]Video saved → {out_mp4}[/]")


# ── casual (moving / heterogeneous / unsynced multi-cam → nerfies 4DGS) ─────────

@app.command()
def casual(
    source: Annotated[Path, typer.Argument(help="Dir of camN.mp4 casual captures (moving/heterogeneous/unsynced).")],
    output: Annotated[Optional[Path], typer.Option("--output", "-o")] = None,
    name: Annotated[str, typer.Option("--name", "-n", help="Scene label")] = "",
    moving_cams: Annotated[str, typer.Option("--moving-cams", help="Comma-sep indices of cameras that MOVE, e.g. '2' or '0,2'. Empty = all static.")] = "",
    holdout_cams: Annotated[str, typer.Option("--holdout-cams", help="Comma-sep camera indices to hold out ENTIRELY for novel-VIEW eval (e.g. '2,6'). Empty = held-out-time split.")] = "",
    audio_sync: Annotated[bool, typer.Option("--audio-sync/--no-audio-sync", help="Cross-correlate audio to align cameras. Disable for hardware-synced rigs (e.g. AIST) to avoid spurious sub-frame offsets.")] = True,
    # density / sampling
    n_time: Annotated[int, typer.Option("--n-time", help="Temporal samples (dominant quality lever).")] = 300,
    n_keyframes: Annotated[int, typer.Option("--n-keyframes", help="MASt3R calib keyframes per moving cam.")] = 18,
    static_calib_frames: Annotated[int, typer.Option("--static-calib-frames")] = 3,
    seg_start: Annotated[float, typer.Option("--seg-start", help="Clip window start (s, cam0 clock).")] = 3.0,
    seg_end: Annotated[float, typer.Option("--seg-end", help="Clip window end (s); <=0 = full overlap.")] = -1.0,
    width: Annotated[int, typer.Option("--width")] = 1024,
    height: Annotated[int, typer.Option("--height")] = 576,
    fps: Annotated[float, typer.Option("--fps")] = 30.0,
    # calibration
    mast3r_size: Annotated[int, typer.Option("--mast3r-size", help="MASt3R res (↓=less VRAM, worse calib).")] = 512,
    mast3r_niter: Annotated[int, typer.Option("--mast3r-niter")] = 400,
    edge_thr_modular: Annotated[int, typer.Option("--modular-above", help="Use lower-VRAM Modular optimizer above this image count.")] = 24,
    # person conf-downweighting
    mask_person: Annotated[bool, typer.Option("--mask-person/--no-mask-person", help="Downweight the moving subject in the pose solve.")] = True,
    mask_downweight: Annotated[float, typer.Option("--mask-downweight", help="Downweight strength 0..1 (1=fully ignore person's pixels).")] = 1.0,
    mask_dilate: Annotated[int, typer.Option("--mask-dilate")] = 9,
    mask_score: Annotated[float, typer.Option("--mask-score", help="Person-detector score threshold.")] = 0.7,
    # init cloud + training
    init_conf_thr: Annotated[float, typer.Option("--init-conf-thr")] = 1.5,
    max_init_pts: Annotated[int, typer.Option("--max-init-pts")] = 100_000,
    iterations: Annotated[int, typer.Option("--iterations")] = 14_000,
    bake_keyframes: Annotated[int, typer.Option("--keyframes", help="Keyframes to bake for the viewer.")] = 100,
    # viewer floater pruning (free-orbit viewers fly through faint/huge floaters → haze)
    prune_opacity: Annotated[float, typer.Option("--prune-opacity", help="Cull viewer Gaussians below this opacity (0=off).")] = 0.05,
    prune_scale_mult: Annotated[float, typer.Option("--prune-scale-mult", help="Cull Gaussians with max-scale > mult×p99 (0=off).")] = 5.0,
    prune_dist_mult: Annotated[float, typer.Option("--prune-dist-mult", help="Cull Gaussians > mult×p90-radius from center (0=off).")] = 3.0,
    configs: Annotated[Optional[str], typer.Option("--configs", help="4DGaussians config (default: backend hypernerf/default.py).")] = None,
    opt_override: Annotated[list[str], typer.Option("--opt-override", help="Override any OptimizationParams knob, repeatable, key=val (e.g. --opt-override opacity_reset_interval=3000 --opt-override lambda_dssim=0.2).")] = [],
    # safety
    vram_guard: Annotated[int, typer.Option("--vram-guard", help="Kill the run if total GPU VRAM exceeds this many MiB (0=off). Protects a display-shared GPU.")] = 12000,
    skip_build: Annotated[bool, typer.Option("--skip-build", help="Reuse existing nerfies dataset in output/.")] = False,
    skip_train: Annotated[bool, typer.Option("--skip-train")] = False,
) -> None:
    """Casual multi-camera → 4DGaussians (nerfies). Handles MOVING + heterogeneous
    + unsynced cameras that the dynerf `run` path cannot. Every knob is a flag."""
    import os
    from videosplat.config import get_backend_dir, get_mast3r_dir
    from videosplat.pipeline.casual_capture import build_casual_nerfies_dataset, export_casual_viewer
    from videosplat.pipeline.train import train_nerfies

    out_dir = (output or Path("outputs") / f"{source.name}_casual").resolve()
    backend = get_backend_dir(); mast3r = get_mast3r_dir()
    cfg = configs or str(backend / "arguments" / "hypernerf" / "default.py")
    mcams = tuple(int(x) for x in moving_cams.split(",") if x.strip() != "")
    hcams = tuple(int(x) for x in holdout_cams.split(",") if x.strip() != "")
    def _coerce(v: str):
        try: return int(v)
        except ValueError:
            try: return float(v)
            except ValueError: return v
    opt_ovr = {k.strip(): _coerce(v.strip()) for k, v in (o.split("=", 1) for o in opt_override)}
    seg = None if seg_end <= 0 else (seg_start, seg_end)
    try:
        import imageio_ffmpeg; ff = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        ff = shutil.which("ffmpeg") or "ffmpeg"

    # self-protect the shared display-GPU: own process group + VRAM guardian
    guard = None
    if vram_guard > 0:
        try: os.setpgrp()
        except OSError: pass
        gscript = Path(__file__).parent / ".." / ".." / ".lab" / "bin" / "vram_guard"
        gscript = gscript if gscript.exists() else (Path(__file__).parent / "vram_guard")
        if Path(gscript).exists():
            guard = subprocess.Popen(["bash", str(gscript), str(os.getpgid(0)), str(vram_guard)],
                                     start_new_session=True)
            console.print(f"  [dim]VRAM guardian active (kill > {vram_guard} MiB)[/]")
    try:
        console.rule(f"[bold]Casual capture → nerfies 4DGS  ({source.name})")
        if not skip_build:
            build_casual_nerfies_dataset(
                source, out_dir, ffmpeg=ff, mast3r_dir=mast3r, moving_cams=mcams,
                n_time=n_time, n_keyframes=n_keyframes, seg=seg, W=width, H=height, fps=fps,
                mast3r_size=mast3r_size, mast3r_niter=mast3r_niter, mask_person=mask_person,
                mask_downweight=mask_downweight, mask_dilate=mask_dilate, mask_score=mask_score,
                init_conf_thr=init_conf_thr, max_init_pts=max_init_pts,
                static_calib_frames=static_calib_frames, edge_thr_modular=edge_thr_modular,
                holdout_cams=hcams, audio_sync=audio_sync,
            )
        if not skip_train:
            train_nerfies(out_dir, out_dir / "model", backend, configs=cfg,
                          iterations=iterations, n_keyframes=bake_keyframes,
                          train_python=sys.executable, prune_opacity=prune_opacity,
                          prune_scale_mult=prune_scale_mult, prune_dist_mult=prune_dist_mult,
                          opt_overrides=opt_ovr)
            export_casual_viewer(out_dir, label=(name or source.name))
        console.print(Panel(
            f"Done!  nerfies model + viewer → {out_dir}\n\n"
            f"Render a fly-through:  videosplat render-path {out_dir} --configs {cfg} --frames {int(107*fps)} --fps {int(fps)}\n"
            f"Open the viewer:       videosplat view {out_dir}",
            title="[bold cyan]VideoSplat casual[/]", border_style="green"))
    finally:
        if guard: guard.terminate()


# ── view ───────────────────────────────────────────────────────────────────────

@app.command()
def view(
    output_dir: Annotated[Path, typer.Argument(help="splat output directory containing viewer/")],
    port: Annotated[int, typer.Option("--port")] = 8080,
) -> None:
    """Open an existing scene in the browser."""
    viewer_dir = output_dir / "viewer"
    if not (viewer_dir / "index.html").exists():
        console.print(f"[red]Viewer not found in {output_dir}[/]")
        raise typer.Exit(1)
    _open_viewer(viewer_dir, port=port)


# ── config ─────────────────────────────────────────────────────────────────────

@app.command()
def config(
    backend_dir: Annotated[Optional[Path], typer.Option("--backend-dir",
        help="Path to 4DGaussians repo")] = None,
    stg_dir: Annotated[Optional[Path], typer.Option("--stg-dir",
        help="Path to SpacetimeGaussians repo")] = None,
    gflow_dir: Annotated[Optional[Path], typer.Option("--gflow-dir",
        help="Path to Gaussian-Flow repo")] = None,
    rotor_dir: Annotated[Optional[Path], typer.Option("--rotor-dir",
        help="Path to 4D-Rotor-Gaussians repo")] = None,
    mast3r_dir: Annotated[Optional[Path], typer.Option("--mast3r-dir",
        help="Path to MASt3R repo (github.com/naver/mast3r)")] = None,
) -> None:
    """Show or update algorithm backend paths."""
    from videosplat.config import (save, get_backend_dir, get_stg_dir,
                                   get_gflow_dir, get_rotor_dir, get_mast3r_dir)
    import shutil as _sh

    if any(x is not None for x in (backend_dir, stg_dir, gflow_dir, rotor_dir, mast3r_dir)):
        save(backend_dir=backend_dir, stg_dir=stg_dir,
             gflow_dir=gflow_dir, rotor_dir=rotor_dir, mast3r_dir=mast3r_dir)
        console.print("[green]Config saved.[/]")

    def _ok(p: Path, check: str = "train.py") -> str:
        return "[green]OK[/]" if (p / check).exists() else "[red]NOT FOUND[/]"

    t = Table(title="VideoSplat config")
    t.add_column("Algo",    style="bold")
    t.add_column("Setting")
    t.add_column("Path")
    t.add_column("Status")

    d = get_backend_dir()
    t.add_row("4dgs",          "backend_dir", str(d),             _ok(d))
    d = get_stg_dir()
    t.add_row("stg",           "stg_dir",     str(d),             _ok(d))
    d = get_gflow_dir()
    gflow_ok = (d / "launch.py").exists() or (d / "train.py").exists()
    t.add_row("gaussian-flow", "gflow_dir",   str(d),
              "[green]OK[/]" if gflow_ok else "[red]NOT FOUND[/]")
    d = get_rotor_dir()
    t.add_row("4d-rotor",      "rotor_dir",   str(d),
              "[green]OK[/]" if _sh.which("ns-train") else "[yellow]ns-train not on PATH[/]")
    d = get_mast3r_dir()
    ckpt_ok = any(d.glob("checkpoints/*.pth"))
    t.add_row("mast3r",        "mast3r_dir",  str(d),
              "[green]OK[/]" if ckpt_ok else "[red]no checkpoint[/]")
    console.print(t)


# ── Helpers ─────────────────────────────────────────────────────────────────────

def _print_header(name: str, source: Path, out_dir: Path, iterations: int,
                  algo: str = "4dgs") -> None:
    console.print(
        Panel(
            f"Scene:      [bold]{name}[/]\n"
            f"Source:     {source}\n"
            f"Output:     {out_dir}\n"
            f"Algorithm:  [cyan]{algo}[/]\n"
            f"Iterations: {iterations:,}",
            title="[bold cyan]VideoSplat — 4D Gaussian Splatting replay[/]",
        )
    )


def _intrinsics_from_colmap(sparse_dir: Path) -> tuple[float, int, int]:
    """Return (focal_length, height, width) from COLMAP cameras.bin."""
    try:
        from videosplat.pipeline.convert import _read_cameras_bin, _focal_from_params
        cams = _read_cameras_bin(sparse_dir / "cameras.bin")
        if cams:
            c = next(iter(cams.values()))
            focal = _focal_from_params(c["model"], c["params"])
            return focal, int(c["height"]), int(c["width"])
    except Exception:
        pass
    return 0.0, 0, 0


def _install_viewer(viewer_dir: Path, scene_name: str) -> None:
    viewer_dir.mkdir(parents=True, exist_ok=True)
    static_src = Path(__file__).parent / "viewer" / "static"
    for asset in static_src.iterdir():
        if asset.suffix in (".html", ".js", ".css"):
            shutil.copy2(asset, viewer_dir / asset.name)
    index = viewer_dir / "index.html"
    if index.exists():
        html = index.read_text()
        html = html.replace("{{SCENE_NAME}}", scene_name)
        index.write_text(html)


def _open_viewer(viewer_dir: Path, port: int = 8080) -> None:
    from videosplat.viewer.serve import serve
    serve(viewer_dir, port=port)


def _env_with_colmap() -> dict:
    """Return os.environ with a colmap binary injected into PATH if needed."""
    import os
    import shutil
    env = dict(os.environ)
    if shutil.which("colmap"):
        return env  # already on PATH

    candidates = [
        Path.home() / "miniconda3/envs/neural-rendering/bin",
        Path.home() / "anaconda3/envs/neural-rendering/bin",
        Path("/usr/bin"),
        Path("/usr/local/bin"),
    ]
    for d in candidates:
        if (d / "colmap").exists():
            env["PATH"] = f"{d}:{env.get('PATH', '')}"
            console.print(f"  [dim]colmap found at {d}/colmap[/]")
            return env

    console.print(
        "[yellow]colmap binary not found. Install it:[/]\n"
        "  sudo apt install colmap\n"
        "  or: conda install -c conda-forge colmap"
    )
    return env


def _infer_sync_meta(sync_dir: Path) -> dict:
    cam_dirs = sorted(sync_dir.glob("cam_*"))
    n_frames = min((len(list(d.glob("*.jpg"))) for d in cam_dirs), default=0)
    return {"n_cameras": len(cam_dirs), "n_frames": n_frames, "fps": None}
