# VideoSplat

Free-viewpoint video replay via 4D Gaussian Splatting. Turns a folder of
synchronised multi-camera videos into a trained 4D Gaussian model you can replay
from arbitrary viewpoints in the browser.

Pipeline (one command): camera videos → frame extraction + sync → camera
calibration (COLMAP or MASt3R) → 4DGS training → keyframe bake → web viewer.

## Install

```bash
pip install -e .            # or: uv pip install -e .
videosplat --help
```

Core deps (typer, rich, opencv, numpy, scipy, plyfile, pycolmap) install with the
package. Training backends are **external repos** you clone separately (below).

## Backend setup (required for training)

VideoSplat drives external Gaussian-Splatting backends as subprocesses. For the
default `--algo 4dgs`:

```bash
git clone https://github.com/hustvl/4DGaussians      # follow its own install (CUDA rasterizer, etc.)
videosplat config --backend-dir /path/to/4DGaussians
```

> [!IMPORTANT]
> **Apply the bundled patch to your 4DGaussians clone.** VideoSplat needs a
> one-line modification to 4DGaussians so the `--downsample` flag actually takes
> effect (upstream hardcodes the dynerf training resolution). Without it,
> `--downsample` is silently ignored and scenes train at full native resolution —
> which materially hurts quality/speed.
>
> ```bash
> cd /path/to/4DGaussians
> git apply /path/to/videosplat/patches/4dgaussians-dynerf-downsample.patch
> ```
>
> See [`patches/README.md`](patches/README.md) for what the patch changes and why
> it is load-bearing (the reference coffee_martini result, test PSNR ~28, was
> trained with `--downsample 2`, which only works once this patch is applied).

Other algorithms (`stg`, `gaussian-flow`, `4d-rotor`) use their own backend repos;
set their paths with `videosplat config --stg-dir / --gflow-dir / --rotor-dir`.

## Usage

```bash
# Full pipeline: videos dir (cam00.mp4 …) → trained model + viewer
videosplat run data/n3v/coffee_martini/ --output outputs/coffee --algo 4dgs --downsample 2

# Reopen an existing scene in the browser
videosplat view outputs/coffee
```

Key flags: `--algo`, `--downsample`, `--max-cameras`, `--calib-method`
(`colmap`|`mast3r`), `--iterations`, `--extra-args` (forwarded to the backend),
`--skip-sync`/`--skip-calibrate`/`--skip-train`.

## Repo layout

- `videosplat/` — pipeline (`sync`, `calibrate`, `convert`, `train`, `export`) + CLI.
- `patches/` — required modifications to external backend repos (see above).
- `Program.md` — durable project notes / autonomous-experiment context.
