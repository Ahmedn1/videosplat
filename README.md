# videosplat

**Free-viewpoint replay of dynamic scenes from multi-camera video, via 4D Gaussian Splatting.**

Point it at a folder of `camNN.mp4` clips of the same moment — a synced rig, a casual
handful of phones, even moving/unsynced cameras — and it calibrates, trains a 4D
Gaussian model, and bakes a browser viewer you can scrub and fly through.

```bash
videosplat run data/n3v/coffee_martini --algo 4dgs --name "Coffee Martini"
```

[![Live Demo](https://img.shields.io/badge/🔗_Live_Demo-Netlify-blue)](https://videosplat-demo.netlify.app)

The live demo is a gallery of five reconstructions — each with a rendered fly-through
video **and** a single-frame interactive splat you can orbit in-browser. The full 4D
timeline player (every keyframe, view-dependent colour) runs locally via
`videosplat view` (per-scene keyframes are multiple GB, too big to host).

> Read **[Findings.md](Findings.md)** for the experimental write-up across all five datasets
> (what worked, what didn't, and why).

---

## How it works

```
 camNN.mp4 ──► extract ──► audio sync ──► calibrate ──► convert ──► train ──► bake ──► viewer
 (multi-cam)   frames      (optional)    COLMAP /       per-algo    4DGS/STG/  keyframe   browser
                                          MASt3R         format      GF/Rotor   PLYs       (scrub + fly)
```

1. **Extract** frames from each camera at a target FPS.
2. **Sync** unsynced clips by audio cross-correlation (optional; off for hardware-synced rigs).
3. **Calibrate** camera poses + intrinsics with **COLMAP** or **MASt3R**.
4. **Convert** to the format the chosen backbone expects.
5. **Train** one of four 4D Gaussian backbones.
6. **Bake** N keyframe PLY snapshots and assemble a static web viewer.

---

## Prerequisites

| Requirement | Notes |
|---|---|
| NVIDIA GPU + CUDA | 16 GB works (the pipeline is VRAM-budget-aware; see the VRAM guardian) |
| Python ≥ 3.10 | |
| COLMAP | for the COLMAP calibration path (`pycolmap` is a Python dep) |
| ffmpeg | frame extraction / video encode |

**External backbone repos** (clone whichever algos you'll use, then register their paths):

| Algo | Repo |
|---|---|
| `4dgs` | [4DGaussians](https://github.com/hustvl/4DGaussians) |
| `stg` | [SpacetimeGaussians](https://github.com/oppo-us-research/SpacetimeGaussians) |
| `gaussian-flow` | [Gaussian-Flow](https://github.com/Linyou/Gaussian-Flow) / Pointrix |
| `4d-rotor` | [4D-Rotor-Gaussians](https://github.com/weify627/4D-Rotor-Gaussians) (via nerfstudio) |
| calibration | [MASt3R](https://github.com/naver/mast3r) (for the MASt3R path) |

---

## Installation

```bash
git clone https://github.com/Ahmedn1/videosplat
cd videosplat
pip install -e .            # or: uv pip install -e .

# point videosplat at your cloned backbones (stored in ~/.videosplat/config.json)
videosplat config --backend-dir ~/4DGaussians \
                   --stg-dir    ~/SpacetimeGaussians \
                   --gflow-dir  ~/Gaussian-Flow \
                   --rotor-dir  ~/4D-Rotor-Gaussians \
                   --mast3r-dir ~/mast3r
videosplat config            # verify resolved paths
```

**Required 4DGaussians patch.** The upstream dynerf reader hardcodes the downsample
factor to 1.0 (full native resolution). Apply the bundled patch so `--downsample`
is honoured (this is load-bearing — `--downsample 2` is what makes N3V training fit
in 16 GB and reach published quality):

```bash
cd ~/4DGaussians && git apply /path/to/videosplat/patches/4dgaussians-dynerf-downsample.patch
```

---

## Quick start

```bash
# Synced multi-camera rig (e.g. N3V) → 4DGaussians
videosplat run data/n3v/coffee_martini --algo 4dgs --downsample 2 --name "Coffee Martini"

# Casual / moving / heterogeneous / unsynced cameras → nerfies 4DGS
videosplat casual data/piano --name "Piano" --n-time 300

# Open the baked 4D viewer (local timeline player)
videosplat view outputs/coffee_martini_splat4d
```

---

## The four reconstruction algorithms

Choose with `--algo` on `videosplat run`. All share the calibration + viewer steps;
they differ in the 4D representation and training backend.

### `4dgs` — 4DGaussians (default)
Deformation-field 4DGS. The all-rounder; best fidelity on calibrated synced rigs, and
the only path that also handles **moving/heterogeneous** cameras (via the nerfies
reader — see `videosplat casual`).

```bash
videosplat run <scene> --algo 4dgs --downsample 2 --iterations 14000
```

| Param | Preferred | Why |
|---|---|---|
| `--iterations` | `14000` | densification plateaus ~15k; more iters just memorise noise |
| `--downsample` | `2` | 4× fewer pixels; fits 16 GB and *raises* novel-view PSNR (needs the patch above) |
| `--max-cameras` | all (omit) | use every training camera — coverage is the dominant novel-view lever |

### `stg` — SpacetimeGaussians
Per-Gaussian polynomial motion; real-time-render capable, memory-friendly. Calibration
is **per-frame COLMAP (~1h)** vs the single 24h COLMAP that `4dgs` needs on N3V — so
it's the right choice for fast iteration.

```bash
videosplat run <scene> --algo stg --downsample 2 --iterations 20000 [--stg-config cfg.json]
```

### `gaussian-flow` — Gaussian-Flow (Pointrix or original)
Flow-based dynamic GS. Auto-detects the Pointrix `launch.py` vs the original `train.py`.

```bash
videosplat run <scene> --algo gaussian-flow --iterations 30000
```

### `4d-rotor` — 4D-Rotor-Gaussians
Rotor (geometric-algebra) 4D GS, trained through nerfstudio (`ns-train splatfacto-big`).
Note: temporal keyframe baking isn't wired yet — exports a single static frame.

```bash
videosplat run <scene> --algo 4d-rotor --iterations 30000
```

| Algo | Default iters | Calibration cost | Best for |
|---|---|---|---|
| `4dgs` | 14000 | 24h COLMAP (or MASt3R) | fidelity; moving/heterogeneous cams |
| `stg` | 20000 | ~1h per-frame COLMAP | fast iteration |
| `gaussian-flow` | 30000 | COLMAP | flow-based dynamics |
| `4d-rotor` | 30000 | COLMAP | rotor representation (static export) |

---

## Calibration: COLMAP vs MASt3R

Pick with `--calib-method` (`videosplat run`) — MASt3R is always used by `videosplat casual`.

**COLMAP** (`--calib-method colmap`, default). SIFT features + bundle adjustment.
Fast and accurate on **textured, synced rigs**. Uses a few-frame static-rig extraction
(default 15 frames) — far faster and *more* accurate than feeding all frames (dynamic
foreground in extra frames biases the shared intrinsic). Camera model:
`--camera-model SIMPLE_PINHOLE`.

**MASt3R** (`--calib-method mast3r`). Learned dense matching — registers cameras where
SIFT fails: **wide baselines, low/no texture, top-down or unusual angles, unsynced clips.**
Extracts **per-camera** intrinsics (not one shared focal).

```bash
videosplat run <scene> --calib-method mast3r --static-cameras --calib-frames 3 --mast3r-size 512
```

| Flag | Default | Notes |
|---|---|---|
| `--static-cameras` | off | fixed rig: sample a few frames/cam, average the pose |
| `--calib-frames` | 3 (static) / all (moving) | frames per camera fed to MASt3R |
| `--mast3r-size` | 512 | inference resolution (↓ = less VRAM, worse calib) |
| `--mast3r-niter` | 500 | global-alignment iterations |
| `--mast3r-refine` | off | optional pycolmap SIFT refinement — only helps on textured outdoor scenes (SIFT finds too few matches on low-texture; default off) |

> MASt3R complete-graph matching OOMs above ~22 images on 16 GB — keep the calibration
> set small (≤ ~18 frames) or use the lower-VRAM modular optimizer (`--modular-above`).

---

## Casual capture (moving / heterogeneous / unsynced)

`videosplat casual` handles captures the synced `run` path can't: cameras that **move**,
have **different resolutions**, and start at **different times**. It audio-syncs, calibrates
with MASt3R (per-camera intrinsics, per-frame poses for moving cams), letterboxes to a
common resolution, writes a nerfies/HyperNeRF dataset, and trains 4DGaussians.

```bash
videosplat casual data/piano --name Piano \
    --moving-cams 2 --n-time 300 \
    --holdout-cams 2,6 --no-audio-sync          # novel-view eval / synced-rig
```

Key knobs (all defaulted; everything is a flag):

| Group | Flags |
|---|---|
| cameras | `--moving-cams 0,2` · `--holdout-cams 2,6` (held-out novel-**view** eval) · `--audio-sync/--no-audio-sync` |
| temporal | `--n-time 300` (the dominant quality lever) · `--n-keyframes` · `--seg-start/--seg-end` |
| calibration | `--mast3r-size 512` · `--static-calib-frames 1` · `--modular-above 24` |
| person mask | `--mask-person/--no-mask-person` · `--mask-downweight` · `--mask-dilate` · `--mask-score` |
| init / train | `--init-conf-thr` · `--max-init-pts` · `--iterations 14000` · `--opt-override key=val` |
| viewer prune | `--prune-opacity` · `--prune-scale-mult` · `--prune-dist-mult` |
| safety | `--vram-guard 12000` |

### Person masking & confidence-downweighting

A moving subject (a pianist's hands, a dancer) corrupts a static pose solve — but
**blacking it out is worse**: it also removes the static structure right next to the
person (keyboard, contact points) that the solver needs. So videosplat **downweights**
instead of masks: it detects people (Mask R-CNN) and lowers their pixels' *confidence*
in the MASt3R global alignment, keeping the full image intact.

```
conf_person ← 1 + (conf − 1)·(1 − s)      # s = --mask-downweight, 1.0 = fully ignore
```

| Flag | Default | Effect |
|---|---|---|
| `--mask-person` / `--no-mask-person` | on | enable/disable downweighting in the pose solve |
| `--mask-downweight` | `1.0` | strength `s∈[0,1]`; `1.0` = person pixels don't vote on pose |
| `--mask-dilate` | `9` | grow the mask (px) to cover motion-blur edges |
| `--mask-score` | `0.7` | detector confidence threshold |

On the piano capture this lifted frontier PSNR from 18.98 (black-out) → 23.31 (unmasked)
→ **23.65 (downweight)**. See [Findings.md](Findings.md).

---

## Output structure

```
outputs/<scene>_splat4d/
├── sparse/0/                 # COLMAP/MASt3R calibration (cameras, images, points3D)
├── poses_bounds.npy          # LLFF poses (4dgs/stg)
├── camNN.mp4                 # extracted per-camera clips
├── model/
│   ├── point_cloud/iteration_NNNNN/   # trained 4D model
│   └── keyframes/keyframe_*.ply       # baked per-timestep snapshots
└── viewer/                   # static web viewer (deployable)
    ├── index.html            # 4D timeline player (scrub + fly-through)
    ├── scene_meta.json       # cameras, bounds, fps, n_keyframes
    └── frames/keyframe_*.ply
```

---

## Web viewer & live demo

**Local** — the full 4D player (timeline scrubber, play/pause, free orbit):

```bash
videosplat view outputs/<scene>_splat4d        # serves viewer/ on localhost
```

**Hosted demo** — `docs/` is a self-contained static gallery for Netlify (publish dir
`docs`, configured in `netlify.toml`). Each scene card embeds a rendered fly-through
plus a single-frame interactive `.splat`. Rebuild the demo assets with:

```bash
python scripts/build_demo.py        # PLY→.splat + copies videos → docs/scenes/
cd docs && python -m http.server 8088
```

---

## VRAM guardian (shared-display GPUs)

If your training GPU also drives your desktop, a job that spikes VRAM can crash the X
session. `--vram-guard 12000` (default for `casual`) runs a watcher that kills the job's
process group if total VRAM exceeds the threshold, leaving headroom for the display.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| N3V renders look 2× too wide / low PSNR | the dynerf **focal-units** issue — videosplat scales focals to the 2704px N3V reference in `convert.py`; ensure you're on a current build |
| Novel views explode into floaters; test PSNR *drops* during training | HyperNeRF disables opacity reset → floater collapse on sparse/textureless scenes. The casual path now defaults `opacity_reset_interval=3000`; override with `--opt-override opacity_reset_interval=...` |
| Sharp from training views, garbage elsewhere | novel-**view** overfit — sparse-camera captures only constrain geometry where cameras overlap. Add cameras / coverage |
| `--downsample` seems ignored on 4dgs | apply `patches/4dgaussians-dynerf-downsample.patch` |
| MASt3R OOM | keep ≤ ~18 calibration frames; use `--modular-above` |
| Hardware-synced rig looks doubled/blurry | `--no-audio-sync` (audio xcorr can inject spurious sub-frame offsets) |

---

## License

MIT — see [LICENSE](LICENSE). Built on 3D Gaussian Splatting (© Inria / GRAPHDECO,
research use) and the backbone repos (4DGaussians, SpacetimeGaussians, Gaussian-Flow,
4D-Rotor-Gaussians, MASt3R), each under its own license. Browser rendering by
[@mkkellogg/gaussian-splats-3d](https://github.com/mkkellogg/GaussianSplats3D).
