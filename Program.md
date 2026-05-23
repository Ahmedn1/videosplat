# Program — videosplat autonomous-research notes

This file plays the same role as `program.md` does for
[autoresearch](https://github.com/karpathy/autoresearch): it's the durable
context a coding agent reads to understand the project, what's been tried,
what the metric is, and how to run an experiment loop autonomously.

The autoresearch repo (`/media/ahmed/Data/Work/autoresearch`) and videosplat
are **two unrelated codebases** — autoresearch trains a tiny GPT in 5
minutes per experiment; videosplat trains 4D Gaussian Splatting models on
multi-camera video in 1–2 hours per experiment. We adapt the *pattern* (an
agent autonomously editing one file, running, keeping/discarding); we do
not share data, venvs, or training code.

---

## Project context

videosplat turns a folder of synchronised multi-camera videos into a
free-viewpoint 4D scene the user can fly through in a browser.

Pipeline (entry: `videosplat run <videos_dir>`):

1. **Step 1 — Sync** (`pipeline/sync.py`): extract frames from each `camNN.mp4`, audio-sync if multiple takes start at offset times.
2. **Step 2 — Calibrate** (`pipeline/calibrate.py`): COLMAP or MASt3R, both write COLMAP-format `sparse/0/` + LLFF `poses_bounds.npy`.
3. **Step 3 — Train** (`pipeline/train.py`): one of four 4D backends, each in its own external repo we shell out to:
   - `--algo 4dgs` → 4DGaussians at `~/4DGaussians/`
   - `--algo stg`  → SpacetimeGaussians at `~/SpacetimeGaussians/`
   - `--algo gaussian-flow` → Gaussian-Flow at `~/Gaussian-Flow/` (Pointrix or original variant)
   - `--algo 4d-rotor` → 4D-Rotor (nerfstudio format)
4. **Step 4 — Export** (`pipeline/export.py`): bake N keyframe PLYs from the trained model, write a static `viewer/` directory the user opens in a browser.

The codebase lives in `videosplat/` (pipeline modules) + `videosplat/cli.py`
(Typer CLI). Calibration outputs and intermediate artefacts go to
`outputs/<scene>/`. Backend training repos are at `~/4DGaussians/`, etc.

---

## What's already been done (chronological, with key findings)

### 1. Pointrix Gaussian-Flow integration (Apr 2026)

Wrapped the Pointrix-based Gaussian-Flow variant so users can pick
`--algo gaussian-flow` with either the original Linyou/D3DGS train.py or
the newer Pointrix `launch.py`. Required:

- New `prepare_gflow_pointrix_source()` in `convert.py` — frame-major
  filename layout, CustomDataset YAML, scale parameter (image downsample).
- `train_gflow()` in `train.py` auto-detects which variant the user has
  (`launch.py` present + `train.py` absent → Pointrix path).
- New `bake_gflow_pointrix_keyframes.py` — loads the Pointrix checkpoint
  and evaluates `pc.set_timestep(t)` to produce N keyframe PLYs.
- Patched the installed `pointrix` package's renderer to handle a
  diff-gaussian-rasterization API change (returns `(color, radii, depth)`
  not `(color, radii)`). Renderer override in `~/Gaussian-Flow/model/renderer.py`.
- Patched `get_loss_dict()` to accept `step=` kwarg from newer Pointrix
  trainer.
- Worked around a list-of-rec frame-data-ids bug in `controller/gs.py`:
  `generate_split_mask` got `[N, 3]` grads instead of `[N]`; added a
  pre-flatten step in `controller/gf.py:GFDensificationController`.

### 2. MASt3R calibration backend (Apr–May 2026)

Added MASt3R as a learned-matching alternative to COLMAP for low-texture
scenes where COLMAP fails (basketball dome, etc.):

- `_run_mast3r()` in `calibrate.py` — full inference + global alignment
  pipeline.
- Chunked inference (`_mast3r_chunked_inference()`) — strips
  `desc`/`desc_conf` per chunk to avoid runaway RAM use on large pair counts.
- Optimizer ladder: pair_count > 900 → start with `ModularPointCloudOptimizer`
  (skip the doomed `PointCloudOptimizer` attempt that always OOMs).
- Multi-frame static-camera mode with sparse scene graph: within-camera
  complete + cross-camera complete at one representative time, instead of
  the O(N²) full graph. For N=3 frames × 31 cams → 1,116 pairs (vs 8,556).
- **Stage 2 scene normalization**: rescale `im_poses` translations and
  `xyz_all` so max-camera-distance = 1.0. Critical for 4DGS densification
  heuristics to behave (they depend on `cameras_extent`).
- `--mast3r-image-size` CLI (default 512, drop to 384 to cut Modular VRAM ~44%).
- `--mast3r-niter` (default 500, was 300 hardcoded).
- `--mast3r-refine` (default **off** — see findings below).

### 3. The pycolmap refinement saga (4 attempts)

Stage 3 of the MASt3R work was a "refine MASt3R poses with COLMAP BA over
SIFT matches" pass. It took four rounds of debugging to make it run end-to-end:

1. **PINHOLE vs SIMPLE_RADIAL camera model mismatch** — fixed by forcing PINHOLE in `ImageReaderOptions`.
2. **Frame.DataIds() mismatch after binary roundtrip** — switched to in-memory `Reconstruction` build with `add_image_with_trivial_frame`.
3. **Same mismatch returned** — because `transcribe_image_ids_to_database` is buggy in pycolmap 4.0.4: it remaps `image_id` and `image.frame_id`, but the `frame.data_ids` still point to old image_ids.
4. **Build rec using DB-assigned image_ids from the start** — finally worked end-to-end.

Then discovered `triangulate_points` keeps poses *fixed* by design (only triangulates new points + refines intrinsics). Had to add an explicit `pycolmap.bundle_adjustment(rec)` pass after triangulation to actually refine extrinsics.

**Net finding on basketball dome:** even when refinement runs cleanly with BA enabled, test PSNR doesn't improve (and often gets worse). SIFT finds ~13 matches per camera pair on low-texture scenes, which is too few to outperform MASt3R's dense-matching prior. **Defaulted `--mast3r-refine` to off**; document that it's only useful on textured outdoor scenes where SIFT finds 100+ matches/pair.

### 4. D3DG-specific code removed (May 2026)

Stripped ~600 lines of Dynamic-3D-Gaussians dataset-format compatibility
(`run-d3dg`, `d3dg-to-videos` subcommands; `--d3dg-calib` flag;
`_import_precalibrated`; six converter functions; etc.). The pipeline is
now generic videos-only. Standard COLMAP/MASt3R paths fully preserved.

### 5. 4DGS quality on coffee_martini (May 2026)

First end-to-end run on N3V coffee_martini (18 GoPro cameras, real kitchen):

- **COLMAP took 24 hours** for the calibration step (1800 images × O(N²) exhaustive feature matching). That's expected for `--algo 4dgs` (one big COLMAP across all frames). The user's earlier `flame_steak` "fast" run was actually `--algo stg` (50 small per-frame COLMAPs, ~1h total).
- **First training run: catastrophic** — train PSNR 25, test PSNR 10, needle-Gaussian collapse. NaN loss twice, auto-restarted.
- **Diagnosed**: AABB span of 158 × 96 × 199 from outlier sparse-point triangulations (background through windows, ceiling, glare). Fixed by adding a camera-anchored inlier filter in `_ensure_init_pointcloud` (`train.py`): keep only points within K × camera_radius of camera centroid, K=3. After fix, AABB is 27 × 18 × 17. Filter writes log line `init cloud: N → M points (kept inliers within ... of camera centroid)`.
- **Second training run: still catastrophic** — same test PSNR ~11, despite sane AABB. Deeper diagnosis:
  - Compared our `poses_bounds.npy` to N3V's bundled one element-by-element.
  - Camera positions within 5–20% — fine.
  - **Rotations all off by ~92°, systematically.** Cause: `colmap_to_poses_bounds` was writing rotation columns as `[right, up, back]` (NeRF/OpenGL), but the LLFF `poses_bounds.npy` format spec is `[down, right, back]`. 4DGS' dynerf loader then applies `[col1, -col0, col2]` to convert LLFF → NeRF, so our wrong-convention poses got rotated an extra 90° on read. Train PSNR stayed high (memorizing each pose's own views) while test PSNR collapsed (cross-view geometry wrong).
  - Near/far bounds also way off: ours 0.05–0.30 / 42–48, bundled 5.79–9.00 / 65–344. Caused by spurious lens-flare/glare sparse points right at the camera lens, plus same outlier issue contaminating `_compute_bounds`.
- **Third training run (current, verifying):** after fixing column convention + applying the same K=3 outlier filter to bounds + using 5%/95% percentiles instead of 0.5%/99.5%. Per-camera rotation diff dropped from 92° → ~9° (the residual is real COLMAP-to-COLMAP gauge drift, expected). Bounds 1.26–4.18 / 13.76–15.15 — closer to bundled but still tight on far. Awaiting verification PSNR.

### 6. Driver / system fixes (in-flight)

- **NVIDIA driver mismatch** (kernel 580.142, userspace 580.159.03 after unattended-upgrades): killed CUDA init with error 804. Resolved by rebooting after 24h COLMAP finished. Future-proofed with `apt-mark hold` on nvidia packages.
- **VLC snap broken** → Flatpak install. GNOME Shell didn't pick up the Flatpak `.desktop` files (XDG_DATA_DIRS issue); symlinked into `~/.local/share/applications/` as a workaround until next login picks up `/etc/profile.d/flatpak.sh`.
- **Totem couldn't initialize OpenGL** for N3V mp4s. Use VLC or mpv.

### 7. Datasets staged for testing

- **N3V** (Meta Plenoptic Video): all 6 scenes downloaded from the official GitHub release. 5 in `data/n3v/` ready to use, 1 (flame_salmon_1) reassembled from 4 split parts. ~50 GB total extracted, ~12 GB zipped backup in `data/n3v_zips/`. Each scene is `camNN.mp4 + poses_bounds.npy` — drops straight into `videosplat run`.
- **AIST DanceDB**: one 9-camera breakdance sequence at `data/aist/breakdance_ch01/videos/`. ~316 MB. `data/aist/urls.csv` has 13,940 raw video URLs — any same prefix with varying `c0X` gives a new 9-camera multi-view set.
- **SoccerNet-v3**: skipped (NDA registration required).
- **DiVA-360**: skipped (1.4 TB Dropbox folder; no scripted per-scene URLs).

---

## The metric

The single number we optimise for, by default:

**Test PSNR at iter 14000 from a `videosplat run --algo 4dgs` end-to-end pipeline.**

Why this metric:

- 4DGS' upstream `train.py` evaluates test PSNR at iters 3000, 7000, and 14000 by default — iter-14000 is the latest free evaluation, post most densification rounds.
- It's the standard metric the 4DGS paper benchmarks on N3V (their published numbers: ~30–32 PSNR on these scenes).
- It correlates strongly with `scale_ratio_p99` from the bake step — when training is healthy, p99 is well under the 300:1 cap; when it's failing, p99 = 300 (needle collapse).
- An iter-30000 test eval is available by passing `--extra-args "--test_iterations 3000 7000 14000 20000 30000"` if you want it.

Secondary signals (look for these in the calibration / training log):

- `init cloud: N → M points (kept inliers within ... of camera centroid)` — filter ran.
- `Deformation Net Set aabb [x, y, z]` — span should be < 30 units. > 100 means outliers got through.
- `Scene normalised: radius X → 1.000 (scale Y)` — MASt3R path only.
- `loss is nan, end training, reexecv program now.` — training diverged. Bad sign even if it auto-restarts.

---

## How to run autoresearch in this repo

Three different invocations depending on what you want.

### A. Run autoresearch as-is (tiny LLM training, unrelated to videosplat)

This is what the autoresearch README documents. It doesn't touch videosplat.

```bash
cd /media/ahmed/Data/Work/autoresearch
uv sync                       # if you haven't already
uv run prepare.py             # ~2 min one-time data + tokenizer prep
uv run train.py               # ~5 min baseline experiment
```

Then point an agent (Claude Code, codex, etc.) at that directory with permission to edit `train.py`, and the agent will autonomously iterate per the loop in `/media/ahmed/Data/Work/autoresearch/program.md`.

### B. Use the autoresearch *pattern* against videosplat (this Program.md)

This file is your videosplat-flavoured equivalent of `program.md`. To run the loop against this repo, start a fresh agent session in `/media/ahmed/Data/Work/sports_replay/` and tell it:

```
Read Program.md. Pick up the experiment loop from the "Experiment loop"
section. Default scene: coffee_martini. Default algo: 4dgs. Iterate
autonomously; do not pause to ask for confirmation between experiments.
```

The agent then edits **only the in-scope files** (see below), runs `videosplat run` with the verification command, parses the resulting log for the metric, logs to `results.tsv`, and repeats.

### C. Bridge mode (autoresearch's machinery, videosplat's data)

Not currently supported. autoresearch's `prepare.py` and `train.py` are LLM-specific (tokenizers, BPE, GPT transformer). Adapting them to drive videosplat experiments would mean rewriting both. Not worth the effort vs option B.

---

## Experiment loop (videosplat-flavoured)

Adapted from the autoresearch loop. Key adjustments: each experiment takes ~1–2h (not 5 min); we have a 4-stage pipeline (not just `train.py`); the metric is PSNR (not val_bpb).

### Files the agent CAN edit

- `videosplat/pipeline/*.py` — sync, calibrate, convert, train, export modules.
- `videosplat/cli.py` — CLI signature and defaults.
- New helper modules added to `videosplat/`.

### Files the agent must NOT edit

- `~/4DGaussians/`, `~/SpacetimeGaussians/`, `~/Gaussian-Flow/` — these are external upstream repos. Patches there must be explicit, justified, and noted in `results.tsv` so we know which improvements came from videosplat vs from monkeypatching upstream.
- `data/` — datasets are immutable inputs.
- `outputs/` — kept as experiment results, not edited in place.

### The standard experiment

Default scene: **coffee_martini** (calibration is cached; iteration is GPU-bound only).

```bash
# Wipe the model + init cloud; keep COLMAP sparse/0/ + poses_bounds.npy
rm -rf outputs/coffee_martini_4dgs/model
rm    outputs/coffee_martini_4dgs/points3D_downsample2.ply 2>/dev/null

# If you edited colmap_to_poses_bounds itself, also regenerate the file:
.venv/bin/python -c "
import sys; sys.path.insert(0, '.')
from videosplat.pipeline.convert import colmap_to_poses_bounds
from pathlib import Path
colmap_to_poses_bounds(Path('outputs/coffee_martini_4dgs/sparse/0'),
                        Path('outputs/coffee_martini_4dgs/poses_bounds.npy'))"

# Run training only (re-uses existing calibration)
videosplat run data/n3v/coffee_martini/ \
    --output outputs/coffee_martini_4dgs \
    --algo 4dgs --calib-method colmap \
    --iterations 30000 --max-cameras 16 --no-view \
    --skip-sync --skip-calibrate \
    --extra-args "--test_iterations 3000 7000 14000 20000 30000" \
    > run.log 2>&1
```

Wall time: ~1–1.5 h on a single laptop GPU.

### Parsing results

```bash
# Test PSNR per evaluation iter
grep "Evaluating test" run.log

# scale_ratio_p99 from the bake step
grep "scale_ratio_p99" run.log

# AABB sanity (must be < 30 units in every dimension)
grep "Deformation Net Set aabb" run.log

# Any NaN restarts?
grep "loss is nan" run.log

# Init cloud filter outcome
grep "init cloud:" run.log
```

The primary metric: test PSNR at iter 14000 (line `[ITER 14000] Evaluating test: L1 X PSNR Y`).

### Logging — `results.tsv`

Tab-separated (not CSV — commas appear in descriptions). 7 columns:

```
commit	scene	algo	psnr_3k	psnr_7k	psnr_14k	psnr_30k	scale_p99	status	description
```

- `commit` — short git SHA (7 chars)
- `scene` — e.g. coffee_martini, flame_salmon_1, breakdance_ch01
- `algo` — 4dgs, stg, gaussian-flow, 4d-rotor
- `psnr_3k/7k/14k/30k` — test PSNR at each evaluated iter; use `-` if not evaluated
- `scale_p99` — bake-step `scale_ratio_p99` (300 = needles capped; low is healthy)
- `status` — keep / discard / crash
- `description` — short text describing what the experiment tried

Don't commit `results.tsv` itself — keep it untracked. The git commit history records the code changes; the TSV records the outcomes.

### The loop

LOOP FOREVER:

1. Look at git state: which branch, which commit.
2. Tune one thing in an in-scope file with an experimental idea. Pick from this rough priority list when out of ideas:
   - Init point cloud filter parameter (K=3 → 2 or 5)
   - Bounds percentile (5%/95% → 1%/99% or 10%/90%)
   - Densification thresholds (4DGS hyperparams via `--extra-args`)
   - Camera subset sampling strategy (farthest-point vs alternative)
   - Switching algos for the same scene to compare
3. `git commit -am "experiment: short description"`
4. Run the experiment (above). Redirect to `run.log`, do NOT tee — keep the agent context small.
5. Parse the log; record the row in `results.tsv`.
6. If iter-14k test PSNR improved (or scale_p99 dropped substantially with PSNR flat), keep the commit.
7. If equal or worse, `git reset --hard HEAD~1`.

### Constraints / sanity rules

- **Don't kill a running experiment with another `videosplat run`** to the same `--output`. Either wait or use a distinct output dir.
- **GPU sharing**: only one CUDA experiment at a time. Calibration on CPU is fine in parallel.
- **Out-of-scope changes**: if you find yourself wanting to edit external 4DGS/STG/GFlow code, stop and ask the human first — those are upstream repos and changes there are durable across all videosplat experiments.
- **Crashes**: NaN loss is a known 4DGS failure mode; the auto-restart usually recovers in one retry. If it restarts more than twice in a single run, treat as crash and discard.
- **Time bound**: kill any run that exceeds 3 h wall clock — likely something OOMed or hung. Record as crash.

### Open questions / current state (as of the latest commit)

- Column-convention fix in `colmap_to_poses_bounds` has just landed; verification run on coffee_martini is in progress / awaiting outcome. Expected: test PSNR @ 14k should jump from 11 → ~25 if the fix is sufficient. If it lands at 11 again, the next thing to investigate is per-camera intrinsic deviation vs N3V's bundled intrinsics, OR a test-camera selection mismatch.
- `--mast3r-refine` defaults to off (documented as opt-in for textured scenes).
- `--algo stg` is the right call for "fast iteration" since its calibration is ~1h vs 24h for 4DGS+COLMAP. Consider it as a quicker proxy when iterating on non-calibration knobs.

---

## TL;DR for someone joining cold

1. Read this whole file.
2. Read `videosplat/pipeline/calibrate.py` and `videosplat/pipeline/train.py` — those are the two files where 90% of the recent work has gone.
3. Check `outputs/coffee_martini_4dgs/` — that's the canonical end-to-end test scene right now. `sparse/0/` is the cached 24h COLMAP output; do not delete.
4. Default debugging scene: coffee_martini. Default algo: 4dgs. Default knob to start tuning: the init-cloud filter K in `train.py:_ensure_init_pointcloud`.
5. The whole stack works. The remaining puzzle is whether videosplat-driven 4DGS on N3V can hit canonical PSNR ~30; we're currently at ~11–18 depending on the fix iteration.
