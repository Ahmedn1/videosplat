# Findings — five datasets, what we learned

videosplat was developed by running it end-to-end on five very different multi-camera
captures. This is the honest write-up: the best result on each, and the lessons that
generalise. Metrics are PSNR; note carefully whether a number is **held-out-time**
(novel moment, seen viewpoint) or **held-out-view** (novel viewpoint) — they measure
very different things and conflating them was itself one of the lessons.

| Dataset | Capture | Method | Best PSNR | Headline finding |
|---|---|---|---|---|
| Basketball | 12-cam synced benchmark | STG / 4DGS (pre-calib) | — (validation) | Phase-0 proof the pipeline works end-to-end |
| Coffee Martini | 18-cam synced (N3V) | COLMAP → 4DGaussians | **28.08** (held-out-view) | Focal-units bug (+3); camera **coverage** > calibration accuracy |
| Flame Steak | N3V kitchen | COLMAP → SpacetimeGaussians | published-level | STG = the fast-iteration path (~1h calib vs 24h) |
| Piano | 3-cam casual, moving, unsynced | MASt3R → nerfies 4DGS | **28.0** (held-out-time) | Temporal density is the dominant lever; novel-**view** overfits to camera locations |
| AIST Breakdance | 9-cam synced 360° ring | MASt3R → nerfies 4DGS | 15.13 → **16.50** (held-out-view) | Novel-view collapse was an *optimization* bug (disabled opacity reset), not geometry |

---

## 1. Basketball — Dynamic3DGaussians benchmark (Phase 0)

**Capture.** The Dynamic3DGaussians multi-camera benchmark (a synchronized rig around
a basketball/juggling performer), with calibration shipped in the dataset.

**Why.** Validation before any custom capture: prove sync → calibrate → convert →
train → bake → viewer runs end-to-end on a known-good multi-view dynamic scene. Used
to bring up the STG and 4DGS paths and the browser viewer.

**Finding.** The pipeline works end-to-end; basketball became the regression scene.
A separate "basketball dome" capture later produced a useful *negative* result on
calibration refinement (below).

---

## 2. Coffee Martini — N3V, the quality bring-up (best held-out-view 28.08)

**Capture.** Neural-3D-Video plenoptic kitchen, 18 synchronized GoPros, hard lighting
(glare, windows). COLMAP calibration → 4DGaussians dynerf path. Held-out view = cam0.

**The climb (test PSNR 13 → 28.08 @14k):**

| Step | Change | Result |
|---|---|---|
| baseline | cached 24h COLMAP on all 1800 frames | rot err 4.6°, PSNR ~13 |
| calib | 5 frames/cam static-rig COLMAP | rot err **0.14°** (32× better), **400× faster** |
| — | (PSNR of the better calibration) | still ~13 — **flat** |
| focal fix | scale focals to N3V's 2704px reference | 13 → **16** |
| coverage | all 17 train cams + downsample 2 + denser 35k init | 16 → **28.08** |

**Findings.**
1. **Focal-units bug (+3 PSNR).** The 4DGaussians dynerf reader hardcodes N3V's native
   2704px width and assumes `poses_bounds` focals are in those pixels. We stored focals
   at our 1280px extraction resolution → every view rendered ~2.1× too wide. Fixed in
   `convert.py` (`_DYNERF_REF_WIDTH`).
2. **Calibration accuracy was a red herring for PSNR.** Making COLMAP 32× more accurate
   (and 400× faster) did *not* move PSNR. The fast calibration-error proxy does not
   predict reconstruction quality — we invalidated it mid-study and switched the metric
   to test PSNR. (The faster calibration is still a real infra win.)
3. **Coverage/resolution was the real ceiling.** Training on all cameras + `--downsample 2`
   (4× more pixels than downsample-4) + a denser init cloud lifted PSNR by **+12**.
   Novel-view quality is gated by how well the training cameras *surround* the held-out one.
4. A production 50k-iter run reached ~28.4 with a healthy ~1-point train/test gap;
   gains past 14k are marginal (densification plateaus ~15k).

---

## 3. Flame Steak — N3V via SpacetimeGaussians (the fast path)

**Capture.** Another N3V kitchen scene. Reconstructed with the **SpacetimeGaussians**
backbone (`--algo stg`).

**Finding.** STG calibrates with ~50 small **per-frame** COLMAPs (~1h total) versus the
single 24h exhaustive COLMAP that the 4dgs path needs on N3V. Since calibration accuracy
turned out *not* to be the PSNR bottleneck (see Coffee Martini), STG is the right tool
for **fast iteration** on non-calibration knobs — quick proxy reconstructions while
tuning everything else.

---

## 4. Piano — casual 3-camera capture (best held-out-time 28.0)

**Capture.** Three phones, deliberately adversarial: cam0 side (1280×720), **cam1
top-down** (1080×1920, over the keys), cam2 other side (1024×576) and *moving* in the
second half. Different start times and lengths. No ground truth. This is what
`videosplat casual` was built for.

**The climb (held-out-time PSNR):**

| # | Change | PSNR |
|---|---|---|
| 0 | COLMAP/SIFT calibration | fails — cam1 registers 0/8 |
| 1 | MASt3R calibration | all 3 cams linked |
| 4 | nerfies train, 20 timesteps | 16.3 (deformation *hurts*) |
| 5 | 150 timesteps | 24.6 |
| 8 | 300 timesteps | 27.7 (efficient sweet spot) |
| 9 | 600 timesteps | **28.0** (plateau) |

**Findings.**
1. **SIFT can't calibrate this; MASt3R can.** The top-down camera won't register with
   the side views under SIFT (wide baseline + repetitive keys). MASt3R's dense matching
   links all three.
2. **Only the nerfies/HyperNeRF reader fits.** 4DGaussians' dynerf/MultipleView modes
   assume one shared intrinsic + static poses — wrong for three heterogeneous cameras.
   nerfies gives per-image poses + per-camera intrinsics + per-frame time.
3. **Temporal density is THE lever.** 20→150→300→600 timesteps drove 16.3→24.6→27.7→28.0.
   With too few timesteps the deformation field *hurts* (it can't interpolate motion);
   density is what unlocks it. Sweet spot ≈ 300.
4. **Iterations and resolution don't help.** 30k iters regressed vs 14k; 1600×900
   regressed vs 1024×576. The capture is **sparse-view-limited, not compute-limited.**
5. **Person downweighting > masking** (frontier, moving-cam variant). For the pose solve:
   black-out 18.98 < unmasked 23.31 < **conf-downweight 23.65**. Blacking out the person
   removes the keyboard at the hands and starves the thin cam2 overlap; downweighting
   keeps the full image and only stops the moving person from voting on pose.
6. **Novel-time ≠ novel-view (the overfit we almost missed).** Held-out-*time* PSNR was
   28.0 — but we'd never tested held-out *views*. Rendering the live model from cam0's
   exact pose is sharp; from a novel angle 2.5× the scene radius away it explodes into
   floaters. A 3-camera capture only constrains geometry where cameras overlap; it
   overfits to the **training camera locations**. The usable free-viewpoint product is a
   gentle micro-orbit that stays inside the captured viewing cone — not a full fly-around.

---

## 5. AIST Breakdance — 9-camera synced ring (held-out-view 15.13 → 16.50)

**Capture.** A breakdancer in a white cyclorama, 9 hardware-synced 1080p cameras in a
360° ring. The first dataset with enough surround coverage to do a proper **held-out-VIEW**
eval: hold out 2 of 9 cameras, train on 7, measure PSNR on the novel views.

**The climb (held-out-VIEW PSNR @14k):**

| # | Change | PSNR |
|---|---|---|
| 2 | first run (HyperNeRF defaults) | 15.13 — *novel-view collapse* |
| 3 | + opacity_reset=3000 + dssim=0.2 | 16.17 |
| 4 | opacity_reset only (drop dssim) | 16.13 |
| 5 | + reduce densification (grad 4e-4) | **16.43** |
| 6 | densification grad 6e-4 | 16.50 (plateau) |

**Findings.**
1. **MASt3R calibrates a textureless white studio fine.** Feared blocker, non-issue:
   hardware sync means the dancer is the *same* object in all 9 views — a strong shared
   feature. 9/9 cameras registered into a clean ring (radius CV 11%, planarity 4%).
2. **The novel-view collapse was an *optimization* failure, not scene geometry.**
   HyperNeRF's config sets `opacity_reset_interval=300000` (= never, in a 14k run) and
   `lambda_dssim=0` — disabling the floater suppression standard 3DGS relies on. Fine on
   dense object-centric capture; **catastrophic on a sparse textureless ring**: the model
   fills empty space with floaters that satisfy the 7 training cameras and explode from
   the held-out angles. Test PSNR *degraded across training* (15.9→15.1) — the tell.
   Restoring `opacity_reset_interval=3000` fixed it (+1.0) and removed the degradation.
   **This is now the default for the casual/nerfies path.**
3. **dssim was pure cost** (+0.04 = noise, ~3× slower). Dropped.
4. **Ceiling ~16.5 is the textureless-background under-constraint.** No features on white
   walls → no geometric anchor → floaters persist in unobserved volume. The *dancer*
   reconstructs and is clearly resolvable from novel views; the empty studio caps the
   metric. Beating it would need depth/geometry priors or background masking.
5. **`--no-audio-sync` for synced rigs.** The audio cross-correlation produced spurious
   ±0.17 s offsets that would desync fast motion across cameras; disabling it is correct
   for hardware-synced captures.

---

## Cross-cutting lessons

- **Look at the render, not just the number.** Rendering the held-out view and *looking*
  repeatedly converted a mysterious flat PSNR into a diagnosable problem (wrong FOV →
  blur → floaters → coverage). PSNR alone misled us more than once.
- **Held-out-time vs held-out-view.** A great novel-time score says nothing about novel
  views. Test the thing you actually want (free-viewpoint = held-out-view).
- **Coverage beats calibration polish.** Within reason, *which* and *how many* cameras
  see the held-out region matters far more than shaving degrees off pose error.
- **Sparse-view captures are not compute-limited.** More iterations and higher resolution
  regressed on sparse captures — they memorise noise. Spend the budget on coverage and
  temporal density instead.
- **Restore standard 3DGS regularization.** Backbone configs tuned for dense
  object-centric data (HyperNeRF) silently disable opacity reset; re-enable it for
  sparse/textureless scenes or novel views collapse into floaters.
- **Downweight, don't delete.** Moving subjects should be confidence-downweighted in the
  pose solve, never blacked out — blacking out removes the static structure beside them.
- **Audio sync helps casual captures and hurts synced rigs.** Use it for unsynced phones;
  disable it for hardware-synced rigs.
- **Mind the focal units.** The N3V/dynerf reader expects focals in the 2704px reference;
  extracting at a different resolution without rescaling silently doubles your FOV.
- **VRAM is a hard constraint on a shared GPU.** MASt3R complete-graph OOMs above ~22
  images on 16 GB; heavy training spikes can crash the desktop session. Keep calibration
  sets small and run under the VRAM guardian.
