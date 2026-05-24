# External backend patches

VideoSplat drives the upstream [4DGaussians](https://github.com/hustvl/4DGaussians)
repo as a subprocess. One small local modification to that repo is **required**
for the pipeline to behave correctly; it is not part of upstream, so it is
captured here and must be re-applied to any fresh clone.

## `4dgaussians-dynerf-downsample.patch`

**Target:** `scene/dataset_readers.py` (function `readdynerfInfo`), against
4DGaussians commit `843d5ac`.

**Why it's needed:** upstream hardcodes the dynerf dataset's spatial downsample
factor to `1.0`, so the resolution at which a scene trains is fixed in code and
cannot be controlled from outside. VideoSplat sets a `DYNERF_DOWNSAMPLE`
environment variable (in `videosplat/pipeline/train.py::_build_env`, from the
`--downsample` CLI flag); the patch makes the reader honor it.

**Load-bearing:** without this patch, `videosplat run --downsample N` is
**silently ignored** and every scene trains at full native resolution. The
published-quality result (coffee_martini test PSNR ~28) used `--downsample 2`,
which only takes effect with this patch applied.

**What it changes (3 lines of logic):**
1. Reads `_downsample = float(os.environ.get("DYNERF_DOWNSAMPLE", "4.0"))`
   (default 4.0 — matches VideoSplat's default).
2. Passes `_downsample` instead of the hardcoded `1.0` to the **train**
   `Neural3D_NDC_Dataset`.
3. Same for the **test** `Neural3D_NDC_Dataset` — so train and held-out eval
   render at the same resolution (mismatched here would corrupt test PSNR).
   (Plus a one-character trailing-whitespace cleanup, cosmetic.)

## Apply

```bash
cd /path/to/4DGaussians
git apply /path/to/videosplat/patches/4dgaussians-dynerf-downsample.patch
# verify:
git diff --stat scene/dataset_readers.py   # 1 file changed, 5 insertions(+), 3 deletions(-)
```

To revert: `git apply --reverse 4dgaussians-dynerf-downsample.patch`.
