# ImageNet collection for Paper 1 — fix the design + collect the missing baselines

Hand this to Claude Code on the machine that has the ImageNet data. Paper 1 is now
**"OOD detectors are unsupervised accuracy estimators"**: show OOD-detection-specific scores
match, and sometimes beat, UAE-specific methods (ATC, NuclearNorm, COT, AC, DoC).

## What's wrong with the current ImageNet data, and what "correct" means

The current collection produced **three separate datasets** `ImageNetA`, `ImageNetR`, `ImageNetV2`,
and applied the **full synthetic-shift sweep to each** (`ImageNetA_noise_*`, `ImageNetR_noise_*`, …).
That is redundant and not the intended design.

**Correct design (one base, variants as organic OOD):**
- **InD = base ImageNet validation** (1000-class). Apply the synthetic sweep **once**, to InD.
- **Organic OOD = ImageNet-R, ImageNet-A, ImageNet-V2** (no synthetic augmentation of them —
  they are natural shifts, used only as OOD evaluation folds).
- Result: a **single dataset `ImageNet`** whose `get_ood_dict()` returns all three variants.

So the synthetic sweep is computed once (on base ImageNet), and R/A/V2 are three organic folds.

## Two things to collect (both need the pretrained model; NO training)

1. **8 post-hoc detector features** for the single `ImageNet` dataset (replaces the redundant
   `ImageNetA/R/V2_*` feature files): `msp, energy, cross_entropy, grad_magnitude, mahalanobis,
   knn, vim, react` (NO Typicality/Glow).
2. **Softmax-matrix stats for NuclearNorm + COT** (`figures/softmax_stats.csv`) — the strong
   modern UAE baselines. These need the full [N,C] softmax, which the feature cache doesn't store,
   so they require re-inference via `collect_softmax_stats.py`.

Ours/ATC/AC/DoC come from (1); NuclearNorm/COT from (2). The comparison then mirrors the other
datasets' Table-1.

## Steps

### 0. Data layout (adjust paths to your machine)
```
export IMAGENET_VAL=/path/to/ILSVRC2012/val              # 1000 wnid subdirs
# and under DATA_ROOT/imagenet_shift:  imagenet-r/  imagenet-a/  imagenetv2-matched-frequency-format-val/
```
(see `scripts/download_uae_datasets.sh imagenet` if any are missing.)

### 1. Fix the testbed to use ONE base + three organic OOD folds
Edit `testbeds/imagenet_shift.py`:
- Class name `ImageNetTestBed`, dataset name **"ImageNet"** (single).
- Keep InD = base ImageNet val (via `build_imagenet_shift_dataset` with `variant=None`/base),
  pretrained ResNet-101 with the **restored 1000-class head** (already implemented).
- Rewrite `get_ood_dict()` to load and return **all three** variants at once:
```python
def get_ood_dict(self):
    d = {}
    for v in ["imagenet-r", "imagenet-a", "imagenet-v2"]:
        _, _, _, ov, ot, subset = build_imagenet_shift_dataset(IMAGENET_SHIFT_ROOT, IMAGENET_VAL, self.trans, variant=v)
        d[f"{v} Val"] = self.dl(ov); d[f"{v} Test"] = self.dl(ot)
        self._subset_idx[v] = subset          # 200-class idx list for R/A (None for V2)
    return d
```
- **R/A 200-class masking (correctness-critical):** ImageNet-R and ImageNet-A cover 200 classes.
  When computing accuracy on those folds, mask the 1000-way logits to the fold's `subset_idx`
  before argmax, else accuracy is understated. `build_imagenet_shift_dataset` already returns the
  subset index list; wire it into `BaseTestBed.compute_losses` (mask logits for R/A folds).
  ImageNet-V2 is full-1000 and needs no masking. (If masking is hard to thread cleanly, at minimum
  report R/A accuracy on the 200-class subset and document it.)

### 2. Register
- `testbeds/__init__.py`: add `from testbeds.imagenet_shift import *`
- `eval_detectors.py`: add `"ImageNet": ImageNetTestBed` to `TESTBEDS`.

### 3. Collect the 8 detector features (single ImageNet)
```bash
python -c "from eval_detectors import *; collect_all(detectors=DETECTORS_ALL_NO_GLOW, datasets=['ImageNet'], models=['resnet'], overwrite=True)"
```
Delete the redundant old files afterward: `rm data/resnet/feature_data/ImageNet{A,R,V2}_*`.

### 4. Collect NuclearNorm + COT softmax stats
In `collect_softmax_stats.py`, add `"ImageNet"` to `CLASSIF` (and keep `MODELS=["resnet"]` since
only ResNet is set up), then:
```bash
python collect_softmax_stats.py
```
This re-runs the pretrained model over ImageNet's folds and appends ImageNet rows (nuclear_norm, cot,
true_acc per fold) to `figures/softmax_stats.csv`.

### 5. Copy back to the main machine
- `data/resnet/feature_data/ImageNet_*` (the 80 new feature CSVs)
- `figures/softmax_stats.csv` (now with ImageNet rows)

Then on the main machine the full comparison (Ours vs ATC/AC/DoC/NuclearNorm/COT) drops straight in.

## Gotchas
- **No training** — ImageNet uses the torchvision-pretrained ResNet-101 with its original 1000-class
  head restored (`testbeds/imagenet_shift.py` already does this). Do NOT fine-tune.
- **R/A logit masking** to the 200-class subset for accuracy — the single biggest correctness item.
- **Reference cap** (50k) applies to knn/maha/vim on the ImageNet-train reference (already handled by
  `FeatureSD.REFERENCE_CAP`); if you don't have ImageNet-train, use the InD-val fold as the reference.
- **No Typicality/Glow** anywhere (dropped from the paper).
- ViT is not wired for ImageNet (the testbed asserts resnet); resnet-only is fine for this data point.

## Verification
- `data/resnet/feature_data/ImageNet_normal_msp.csv` folds include `ind_val`, `ind_test`, and
  `imagenet-r Val/Test`, `imagenet-a Val/Test`, `imagenet-v2 Val/Test` (all three variants, one file).
- Synthetic files `ImageNet_noise_*` etc. exist once (not per-variant).
- `figures/softmax_stats.csv` has `Dataset==ImageNet` rows with non-NaN `nuclear_norm` and `cot`.
- Sanity: base-ImageNet InD top-1 ≈ 0.78 (full 1000-class); ImageNet-V2 lower; R/A higher on their
  200-class subset. (The current mis-collected data shows InD acc 0.94 for R/A because InD was
  already restricted to the 200-class subset — under the correct design InD is full 1000-class ≈0.78.)
```
