# UAE benchmark data collection — handoff

Continue the "bridge" paper (OOD detection = unsupervised accuracy estimation) by adding
the datasets standard in the UAE / OOD-performance-prediction literature (ODP-Bench, ICCV'25,
arXiv:2510.27263; AutoEval; ATC) that we currently lack. This preempts the "standard
benchmarks missing" reviewer complaint and evaluates on the UAE community's home turf.

**This file is written for another machine (with the datasets + a working Python/GPU env).**
Give it to Claude Code there and work through the per-dataset checklists.

Current benchmarks in the paper: CCT, OfficeHome, Office31, NICO, Camelyon17, iWildCam
(classification) + Polyp (segmentation). Priority additions, in order:
1. **CIFAR-10 family** (CIFAR-10-C, CIFAR-10.1, CINIC-10) — cheapest, most expected.
2. **ImageNet shift** (ImageNet-R/-A/-V2/-C) — high impact, near-free (pretrained backbone, no training).
3. **WILDS FMoW + RxRx1** — completes WILDS; pipeline already exists.
4. **DomainNet** — optional breadth.

Detectors are the **eight post-hoc detectors, NO Typicality/Glow**:
`grad_magnitude, cross_entropy (entropy), energy, knn, msp, mahalanobis, react, vim`.

---

## Pipeline (how a dataset becomes a paper number)

```
datasets/<name>.py         builder -> (ind_train, ind_val, ind_test, ood_val, ood_test), each
                           yields (img, int_label, idx) and exposes .num_classes
testbeds/<name>.py         <Name>TestBed(BaseTestBed): loads builder + classifier checkpoint,
                           defines get_ood_dict() (organic OOD folds)
testbeds/__init__.py       add `from testbeds.<name> import *`
train (or pretrained)      classifier_logs/<model>/<ClassName>/checkpoints/best.ckpt
eval_detectors.py          add to TESTBEDS; run collect_all -> data/<model>/feature_data/
                           <Dataset>_<mode>_<detector>.csv   (mode in SYNTHETIC_SHIFTS + "normal")
experiments/...            regenerate Table 1 / interchangeability / quality-vs-uae
```

Synthetic-shift folds are produced automatically by `BaseTestBed` transforming `ind_test`
(the 9 modes in `utils.SYNTHETIC_SHIFTS`). Organic folds come from `get_ood_dict()`.
Distance/density detectors (knn/maha/vim) use `ind_train` as reference, capped at
`REFERENCE_CAP` inside `ooddetectors.FeatureSD`.

## Files already added this session (drafts, verify on-machine)

- `scripts/download_uae_datasets.sh` — downloads CIFAR / ImageNet-shift / WILDS / DomainNet.
- `datasets/cifar_datasets.py`, `testbeds/cifar10.py` — CIFAR-10 + CIFAR-10.1 + CIFAR-10-C. **Ready.**
- `datasets/imagenet_shift_datasets.py`, `testbeds/imagenet_shift.py` — ImageNet variants,
  pretrained backbone. **Ready for ResNet + ImageNet-V2; see TODOs for ViT and R/A masking.**

---

## Checklist per dataset

### 1. CIFAR-10 family  (train needed; cheap)
```bash
DATA_ROOT=../Datasets bash scripts/download_uae_datasets.sh cifar
```
- [ ] Register: add `from testbeds.cifar10 import *` to `testbeds/__init__.py`; add
      `"CIFAR10": CIFAR10TestBed` to `TESTBEDS` in `eval_detectors.py`.
- [ ] Train ResNet-101 + ViT on CIFAR-10 (mirror `train_wilds.py::train_one`, but with
      `build_cifar10_dataset`; checkpoint dir `classifier_logs/<model>/CIFAR10/checkpoints/best.ckpt`).
      Recipe must match the others: Adam lr=1e-3, `CosineAnnealingWarmRestarts(100,2)`, early-stop on val_acc.
- [ ] Collect: `python -c "from eval_detectors import *; collect_all(detectors=DETECTORS_ALL_NO_GLOW, datasets=['CIFAR10'], models=['resnet','vit'], overwrite=True)"`
- [ ] Sanity: `ls data/resnet/feature_data/CIFAR10_*` should show 9 synthetic modes + `normal`, x8 detectors.

Note: `testbeds/cifar10.py` sets `INCLUDE_CIFAR10C=True`, so CIFAR-10-C corruptions become
extra organic folds alongside CIFAR-10.1. Turn off if you want CIFAR-10.1 only.

### 2. ImageNet shift  (NO training; pretrained backbone)
```bash
DATA_ROOT=../Datasets bash scripts/download_uae_datasets.sh imagenet
export IMAGENET_VAL=/path/to/ILSVRC2012/val         # you must provide base val (1000 wnid dirs)
```
- [ ] Register: `from testbeds.imagenet_shift import *`; `"ImageNet": ImageNetShiftTestBed`.
- [ ] Pick the variant via env: `export IMAGENET_VARIANT=imagenet-v2` (start here — no class masking).
- [ ] Collect (resnet only until ViT wired): `python -c "from eval_detectors import *; collect_all(detectors=DETECTORS_ALL_NO_GLOW, datasets=['ImageNet'], models=['resnet'], overwrite=True)"`
- [ ] **Gotchas to fix on-machine:**
  - ViT: `ImageNetShiftTestBed` asserts resnet. Add a ViT branch (ViTClassifier + restore its
    pretrained head) mirroring the ResNet one.
  - ImageNet-R / -A cover 200 classes. `build_imagenet_shift_dataset` returns `subset_idx`;
    mask logits to it before argmax in `BaseTestBed.compute_losses` (else accuracy is wrong).
    ImageNet-V2 / -C are full-1000 and need no masking, so do V2 first.
  - ImageNet-C is ~75 GB; enable in the download script only if you want the corruption suite.

### 3. WILDS FMoW + RxRx1  (train needed; pipeline exists)
```bash
DATA_ROOT=../Datasets bash scripts/download_uae_datasets.sh wilds
```
Add builders to `datasets/wilds_datasets.py` (the `_build` machinery already handles the split
resolution — this is a two-liner each):
```python
class FMoW(_WILDSFold): pass
class RxRx1(_WILDSFold): pass
def build_fmow_dataset(root, tr, va, caps=None):  return _build(FMoW,  "fmow",  root, tr, va, caps)
def build_rxrx1_dataset(root, tr, va, caps=None): return _build(RxRx1, "rxrx1", root, tr, va, caps)
```
Then copy `testbeds/camelyon17.py` -> `testbeds/fmow.py` / `testbeds/rxrx1.py`, swap the builder
and the `classifier_logs/<model>/<FMoW|RxRx1>/` path, register, add to `train_wilds.py::BUILDERS`,
train, and collect. FMoW = time/region shift, RxRx1 = experimental-batch shift.

### 4. DomainNet  (optional)
```bash
DATA_ROOT=../Datasets bash scripts/download_uae_datasets.sh domainnet
```
6 domains (clipart, infograph, painting, quickdraw, real, sketch), 345 classes. Treat `real` as
in-distribution, the other 5 as organic OOD. Write an ImageFolder-style builder + testbed like CCT.

---

## Slotting results into the paper

Everything lands in `paper_drafting/`. After collection, regenerate:
- **Table 1** (`figures/table1_main.csv`): Ours(best OOD detector) vs Nuclear Norm, COT, ATC, AC, DoC.
- **Interchangeability heatmap** (`figures/table2_interchangeability.csv` -> `detector_interchangeability.pdf`).
- **OOD-quality vs UAE-coupling** (`figures/detector_quality_vs_coupling.csv` + `raw_vs_dr_signed_rho.csv`
  -> `quality_vs_uae.pdf`). More datasets = more points on this scatter, which is the paper's
  central bridge figure — so this is the highest-value artifact to extend.

The regeneration entry points live in `experiments/reviewer_experiments.py` and
`experiments/accuracy_prediction.py`; the figure scripts are in this session's history
(signed-rho, interchangeability, quality-vs-uae). Baselines Nuclear Norm / COT need the softmax
matrix (`collect_softmax_stats.py`).

## Gotchas summary
- **No Typicality/Glow** anywhere (dropped from the paper: weak, not post-hoc, no WILDS coverage).
- Reference cap (50k) applies to knn/maha/vim on large train sets (CIFAR ok, ImageNet/FMoW large).
- ImageNet uses a **pretrained head** (no fine-tuning); every other dataset trains from ImageNet init.
- Keep the training recipe identical across datasets (Adam 1e-3 + CosineAnnealingWarmRestarts(100,2)),
  so accuracy differences are the dataset, not the optimizer.
- Verify `.num_classes`, the (img, int_label, idx) item shape, and that `ind_test` is clean
  in-distribution (synthetic folds are derived from it).

## Verification checklist
- [ ] `python -c "from testbeds import *; b=CIFAR10TestBed(model='resnet', mode='normal'); print(b.num_classes, len(b.ind_test))"`
- [ ] One collection cell writes 8 CSVs with columns `[fold, feature_name, feature, loss, acc, idx, class]`.
- [ ] `ind_val` fold present in each CSV (needed for the InD threshold/quantile).
- [ ] New dataset appears in Table 1 with Ours competitive vs Nuclear Norm/COT and better than ATC.
- [ ] New (dataset, detector) points fall on the upward quality-vs-coupling trend.
