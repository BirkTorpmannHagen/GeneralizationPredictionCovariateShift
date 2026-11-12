# GPU machine: reruns needed after the hygiene / baseline fixes

Written 2026-11-09. Everything here needs the trained checkpoints, the datasets and a GPU.
Everything that could be fixed from cached features has already been done on the Mac.

## Recommended order (avoids collecting entropy twice)

1. Sync the code (section 0).
2. Run the softmax collection on its own: `python collect_softmax_stats.py`. It is independent of the feature recollection and unblocks COT and Nuclear Norm.
3. Run the full recollection in section 0b. It already includes the entropy detector, for the classification datasets and Polyp, and regenerates the Polyp smear folds with warped masks (0c).
4. Retrain NICO (section 1). Then recollect NICO only: `collect_all(detectors=DETECTORS_ALL_NO_GLOW, datasets=['NICO'], models=['resnet','vit'], overwrite=True)`, and rerun `python -c "import collect_softmax_stats as c; c.collect(datasets=['NICO'])"`.

Don't also run `collect_gpu_followups.py` after 0b: its entropy step repeats work 0b has already done. It's only the shortcut for when 0b is skipped.

## 0. Sync code first (both directions)

The GPU machine's `utils.py` was stale too. Copy this repo's versions of these files over before running anything:
- `utils.py`
- `features.py`
- `eval_detectors.py`
- `collect_softmax_stats.py` and `uae_stats.py` (the batch statistics it imports)
- `collect_gpu_followups.py`
- `datasets/nico.py`, `datasets/office31.py`, `datasets/officehome.py`
- `testbeds/base.py` (unshuffled evaluation loaders; see 0b)
- `datasets/synthetic_shifts.py` (smear now warps the segmentation mask; see 0c)
- `experiments/`

## 0b. Per-sample pairing bug: recollect all features (required for per-sample analyses)

**The bug.** `ooddetectors.FeatureSD.compute_features_and_loss_for_loaders` makes two passes over each data loader and zips the results by position:
- one pass computes loss, correctness, index and class (`testbed.compute_losses`);
- one pass computes the detector scores (`get_features`).

The loaders were built with `shuffle=True` (`testbeds/base.py`), so each pass saw a different order. In every cached feature CSV, a row's detector score therefore belongs to a different sample than its loss, correctness and class. The evidence: on the in-distribution validation fold, max-softmax and correctness correlate at about 0.00 for every dataset and model.

**What is and isn't affected.**
- *Unaffected:* fold-level averages, because both passes cover the same fold. That covers all the main results: detection rates, mean scores, fold accuracy, ATC/AC/DoC, MDE, the rankings, calibration, and shift-detection AUROC.
- *Invalid:*
  - PRE, which conditions accuracy on per-sample detector verdicts (`simulations.py`);
  - the failure-detection AUROC in `ablations.detector_quality_vs_coupling`;
  - the label-shift stress test (`experiments/label_shift.py`, which refuses to run on mispaired data);
  - any other per-sample score/correctness or score/class analysis.

**The fix (already in the code).** `testbeds/base.py` now uses `shuffle=False` for evaluation loaders, so both passes see the same order. `collect_softmax_stats.py` subsamples folds randomly rather than taking the first samples, since unshuffled loaders are ordered by class.

**What to run.** Recollect every detector for every dataset and model, overwriting the cache:
```bash
python -c "from eval_detectors import *; collect_all(detectors=DETECTORS_ALL_NO_GLOW, datasets=['CCT','OfficeHome','Office31','Camelyon17','IWildCam'], models=['resnet','vit'], batch_size=16, overwrite=True)"
python -c "from eval_detectors import *; from features import grad_magnitude, cross_entropy, entropy, energy, knn; collect_all(detectors=[grad_magnitude, cross_entropy, entropy, energy, knn], datasets=['Polyp'], models=['deeplabv3plus','unet','segformer'], batch_size=16, overwrite=True)"
```
NICO is recollected by `collect_gpu_followups.py --nico` after retraining (step 1). `DETECTORS_ALL_NO_GLOW` includes the new entropy detector.

**Then, on either machine:**
1. Regenerate the detection-rate caches.
2. Recompute PRE: `experiments/pra.py`.
3. Run the stress test, then rerun all analyses:

```bash
python -m experiments.label_shift
python -c "from experiments.label_shift import paper_assets; paper_assets()"
```

The fold-level results should change only marginally (`drop_last` now drops the same samples in both passes); rerun them anyway for consistency.

## 0c. Polyp smear: mask was not warped (excluded until recollected)

**The bug.** `smear` is a geometric `GridDistortion`. `TransformedDataset` warped the image but passed the segmentation mask through unchanged, so for Polyp the prediction is scored against a mask in the wrong place. The IoU (and "accuracy") collapses with intensity even though the image still looks in-distribution: every detector's DR stays near its InD level while accuracy falls from 0.88 to 0.09. Classification is unaffected (a warp does not change a class label).

**Current state.** `utils.INVALID_FOLDS = {"Polyp": ("smear",)}` drops these folds in `load_data`, the PRE results loader and `streaming_eval`, so every analysis and paper asset currently excludes Polyp smear. PRE's trees were still fitted with smear among the calibration families; recomputing PRE (0b, step 2) fixes that.

**The fix (already in the code).** `smear(img, intensity, mask=...)` warps image and mask with the same grid, and `TransformedDataset` uses it whenever the target is a 3-D mask tensor.

**What to run.** Nothing extra: the Polyp recollection in 0b regenerates the smear folds, provided `datasets/synthetic_shifts.py` is synced first. Afterwards, on either machine, set `INVALID_FOLDS = {}` in `utils.py`, delete `figures/perfold_tables.pkl`, and rerun the analyses.

## 0d. Do not analyse a partial sync (GradNorm changed)

The new collection computes GradNorm with the closed-form `features.grad_magnitude` (head-input hook) instead of the old per-sample autograd loop. The new files are on a different scale, and not by a constant factor: for ResNet, new/old means on the mildest synthetic fold are about 2x (CCT), 8x (OfficeHome), 34x (Office31), 0.2x (iWildCam) and 0.8x (Camelyon17). kNN, MSP, Energy, Uniform-CE and ReAct files agree in scale between collections.

Consequence: a data folder that mixes old and new files gives nonsense for GradNorm (a mean-score estimator reached MAE 3.6 on Office31/ResNet). Copy back **all** recollected files for a dataset before running any analysis, then delete `figures/perfold_tables.pkl` and rerun the full chain:
```bash
python -m experiments.ranking_analysis && python -m experiments.calibration_analysis && python -m experiments.paper_figures && python -m experiments.paper_artifacts
```
Also worth checking once on the GPU machine: that the new `grad_magnitude` matches the autograd loop (`_grad_magnitude_loop`) on a few batches, as its docstring claims, since the old numbers did not.

## 0e. GradNorm is all zeros for Polyp (fix needed before the next Polyp recollection)

`features.grad_magnitude` falls back to `_grad_magnitude_loop` for the segmentation models, which takes `log_softmax` over the output channels. Binary segmentation has a single channel, so the log-softmax is identically 0 and every score is 0 (all `data/*/feature_data/Polyp_*_grad_magnitude.csv` from 2026-11-11 are constant). GradNorm therefore drops out of every Polyp analysis. Fix: for single-channel outputs use the sigmoid analogue (binary cross-entropy of the per-pixel probability against a 0.5 target), then recollect GradNorm for Polyp only.

## 1. Retrain NICO (required)

**What was wrong:**
- `datasets/nico.py` shuffled the in-distribution ("dim") image list without a seed, and both `classifier/train.py` and every feature collection call it. So the train/val/test split was re-drawn on each build.
- The cached "ind_val"/"ind_test" folds therefore largely contain the classifier's own training images. This inflates NICO in-distribution accuracy and every NICO accuracy gap.
- The organic domain names were also scrambled: the per-domain image lists were shuffled after the names were fixed.

**What to do:**
- Retrain the NICO classifiers with the fixed builder (seeded, sorted):
  ```bash
  python -m classifier.train   # NICO, resnet + vit; same recipe as before
  ```
- While doing this, check why the current NICO models collapse to 5–16% accuracy on the other domains (chance is 1.7%) even though in-distribution accuracy is about 90%. A quick diagnostic is a histogram of predicted classes on one organic domain.
  - If the predictions pile onto a few classes, the cause is the model or preprocessing.
  - Preprocessing to check: NICO's testbed uses `ToTensor()` without ImageNet normalization. Make sure that matches training.

## 2. Collect features and statistics

```bash
python collect_gpu_followups.py --nico     # after step 1; omit --nico to skip NICO
```

This runs three steps:
1. **NICO:** recollects every post-hoc detector (8 + entropy).
2. **Entropy:** adds the predictive-entropy detector (`*_entropy.csv`) for all datasets, Polyp included.
   - The old "Entropy" (`cross_entropy`) was the loss against an all-ones target: −Σ_c log p_c, or a Jaccard loss for Polyp.
   - It stays in the detector set under its honest name, UniformCE.
   - ATC-NE and Reg-Entropy now read the real entropy.
3. **Softmax statistics:** runs `collect_softmax_stats.py`, which writes `figures/softmax_stats.csv` and `figures/softmax_stats_subsets.csv`. Its columns:
   - `nuclear_norm`: paper normalization √(min(N,C)·N). After the linear calibration the results are identical to the old one, since N > C everywhere.
   - `cot_err`: real COT (Lu et al. 2023). It's the exact optimal-transport cost with cost ‖p − e_y‖∞ against InD-val label marginals.
   - `label_shift_l1`: the old "cot" statistic, which was not COT.
   - Bootstrap batches of 100, used for the set-level shift-detection AUROC.

   It also saves **per-sample softmax matrices** (with labels and sample indices) to `data/softmax/<model>/<dataset>/<fold>.npz`: up to 2,000 random samples per fold, for every fold including `ind_val` and `ind_test`. The label-shift test (`experiments/label_shift.py`) re-batches these offline into class-biased batches (Dirichlet class mixes, 1/2/5-class batches, hardest or easiest classes) to evaluate Nuclear Norm, COT and the label-marginal statistic. **No separate collection of class-biased batches is needed, but these files must be copied back.** The detector-based methods and MDE are re-batched from the recollected feature CSVs in 0b.

## 3. Copy back

- `data/{resnet,vit}/feature_data/*_entropy.csv`
- `data/{deeplabv3plus,unet,segformer}/feature_data/Polyp_*_entropy.csv`
- `data/{resnet,vit}/feature_data/NICO_*.csv` (all, after retraining)
- `figures/softmax_stats.csv`, `figures/softmax_stats_subsets.csv`
- `data/softmax/` (per-sample softmax matrices, for the label-shift test of Nuclear Norm / COT)
- all of `data/{resnet,vit,deeplabv3plus,unet,segformer}/feature_data/` after the recollection in 0b

Then, on either machine:

```bash
python -c "from experiments.runtime_classification import ood_detector_correctness_prediction_accuracy as f; [f(1, model=m) for m in ['resnet','vit']]"
python -m experiments.ranking_analysis
python -m experiments.shift_detection
```

## Recommended, not required

- **Weak backbones.** In-distribution accuracy is 0.40 for OfficeHome ResNet and 0.38 for Office31 ViT. Reviewers will question whether accuracy estimation on near-chance models is meaningful. Retrain them with the standard recipe, or report results excluding them.
- **ImageNet-R/-A.** These need 200-class logit masking before argmax (see `UAE_DATA_COLLECTION.md`). Verify it before using the cached ImageNet features.
