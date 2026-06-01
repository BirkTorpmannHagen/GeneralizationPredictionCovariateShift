# GeneralizationPredictionCovariateShift

Deep vision models often experience accuracy degradation under covariate shift. This raises challenges for reliability and trustworthiness, as deployment streams typically lack labels to measure this degradation. We study whether aggregate out-of-distribution (OOD) detection rates can serve as sequence-level indicators of network accuracy under shift. Across five benchmarks, multiple architectures, six post-hoc OOD detectors, synthetic perturbations, and naturally occurring domain shifts, we observe that detection rates frequently correlate with generalization gaps. We exploit this relationship by fitting a simple linear model that maps OOD detection rates to accuracy degradation using labeled synthetic shifts. At inference time, the estimator requires only unlabeled inputs. Under a leave-one-shift-out cross-validation protocol, it achieves mean absolute errors between 0.02 and 0.10 and outperforms competing methods for unlabeled accuracy estimation both in terms of absolute error and calibration across synthetic and naturally occurring shifts. These results indicate that OOD detectors can provide useful signals for monitoring model performance under covariate shift, with the caveat that their effectiveness depends on the reliability of the observed alignment between detectability and accuracy degradation.

---

## Repository layout

```
classifier/       ResNet / ViT classifiers + training entry point
segmentor/        Segmentation models (DeepLabV3+, U-Net, SegFormer) for the Polyp testbed
glow/             Glow normalizing-flow likelihood model used by the `typicality` OOD detector
datasets/         Dataset builders + synthetic-shift transforms
testbeds/         Per-dataset wrappers tying (dataset, classifier, glow) together
features.py       The six post-hoc OOD detector score functions
ooddetectors.py   FeatureSD: sweeps a testbed and writes per-fold detector scores
eval_detectors.py Entry point that drives `ooddetectors.FeatureSD` across all cells
components.py     OODDetector, calibrators, ensembles, helper tree structures
rateestimators.py ErrorAdjustmentEstimator / SimpleEstimator (the "PRE" baseline)
riskmodel.py      Event-tree risk decomposition used by the simulator
simulations.py    UniformBatchSimulator (sequence-level evaluation harness)
experiments/      All paper figures + tables
  pra.py                       Leave-one-shift-type-out PRE-calibration sweep (cached output)
  runtime_classification.py    Per-(dataset, model, detector) DR vs. accuracy table builder
  accuracy_prediction.py       All plotting / statistical-test / LOSO-MAE code
experiments.py    Top-level: runs the full set of paper experiments
data/             Cached intermediate results (detector features, DR tables, PRA results)
figures/          Output directory for the paper's figures and tables
```

## Installation

A working PyTorch + PyTorch-Lightning environment is required. The main third-party dependencies are:

```
torch, torchvision, pytorch-lightning
numpy, pandas, scipy, scikit-learn,
matplotlib, seaborn, tqdm
```

For the segmentation testbed: `segmentation-models-pytorch` and `transformers` (SegFormer).

## Datasets

Five benchmarks are used. By default the testbeds in `testbeds/` look for them under `../../Datasets/<name>`:

* **OfficeHome** → `../../Datasets/OfficeHome`
* **Office31** → `../../Datasets/office31`
* **NICO++** → `../../Datasets/NICO++`
* **CCT (Caltech Camera Traps)** → `../../Datasets/CCT`
* **Polyps (Kvasir-SEG + organic shift sets)** → `../../Datasets/Polyps`

Edit the paths at the top of each file in `testbeds/` (and in `classifier/train.py` / `segmentor/train.py` / `glow/trainpl.py`) if your layout differs.

## Reproducing the results

The end-to-end pipeline has four stages. Stages 1–3 produce artefacts that are *already cached in the repo under `data/`*, so you can skip directly to stage 4 to regenerate the paper's figures and tables.

### 1. Train backbones (optional — cached as Lightning checkpoints)

```bash
# classifiers: ResNet-101 and ViT for the four classification testbeds
python -m classifier.train

# segmentation backbones for the Polyp testbed
python -m segmentor.train

# Glow normalizing flows (only needed because the `typicality` detector queries log p(x))
python -m glow.trainpl
```

Checkpoints are written under `classifier_logs/`, `segmentation_logs/` and `glow_logs/`. The testbed constructors in `testbeds/` load specific checkpoint paths — you will need to update those paths if you retrain.

### 2. Collect OOD-detector feature data (optional — cached under `data/<model>/feature_data/`)

```bash
python eval_detectors.py
```

This sweeps every `(dataset, model, shift)` cell and writes one CSV per detector to `data/<model>/feature_data/<dataset>_<mode>_<detector>.csv`. Edit `DETECTORS`, `TESTBEDS`, or the `collect_all(...)` keyword arguments at the bottom of `eval_detectors.py` to recompute only a subset.

### 3. Collect PRE calibration data (optional — cached under `data/<model>/pra_data/`)

```bash
python -c "from experiments.pra import collect_re_accuracy_estimation_data; collect_re_accuracy_estimation_data()"
```

This runs the leave-one-synthetic-shift-type-out calibration of the `ErrorAdjustmentEstimator` PRE baseline and writes `data/<model>/pra_data/<dataset>_pre_results.csv`. The cached results are for the best-performing PRA configuration in terms of detector BA. 

### 4. Run the paper experiments

```bash
python experiments.py
```

This calls every figure/table function in `experiments/accuracy_prediction.py` and writes outputs into `figures/`. The key artefacts:

* `figures/corrected_shift_type_loo_comparison.csv` — LOSO MAE pivot
* `figures/shift_type_loo_rows.csv` — per-fold predictions used by the plotting routines
* the predicted-vs-true gap grid, per-accuracy error plot, and intensity breakdown PDFs

## Contact
For questions, send an e-mail to (ANONYMIZED)
