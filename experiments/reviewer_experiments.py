"""
Additional experiments added in response to the NeurIPS reviews.

Everything here is computed from the cached per-sample detector scores under
``data/<model>/feature_data/*.csv`` (columns: fold, feature_name, feature, loss,
acc, idx, class). No model retraining or re-inference is required, with the sole
exception of the *hard* Dice/mIoU segmentation metric (see
``experiments/segmentation_hard_metrics.py``), which needs the segmentation
checkpoints and is therefore shipped as a separate, user-run script.

Reviewer-point shorthand used in docstrings: RBFp, 67nM, YLV1 (Wn=weakness, Qn=question).

Outputs are written under ``figures/``. Each public function returns the table it
writes so it can also be inspected programmatically.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["text.usetex"] = False

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from sklearn.linear_model import LinearRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor

from utils import (
    load_all,
    load_data,
    DATASETS,
    DSDS,
    SEG_MODELS,
    SYNTHETIC_SHIFTS,
    DSD_PRINT_LUT,
)
from experiments.accuracy_prediction import (
    _parse_shift_type_from_fold,
    _is_synthetic_shift_type,
    _atc_threshold,
    _prepare_dr_gap_data,
    shift_type_loo_predictions,
    atc_predictions,
    pre_predictions,
    _shift_type_loo_bernoulli_sequences,
)

FIGDIR = "figures"


# ----------------------------------------------------------------------------
# Shared machinery: per-fold aggregate statistics + leave-one-shift-family-out
# regression, mirroring the exact protocol of shift_type_loo_predictions.
# ----------------------------------------------------------------------------
def _regressor_factories():
    """Named regressor factories for the linear-vs-nonlinear ablation (A5)."""

    class _Isotonic:
        """1-D isotonic regressor; direction auto-inferred (Spearman)."""

        def __init__(self):
            self._m = IsotonicRegression(out_of_bounds="clip", increasing="auto")

        def fit(self, X, y):
            self._m.fit(np.asarray(X).ravel(), np.asarray(y))
            return self

        def predict(self, X):
            return self._m.predict(np.asarray(X).ravel())

    def _gam():
        # pygam is already a project dependency (components.py).
        from pygam import LinearGAM

        class _GAM:
            def __init__(self):
                self._m = None

            def fit(self, X, y):
                self._m = LinearGAM().fit(np.asarray(X), np.asarray(y))
                return self

            def predict(self, X):
                return np.asarray(self._m.predict(np.asarray(X)))

        return _GAM()

    return {
        "Linear": lambda: LinearRegression(),
        "Poly-2": lambda: make_pipeline(PolynomialFeatures(2), LinearRegression()),
        "Isotonic": lambda: _Isotonic(),
        "GAM": _gam,
        "RandomForest": lambda: RandomForestRegressor(
            n_estimators=100, max_depth=4, random_state=0
        ),
    }


def _loso_from_perfold(perfold, dataset, model, feature_name, method, regressor=None):
    """
    Leave-one-shift-family-out regression on a per-fold table.

    ``perfold`` must have columns: fold, shift_type, category, stat, gap.
    Same protocol as accuracy_prediction.shift_type_loo_predictions: hold out one
    synthetic shift family at a time (train on the other synthetic families),
    and evaluate every organic fold with a regressor trained on the full
    synthetic pool. Organic folds are never used for fitting.
    """
    if regressor is None:
        regressor = LinearRegression
    rows = []
    pool = perfold[
        (perfold["category"] == "Synthetic") & (~perfold["shift_type"].isin(["ind"]))
    ]
    if len(pool) < 2:
        return rows

    def emit(r, pred, st, cat):
        rows.append({
            "Dataset": dataset,
            "Model": model,
            "feature_name": feature_name,
            "fold": r["fold"],
            "shift_type": st,
            "category": cat,
            "observed_gap": float(r["gap"]),
            "predicted_gap": float(pred),
            "MAE": abs(float(pred) - float(r["gap"])),
            "Method": method,
        })

    for held in pool["shift_type"].unique():
        train = pool[pool["shift_type"] != held]
        test = pool[pool["shift_type"] == held]
        if len(train) < 2 or test.empty:
            continue
        reg = regressor().fit(train[["stat"]].values, train["gap"].values)
        preds = np.asarray(reg.predict(test[["stat"]].values)).ravel()
        for (_, r), p in zip(test.iterrows(), preds):
            emit(r, p, held, "Synthetic")

    organic = perfold[
        (perfold["category"] == "Organic")
        & (~perfold["fold"].isin(["train", "ind_val"]))
    ]
    if not organic.empty:
        reg = regressor().fit(pool[["stat"]].values, pool["gap"].values)
        for _, r in organic.iterrows():
            p = float(np.asarray(reg.predict(np.array([[float(r["stat"])]]))).ravel()[0])
            emit(r, p, "Organic", "Organic")
    return rows


def _index_raw(raw):
    """Index raw per-sample scores by (Dataset, Model, feature_name) for O(1) lookup.

    The raw frame has ~10M rows, so repeated boolean masks are the dominant cost;
    a one-time groupby dict avoids re-scanning the whole frame per detector.
    """
    return {k: g for k, g in raw.groupby(["Dataset", "Model", "feature_name"])}


def _get_sub(raw, dataset, model, feature_file):
    """Fetch the (dataset, model, detector) slice from a raw frame OR a prebuilt index."""
    if isinstance(raw, dict):
        return raw.get((dataset, model, feature_file))
    sub = raw[
        (raw["Dataset"] == dataset)
        & (raw["Model"] == model)
        & (raw["feature_name"] == feature_file)
    ]
    return sub if not sub.empty else None


def _perfold_stat(raw, dataset, model, feature_file, agg, performance_col=None):
    """
    Build a per-fold table for one (dataset, model, detector-file).

    ``agg`` maps a fold's sample DataFrame -> scalar statistic (stored as 'stat').
    ``performance_col``:
        None  -> performance is the binary correctness rate (mean correct_prediction);
        'acc' -> performance is the continuous per-sample value in the 'acc' column
                 (soft IoU for Polyp; used by the continuous-segmentation experiment A7).
    Returns columns [fold, shift_type, category, stat, Accuracy, gap] or None.
    """
    sub = _get_sub(raw, dataset, model, feature_file)
    if sub is None or sub.empty:
        return None
    ind_val = sub[sub["fold"] == "ind_val"]
    if ind_val.empty:
        return None

    def perf(fdf):
        if performance_col is None:
            return float(fdf["correct_prediction"].mean())
        return float(fdf[performance_col].mean())

    ind_val_perf = perf(ind_val)
    recs = []
    for fold, fdf in sub.groupby("fold"):
        if fold == "train":
            continue
        st = _parse_shift_type_from_fold(fold)
        recs.append({
            "fold": fold,
            "shift_type": st,
            "category": "Synthetic" if _is_synthetic_shift_type(st) else "Organic",
            "stat": float(agg(fdf)),
            "Accuracy": perf(fdf),
        })
    if not recs:
        return None
    out = pd.DataFrame(recs)
    out["gap"] = ind_val_perf - out["Accuracy"]
    return out


def _perfold_dr(raw, dataset, model, feature_file, q, performance_col=None):
    """
    Per-fold OOD detection rate at InD quantile ``q`` (id_quantile thresholding),
    recomputed from raw per-sample scores. Replicates OODDetector(id_quantile).
    Direction (higher_is_ood) is inferred from InD vs. all-OOD means.
    """
    sub = _get_sub(raw, dataset, model, feature_file)
    if sub is None or sub.empty:
        return None
    ind = sub[~sub["ood"]]
    ood = sub[sub["ood"]]
    if ind.empty or ood.empty:
        return None
    higher_is_ood = ood["feature"].mean() > ind["feature"].mean()
    thr = (
        float(ind["feature"].quantile(q))
        if higher_is_ood
        else float(ind["feature"].quantile(1.0 - q))
    )

    def dr(fdf):
        v = fdf["feature"].values
        return float((v > thr).mean()) if higher_is_ood else float((v < thr).mean())

    return _perfold_stat(raw, dataset, model, feature_file, dr, performance_col)


def _best_config_pivot(df, method_order, out_csv=None):
    """Select best (Model, feature) per (Dataset, Method) by mean MAE; return MAE pivot."""
    df = df[~df["fold"].isin(["train", "ind_val"]) & ~df["shift_type"].isin(["ind"])].copy()
    cs = df.groupby(["Dataset", "Method", "Model", "feature_name"], as_index=False)["MAE"].mean()
    best = cs.loc[
        cs.groupby(["Dataset", "Method"])["MAE"].idxmin(),
        ["Dataset", "Method", "Model", "feature_name"],
    ]
    dfb = df.merge(best, on=["Dataset", "Method", "Model", "feature_name"], how="inner")
    summary = dfb.groupby(["Dataset", "Method"], as_index=False)["MAE"].mean()
    pivot = summary.pivot(index="Dataset", columns="Method", values="MAE").reindex(DATASETS)
    order = [m for m in method_order if m in pivot.columns]
    pivot = pivot.reindex(columns=order)
    if out_csv:
        os.makedirs(FIGDIR, exist_ok=True)
        pivot.round(4).to_csv(out_csv)
    return pivot, dfb


# ----------------------------------------------------------------------------
# A1. Equal-calibration-resource baselines (67nM W2/Q1, YLV1 W5, meta-review)
# ----------------------------------------------------------------------------
def regression_baseline_predictions(batch_size=1, raw=None):
    """
    Confidence/entropy/energy/ATC-statistic regressions trained on the SAME
    synthetic calibration shifts as our DR estimator. This isolates whether the
    gain comes from OOD detection rates specifically or merely from fitting a
    regression on labeled synthetic shifts.

    Methods produced:
        Reg-MSP     : mean max-softmax over the fold        (feature file 'softmax')
        Reg-Entropy : mean predictive entropy               (feature file 'cross_entropy')
        Reg-Energy  : mean energy score                     (feature file 'energy')
        Reg-ATC     : ATC soft error rate = P(softmax<tau)  (feature file 'softmax')
    (Reg-MSP/Reg-ATC are unavailable for Polyp: MSP is undefined for segmentation.)
    """
    if raw is None:
        raw = load_all(batch_size=batch_size, shift="")
    if raw.empty:
        return pd.DataFrame()

    mean_feat = lambda f: f["feature"].mean()
    specs = [
        ("Reg-MSP", "softmax", mean_feat),
        ("Reg-Entropy", "cross_entropy", mean_feat),
        ("Reg-Energy", "energy", mean_feat),
    ]

    idx = _index_raw(raw)
    rows = []
    for (dataset, model) in sorted({(ds, m) for (ds, m, _f) in idx}):
        for method, ff, agg in specs:
            pf = _perfold_stat(idx, dataset, model, ff, agg)
            if pf is None or pf.empty:
                continue
            rows += _loso_from_perfold(pf, dataset, model, ff, method)

        # ATC soft-rate feature: fraction of the fold below the InD ATC threshold.
        sm = idx.get((dataset, model, "softmax"))
        iv = sm[sm["fold"] == "ind_val"] if sm is not None else None
        if iv is not None and not iv.empty:
            tau = _atc_threshold(iv["feature"].values, iv["correct_prediction"].values)
            if not np.isnan(tau):
                pf = _perfold_stat(
                    idx, dataset, model, "softmax",
                    lambda f, t=tau: float((f["feature"].values < t).mean()),
                )
                if pf is not None and not pf.empty:
                    rows += _loso_from_perfold(pf, dataset, model, "softmax", "Reg-ATC")

    return pd.DataFrame(rows)


def equal_resource_baseline_table(batch_size=1, ours=None, atc=None, reg=None):
    """A1 main table: Ours vs equal-resource regressions vs vanilla ATC."""
    if ours is None:
        ours = shift_type_loo_predictions(batch_size=batch_size)
    if atc is None:
        atc = atc_predictions(batch_size=batch_size)
    if reg is None:
        reg = regression_baseline_predictions(batch_size=batch_size)
    dfs = [d for d in [ours, atc, reg] if d is not None and not d.empty]
    if not dfs:
        print("[equal_resource_baseline_table] no rows.")
        return pd.DataFrame(), pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True)
    order = ["Ours", "Reg-MSP", "Reg-Entropy", "Reg-Energy", "Reg-ATC", "ATC-MC", "ATC-NE"]
    pivot, dfb = _best_config_pivot(df, order, out_csv=f"{FIGDIR}/equal_resource_baselines.csv")
    print("\n=== A1: Equal-calibration-resource baselines (MAE) ===")
    print(pivot.round(4).to_string())
    return pivot, dfb


# ----------------------------------------------------------------------------
# A2. Anti-correlation / detectable-but-not-harmful stress test (YLV1 W2/W7, RBFp)
# (The signed-rho Figure 2 change lives in accuracy_prediction.dr_gap_correlation_distribution.)
# ----------------------------------------------------------------------------
def stress_test_analysis(batch_size=1, dr_high=0.5, gap_low=0.05, gap_high=0.2, dr_low=0.2):
    """
    Quantify where the DR-gap monotonicity assumption breaks:
      (i)  detectable-but-not-harmful : DR >= dr_high but gap <= gap_low
      (ii) harmful-but-weakly-detectable : gap >= gap_high but DR <= dr_low
    Reported per (Dataset, detector), with the resulting estimator error in those cells.
    """
    df = _prepare_dr_gap_data(batch_size=batch_size, filter_best=False)
    if df.empty:
        return pd.DataFrame()
    df = df[~df["fold"].isin(["train", "ind_val"])].copy()

    df["detectable_not_harmful"] = (df["DR"] >= dr_high) & (df["gap"] <= gap_low)
    df["harmful_not_detectable"] = (df["gap"] >= gap_high) & (df["DR"] <= dr_low)

    rows = []
    for (dataset, feat), g in df.groupby(["Dataset", "feature_name"]):
        n = len(g)
        rows.append({
            "Dataset": dataset,
            "Detector": DSD_PRINT_LUT.get(feat, feat),
            "n_folds": n,
            "pct_detectable_not_harmful": 100.0 * g["detectable_not_harmful"].mean(),
            "pct_harmful_not_detectable": 100.0 * g["harmful_not_detectable"].mean(),
            "spearman_rho": (
                spearmanr(g["DR"], g["gap"]).correlation
                if g["DR"].nunique() > 1 and g["gap"].nunique() > 1
                else np.nan
            ),
        })
    out = pd.DataFrame(rows).sort_values(["Dataset", "Detector"])
    os.makedirs(FIGDIR, exist_ok=True)
    out.round(4).to_csv(f"{FIGDIR}/stress_test.csv", index=False)

    # Annotated scatter (all configs), flagging the two failure quadrants.
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.scatter(df["DR"], df["gap"], s=8, alpha=0.35, color="grey", label="all folds")
    dnh = df[df["detectable_not_harmful"]]
    hnd = df[df["harmful_not_detectable"]]
    ax.scatter(dnh["DR"], dnh["gap"], s=14, color="tab:orange", label="detectable, not harmful")
    ax.scatter(hnd["DR"], hnd["gap"], s=14, color="tab:red", label="harmful, weakly detectable")
    ax.axhline(gap_low, color="tab:orange", lw=0.6, ls="--")
    ax.axhline(gap_high, color="tab:red", lw=0.6, ls="--")
    ax.axvline(dr_high, color="tab:orange", lw=0.6, ls="--")
    ax.axvline(dr_low, color="tab:red", lw=0.6, ls="--")
    ax.set_xlabel("OOD detection rate (DR)")
    ax.set_ylabel("Generalization gap")
    ax.legend(fontsize="x-small", frameon=False)
    plt.tight_layout()
    plt.savefig(f"{FIGDIR}/stress_test.pdf", bbox_inches="tight")
    plt.close(fig)

    print("\n=== A2: DR-gap stress test (share of folds in each failure quadrant) ===")
    print(out.round(3).to_string(index=False))
    return out


# ----------------------------------------------------------------------------
# A2b. Detector quality vs. DR-gap coupling (tests the "weak detector, not broken
# premise" hypothesis for the anti-correlated detectors). Cached data only.
# ----------------------------------------------------------------------------
def detector_quality_vs_coupling(batch_size=1):
    """
    Does the DR-gap coupling strength track a detector's own quality?

    For each (dataset, architecture, detector) we compute two intrinsic quality
    scores from cached per-sample data:
      * ood_auroc  : AUROC of the raw score separating InD vs. organic-OOD samples
      * fail_auroc : AUROC of the raw score separating correct vs. incorrect preds
    (both folded to [0.5,1] since detector orientation is arbitrary), and correlate
    them with the SIGNED Spearman DR-gap rho from figures/dr_gap_signed_correlation.csv.

    If good detectors couple positively and only weak ones go negative, the
    anti-correlations reflect detector quality rather than a broken premise.
    """
    rho_path = f"{FIGDIR}/dr_gap_signed_correlation.csv"
    if not os.path.exists(rho_path):
        print("[detector_quality_vs_coupling] run dr_gap_correlation_distribution first.")
        return pd.DataFrame()
    rho_df = pd.read_csv(rho_path)  # columns: Dataset, Model, Detector (print), rho

    raw = load_all(batch_size=batch_size, shift="normal")
    if raw.empty:
        return pd.DataFrame()

    rows = []
    for (dataset, model, feat), g in raw.groupby(["Dataset", "Model", "feature_name"]):
        g = g.dropna(subset=["feature"])
        y_ood = g["ood"].astype(int).values
        y_fail = (~g["correct_prediction"].astype(bool)).astype(int).values
        s = g["feature"].values

        def _auc(y):
            if len(np.unique(y)) < 2:
                return np.nan
            a = roc_auc_score(y, s)
            return max(a, 1.0 - a)  # orientation-agnostic separability

        rows.append({
            "Dataset": dataset,
            "Model": model,
            "Detector": DSD_PRINT_LUT.get(feat, feat),
            "ood_auroc": _auc(y_ood),
            "fail_auroc": _auc(y_fail),
        })
    q = pd.DataFrame(rows)
    merged = q.merge(rho_df[["Dataset", "Model", "Detector", "rho"]],
                     on=["Dataset", "Model", "Detector"], how="inner")
    if merged.empty:
        print("[detector_quality_vs_coupling] no overlap with rho table.")
        return merged

    os.makedirs(FIGDIR, exist_ok=True)
    merged.round(4).to_csv(f"{FIGDIR}/detector_quality_vs_coupling.csv", index=False)

    def _corr(col):
        sub = merged[[col, "rho"]].dropna()
        if len(sub) < 4:
            return np.nan
        return spearmanr(sub[col], sub["rho"]).correlation

    r_ood = _corr("ood_auroc")
    r_fail = _corr("fail_auroc")
    print("\n=== A2b: detector quality vs. signed DR-gap coupling ===")
    print(merged.round(3).to_string(index=False))
    print(f"\nSpearman(ood-AUROC, signed rho)  = {r_ood:.3f}")
    print(f"Spearman(fail-AUROC, signed rho) = {r_fail:.3f}")
    print("(positive => better detectors couple more positively with the gap)")

    # scatter
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, col, lab, r in [(axes[0], "ood_auroc", "OOD-detection AUROC", r_ood),
                            (axes[1], "fail_auroc", "failure-detection AUROC", r_fail)]:
        ax.scatter(merged[col], merged["rho"], s=18, alpha=0.7)
        ax.axhline(0.0, color="black", lw=0.8)
        ax.set_xlabel(lab)
        ax.set_ylabel("signed DR-gap rho")
        ax.set_title(f"Spearman = {r:.2f}")
    plt.tight_layout()
    plt.savefig(f"{FIGDIR}/detector_quality_vs_coupling.pdf", bbox_inches="tight")
    plt.close(fig)
    return merged


# ----------------------------------------------------------------------------
# Label-free accuracy-estimation SOTA baselines (YLV1 W1: "more recent, stronger
# baselines"). AC / DoC are computable from cached max-softmax; Nuclear Norm and
# COT need the full softmax matrix and are produced by collect_softmax_stats.py.
# ----------------------------------------------------------------------------
def label_free_baselines(batch_size=1, datasets=None):
    """Average-Confidence (AC) and Difference-of-Confidences (DoC, Guillory 2021),
    in their native (non-regression) form, from cached max-softmax.

    Returns rows in the standard schema so they drop into the comparison tables.
    """
    datasets = datasets or ["CCT", "OfficeHome", "Office31", "NICO", "Camelyon17", "IWildCam"]
    rows = []
    for ds in datasets:
        for model in ["resnet", "vit"]:
            try:
                df = load_data(ds, "msp", batch_size=batch_size, samples=1, model=model, shift="")
            except Exception:
                continue  # incomplete cell (e.g. 'normal' mode not yet collected)
            if df.empty:
                continue
            ind = df[df["fold"] == "ind_val"]
            if ind.empty:
                continue
            ind_acc = float(ind["correct_prediction"].mean())
            ind_conf = float(ind["feature"].mean())  # mean max-softmax on InD-val
            ev = df[~df["fold"].isin(["train", "ind_val", "ind_test"])]
            for fold, g in ev.groupby("fold"):
                conf = float(g["feature"].mean())
                true_acc = float(g["correct_prediction"].mean())
                st = _parse_shift_type_from_fold(fold)
                cat = "Synthetic" if _is_synthetic_shift_type(st) else "Organic"
                obs = ind_acc - true_acc
                base = {"Dataset": ds, "Model": model, "feature_name": "softmax",
                        "fold": fold, "shift_type": st, "category": cat, "observed_gap": obs}
                # AC: predicted accuracy = mean confidence
                rows.append({**base, "Method": "AC",
                             "predicted_gap": ind_acc - conf, "MAE": abs(conf - true_acc)})
                # DoC: predicted acc = ind_acc - (ind_conf - conf)
                doc = ind_acc - (ind_conf - conf)
                rows.append({**base, "Method": "DoC",
                             "predicted_gap": ind_acc - doc, "MAE": abs(doc - true_acc)})
    return pd.DataFrame(rows)


def _softmax_stat_predictions(path):
    """Turn per-fold Nuclear-Norm / COT scores (from collect_softmax_stats.py) into
    accuracy predictions via the same leave-one-shift-family-out linear calibration."""
    s = pd.read_csv(path)
    rows = []
    for (ds, m), g in s.groupby(["Dataset", "Model"]):
        ind = g[g["fold"] == "ind_val"]
        if ind.empty:
            continue
        ind_acc = float(ind["true_acc"].mean())
        ev = g[~g["fold"].isin(["train", "ind_val", "ind_test"])].copy()
        ev["shift_type"] = ev["fold"].map(_parse_shift_type_from_fold)
        ev["category"] = ev["shift_type"].map(
            lambda st: "Synthetic" if _is_synthetic_shift_type(st) else "Organic")
        ev["gap"] = ind_acc - ev["true_acc"]
        for method, col in [("NuclearNorm", "nuclear_norm"), ("COT", "cot")]:
            if col not in ev.columns or ev[col].isna().all():
                continue
            pf = ev.rename(columns={col: "stat"})[["fold", "shift_type", "category", "stat", "gap"]].dropna()
            rows += _loso_from_perfold(pf, ds, m, "softmax", method)
    return pd.DataFrame(rows)


def sota_baseline_table(batch_size=1, datasets=None):
    """Compare Ours vs label-free accuracy-estimation baselines (ATC, AC, DoC, and
    Nuclear Norm / COT if collect_softmax_stats.py has been run)."""
    ours = shift_type_loo_predictions(batch_size=batch_size)
    atc = atc_predictions(batch_size=batch_size)
    lf = label_free_baselines(batch_size=batch_size, datasets=datasets)
    frames = [d for d in [ours, atc, lf] if d is not None and not d.empty]
    # optional: Nuclear Norm / COT from the softmax-stats collector
    nn_path = f"{FIGDIR}/softmax_stats.csv"
    if os.path.exists(nn_path):
        frames.append(_softmax_stat_predictions(nn_path))
    df = pd.concat(frames, ignore_index=True)
    order = ["Ours", "ATC-MC", "ATC-NE", "AC", "DoC", "NuclearNorm", "COT"]
    pivot, dfb = _best_config_pivot(df, order, out_csv=f"{FIGDIR}/sota_baselines.csv")
    print("\n=== Label-free accuracy-estimation baselines (MAE) ===")
    print(pivot.round(4).to_string())
    return pivot, dfb


# ----------------------------------------------------------------------------
# A3. Per-fixed-architecture results (RBFp Q1, 67nM W5/Q5, YLV1)
# ----------------------------------------------------------------------------
def per_architecture_table(batch_size=1, ours=None, atc=None, pre=None):
    """
    MAE reported per FIXED architecture (only the detector/estimator is selected,
    not the architecture), plus a single-detector-held-fixed-across-datasets row.
    """
    if ours is None:
        ours = shift_type_loo_predictions(batch_size=batch_size)
    if atc is None:
        atc = atc_predictions(batch_size=batch_size)
    if pre is None:
        pre = pre_predictions()
    df = pd.concat([d for d in [ours, atc, pre] if not d.empty], ignore_index=True)
    df = df[~df["fold"].isin(["train", "ind_val"]) & ~df["shift_type"].isin(["ind"])].copy()

    # best detector per (Dataset, Model, Method)
    cs = df.groupby(["Dataset", "Model", "Method", "feature_name"], as_index=False)["MAE"].mean()
    best = cs.loc[cs.groupby(["Dataset", "Model", "Method"])["MAE"].idxmin()]
    tab = best.pivot_table(index=["Model", "Method"], columns="Dataset", values="MAE")
    tab = tab.reindex(columns=[d for d in DATASETS if d in tab.columns])
    os.makedirs(FIGDIR, exist_ok=True)
    tab.round(4).to_csv(f"{FIGDIR}/per_architecture_mae.csv")

    # single fixed detector across all datasets, for Ours (RBFp Q1)
    ours_df = df[df["Method"] == "Ours"]
    fixed_rows = []
    if not ours_df.empty:
        per_det_ds = ours_df.groupby(["feature_name", "Dataset"])["MAE"].mean().reset_index()
        det_mean = per_det_ds.groupby("feature_name")["MAE"].mean()
        best_det = det_mean.idxmin()
        fx = per_det_ds[per_det_ds["feature_name"] == best_det].set_index("Dataset")["MAE"]
        for ds in DATASETS:
            fixed_rows.append({
                "Dataset": ds,
                "fixed_detector": DSD_PRINT_LUT.get(best_det, best_det),
                "MAE_fixed_detector": float(fx.get(ds, np.nan)),
            })
        fixed = pd.DataFrame(fixed_rows)
        fixed.round(4).to_csv(f"{FIGDIR}/per_architecture_fixed_detector.csv", index=False)
        print(f"\n=== A3: single detector fixed across datasets (Ours): {DSD_PRINT_LUT.get(best_det, best_det)} ===")
        print(fixed.round(4).to_string(index=False))

    print("\n=== A3: per-fixed-architecture MAE ===")
    print(tab.round(4).to_string())
    return tab


# ----------------------------------------------------------------------------
# A4. Threshold sensitivity + feature-type ablation (67nM Q2, RBFp, YLV1 W3)
# ----------------------------------------------------------------------------
def threshold_sensitivity(batch_size=1, quantiles=(0.80, 0.90, 0.95, 0.99), raw=None):
    """MAE of the DR regression as the InD detection-threshold quantile is swept."""
    if raw is None:
        raw = load_all(batch_size=batch_size, shift="")
    if raw.empty:
        return pd.DataFrame()
    idx = _index_raw(raw)
    dm_keys = sorted({(ds, m) for (ds, m, _f) in idx})
    records = []
    for q in quantiles:
        rows = []
        for (dataset, model) in dm_keys:
            for ff in DSDS_INTERNAL:
                pf = _perfold_dr(idx, dataset, model, ff, q)
                if pf is None or pf.empty:
                    continue
                rows += _loso_from_perfold(pf, dataset, model, ff, "Ours")
        if not rows:
            continue
        pivot, _ = _best_config_pivot(pd.DataFrame(rows), ["Ours"])
        for ds in pivot.index:
            records.append({"Dataset": ds, "quantile": q, "MAE": float(pivot.loc[ds, "Ours"])})
    out = pd.DataFrame(records)
    os.makedirs(FIGDIR, exist_ok=True)
    out.pivot(index="Dataset", columns="quantile", values="MAE").round(4).to_csv(
        f"{FIGDIR}/threshold_sensitivity.csv"
    )
    print("\n=== A4: threshold sensitivity (MAE vs InD quantile) ===")
    print(out.pivot(index="Dataset", columns="quantile", values="MAE").round(4).to_string())
    return out


def feature_type_ablation(batch_size=1, raw=None):
    """Compare regressor input: binary DR@0.95 vs mean raw score vs 95th-quantile score."""
    if raw is None:
        raw = load_all(batch_size=batch_size, shift="")
    if raw.empty:
        return pd.DataFrame()

    idx = _index_raw(raw)
    dm_keys = sorted({(ds, m) for (ds, m, _f) in idx})
    feature_types = {
        "DR@0.95": lambda ds, m, ff: _perfold_dr(idx, ds, m, ff, 0.95),
        "MeanScore": lambda ds, m, ff: _perfold_stat(idx, ds, m, ff, lambda f: f["feature"].mean()),
        "Q95Score": lambda ds, m, ff: _perfold_stat(idx, ds, m, ff, lambda f: f["feature"].quantile(0.95)),
    }
    records = []
    for ftype, builder in feature_types.items():
        rows = []
        for (dataset, model) in dm_keys:
            for ff in DSDS_INTERNAL:
                pf = builder(dataset, model, ff)
                if pf is None or pf.empty:
                    continue
                rows += _loso_from_perfold(pf, dataset, model, ff, "Ours")
        if not rows:
            continue
        pivot, _ = _best_config_pivot(pd.DataFrame(rows), ["Ours"])
        for ds in pivot.index:
            records.append({"Dataset": ds, "feature_type": ftype, "MAE": float(pivot.loc[ds, "Ours"])})
    out = pd.DataFrame(records)
    os.makedirs(FIGDIR, exist_ok=True)
    tab = out.pivot(index="Dataset", columns="feature_type", values="MAE")
    tab.round(4).to_csv(f"{FIGDIR}/feature_type_ablation.csv")
    print("\n=== A4: feature-type ablation (MAE) ===")
    print(tab.round(4).to_string())
    return out


# ----------------------------------------------------------------------------
# DR vs raw-score robustness: is the bounded detection rate more robust /
# transferable than a raw aggregated score, even at similar best-case MAE?
# ----------------------------------------------------------------------------
def dr_vs_raw_robustness(batch_size=1, raw=None):
    """
    Compare binary detection rate (DR@0.95) against the mean raw score as the
    regression feature, on THREE robustness axes rather than best-case MAE:
      * best        : MAE of the best detector per dataset (what the paper reports)
      * mean_det    : MAE averaged over ALL detectors (robustness to detector choice)
      * worst_det   : MAE of the worst detector (heavy-tailed-score blow-up)
      * organic     : MAE on organic held-out folds only (real cross-shift transfer)
    A bounded rate should be far less sensitive to detector choice and to
    heavy-tailed raw scores than an unbounded mean.
    """
    if raw is None:
        raw = load_all(batch_size=batch_size, shift="")
    if raw.empty:
        return pd.DataFrame()
    idx = _index_raw(raw)
    dm_keys = sorted({(ds, m) for (ds, m, _f) in idx})
    builders = {
        "DR@0.95": lambda ds, m, ff: _perfold_dr(idx, ds, m, ff, 0.95),
        "MeanScore": lambda ds, m, ff: _perfold_stat(idx, ds, m, ff, lambda f: f["feature"].mean()),
    }
    allrows = []
    for ftype, build in builders.items():
        for (ds, m) in dm_keys:
            for ff in DSDS_INTERNAL:
                pf = build(ds, m, ff)
                if pf is None or pf.empty:
                    continue
                for r in _loso_from_perfold(pf, ds, m, ff, ftype):
                    r["feature_type"] = ftype
                    allrows.append(r)
    df = pd.DataFrame(allrows)
    if df.empty:
        return df

    per_det = df.groupby(["feature_type", "Dataset", "feature_name"])["MAE"].mean().reset_index()
    agg = per_det.groupby(["feature_type", "Dataset"])["MAE"].agg(
        best="min", mean_det="mean", worst_det="max").reset_index()
    org = (df[df["category"] == "Organic"]
           .groupby(["feature_type", "Dataset", "feature_name"])["MAE"].mean().reset_index())
    org_agg = org.groupby(["feature_type", "Dataset"])["MAE"].mean().reset_index().rename(
        columns={"MAE": "organic_mean_det"})
    out = agg.merge(org_agg, on=["feature_type", "Dataset"], how="left")

    os.makedirs(FIGDIR, exist_ok=True)
    out.round(4).to_csv(f"{FIGDIR}/dr_vs_raw_robustness.csv", index=False)

    print("\n=== DR vs raw-score robustness (lower is better) ===")
    for metric in ["best", "mean_det", "worst_det", "organic_mean_det"]:
        piv = out.pivot(index="Dataset", columns="feature_type", values=metric)
        print(f"\n[{metric}]")
        print(piv.round(4).to_string())
    # headline: macro-average across datasets
    macro = out.groupby("feature_type")[["best", "mean_det", "worst_det", "organic_mean_det"]].mean()
    print("\n[macro-average across datasets]")
    print(macro.round(4).to_string())
    return out


# ----------------------------------------------------------------------------
# A5. Linear vs non-linear regressor ablation (YLV1 W3, RBFp Q3, 67nM)
# ----------------------------------------------------------------------------
def regressor_ablation(batch_size=1):
    """
    Compare the deliberate LinearRegression against non-linear alternatives on the
    canonical DR feature (val-optimal DR, matching the paper's 'Ours').
    Reports MAE and the calibration slope (predicted-on-observed gap).
    """
    prep = _prepare_dr_gap_data(batch_size=batch_size, filter_best=False)
    if prep.empty:
        return pd.DataFrame()
    prep = prep.rename(columns={"DR": "stat"})
    prep = prep[~prep["fold"].isin(["train", "ind_val"])].copy()

    records = []
    for name, factory in _regressor_factories().items():
        rows = []
        for (ds, m, fn), g in prep.groupby(["Dataset", "Model", "feature_name"]):
            rows += _loso_from_perfold(g, ds, m, fn, name, regressor=factory)
        if not rows:
            continue
        rdf = pd.DataFrame(rows)
        pivot, dfb = _best_config_pivot(rdf, [name])
        # global calibration slope on the selected configurations
        sl = np.polyfit(dfb["observed_gap"], dfb["predicted_gap"], 1)[0] if len(dfb) > 2 else np.nan
        for ds in pivot.index:
            records.append({
                "Dataset": ds, "regressor": name,
                "MAE": float(pivot.loc[ds, name]), "slope": float(sl),
            })
    out = pd.DataFrame(records)
    os.makedirs(FIGDIR, exist_ok=True)
    mae_tab = out.pivot(index="Dataset", columns="regressor", values="MAE")
    mae_tab.round(4).to_csv(f"{FIGDIR}/regressor_ablation.csv")
    print("\n=== A5: regressor ablation (MAE; slope in CSV) ===")
    print(mae_tab.round(4).to_string())
    return out


# ----------------------------------------------------------------------------
# A6. Detector-count / ensembling ablation (YLV1 W3/Q3)
# ----------------------------------------------------------------------------
def detector_count_ablation(batch_size=1):
    """
    MAE as a function of the number of detectors combined. For k detectors, the
    regressor uses a k-dimensional DR feature vector (multivariate LOSO). k=1 is
    the single-best-detector regime; k=6 uses all detectors jointly.
    """
    prep = _prepare_dr_gap_data(batch_size=batch_size, filter_best=False)
    if prep.empty:
        return pd.DataFrame()
    prep = prep[~prep["fold"].isin(["train", "ind_val"])].copy()

    records = []
    for (dataset, model), g in prep.groupby(["Dataset", "Model"]):
        # wide DR table: one column per detector, indexed by fold.
        # NOTE: _prepare_dr_gap_data yields detector names in PRINT form.
        wide = g.pivot_table(index=["fold", "shift_type", "category", "gap"],
                             columns="feature_name", values="DR").reset_index()
        det_order = list(DSD_PRINT_LUT.values())  # stable print-name order
        det_cols = [c for c in det_order if c in wide.columns]
        wide = wide.dropna(subset=det_cols)
        if wide.empty or len(det_cols) < 1:
            continue
        pool = wide[(wide["category"] == "Synthetic") & (~wide["shift_type"].isin(["ind"]))]
        organic = wide[(wide["category"] == "Organic") & (~wide["fold"].isin(["train", "ind_val"]))]
        if len(pool) < 2:
            continue
        for k in range(1, len(det_cols) + 1):
            cols = det_cols[:k]
            errs = []
            for held in pool["shift_type"].unique():
                tr = pool[pool["shift_type"] != held]
                te = pool[pool["shift_type"] == held]
                if len(tr) < 2 or te.empty:
                    continue
                reg = LinearRegression().fit(tr[cols].values, tr["gap"].values)
                errs += list(np.abs(reg.predict(te[cols].values) - te["gap"].values))
            if not organic.empty:
                reg = LinearRegression().fit(pool[cols].values, pool["gap"].values)
                errs += list(np.abs(reg.predict(organic[cols].values) - organic["gap"].values))
            if errs:
                records.append({"Dataset": dataset, "Model": model, "k": k, "MAE": float(np.mean(errs))})

    out = pd.DataFrame(records)
    if out.empty:
        return out
    # average over architectures per dataset
    tab = out.groupby(["Dataset", "k"])["MAE"].mean().reset_index()
    os.makedirs(FIGDIR, exist_ok=True)
    tab.pivot(index="Dataset", columns="k", values="MAE").round(4).to_csv(
        f"{FIGDIR}/detector_count_ablation.csv"
    )
    print("\n=== A6: detector-count ablation (MAE vs #detectors) ===")
    print(tab.pivot(index="Dataset", columns="k", values="MAE").round(4).to_string())
    return out


# ----------------------------------------------------------------------------
# A7. Continuous soft-IoU segmentation target (YLV1 W10) — cache-feasible part
# ----------------------------------------------------------------------------
def polyp_continuous_iou(batch_size=1, raw=None):
    """
    Refit the Polyp estimator against the CONTINUOUS soft-IoU degradation (the
    per-sample 'acc' column = 1 - Jaccard loss) rather than the binary IoU>0.5
    correctness, and check that low MAE / good calibration still hold.
    Hard mIoU/Dice needs re-inference: see experiments/segmentation_hard_metrics.py.
    """
    if raw is None:
        raw = load_all(batch_size=batch_size, shift="")
    raw = raw[raw["Dataset"] == "Polyp"]
    if raw.empty:
        print("[polyp_continuous_iou] no Polyp data.")
        return pd.DataFrame()

    idx = _index_raw(raw)
    rows = []
    for (dataset, model) in sorted({(ds, m) for (ds, m, _f) in idx}):
        for ff in DSDS_INTERNAL:
            pf = _perfold_dr(idx, dataset, model, ff, 0.95, performance_col="acc")
            if pf is None or pf.empty:
                continue
            rows += _loso_from_perfold(pf, dataset, model, ff, "Ours-softIoU")
    if not rows:
        return pd.DataFrame()
    pivot, dfb = _best_config_pivot(pd.DataFrame(rows), ["Ours-softIoU"])
    slope = np.polyfit(dfb["observed_gap"], dfb["predicted_gap"], 1)[0] if len(dfb) > 2 else np.nan
    os.makedirs(FIGDIR, exist_ok=True)
    pivot.round(4).to_csv(f"{FIGDIR}/polyp_continuous_iou.csv")
    print("\n=== A7: Polyp continuous soft-IoU target (MAE) ===")
    print(pivot.round(4).to_string())
    print(f"calibration slope (predicted vs observed soft-IoU gap): {slope:.3f}")
    return pivot


# ----------------------------------------------------------------------------
# A8. Sequence-length prediction intervals (67nM Q4)
# ----------------------------------------------------------------------------
def sequence_length_intervals(
    lengths=(2, 4, 8, 16, 32, 64, 128, 256, 512), n_samples=200, random_state=0
):
    """
    Empirical prediction intervals of the estimator's error as a function of
    sequence length, giving a practitioner an "attach +/- X at n=64" statement.
    Reuses the Bernoulli sequence simulation from accuracy_prediction.
    """
    seq = _shift_type_loo_bernoulli_sequences(
        lengths=list(lengths), n_samples=n_samples, batch_size=1, random_state=random_state
    )
    if seq.empty:
        return pd.DataFrame()
    seq = seq.copy()
    seq["signed_error"] = seq["predicted_gap"] - seq["observed_gap"]
    g = seq.groupby("sequence_length")
    out = pd.DataFrame({
        "sequence_length": sorted(seq["sequence_length"].unique()),
    }).set_index("sequence_length")
    out["mae_median"] = g["MAE"].median()
    out["mae_p95"] = g["MAE"].quantile(0.95)
    out["err_p05"] = g["signed_error"].quantile(0.05)
    out["err_p50"] = g["signed_error"].quantile(0.50)
    out["err_p95"] = g["signed_error"].quantile(0.95)
    out["half_width_90"] = 0.5 * (out["err_p95"] - out["err_p05"])
    out = out.reset_index()
    os.makedirs(FIGDIR, exist_ok=True)
    out.round(4).to_csv(f"{FIGDIR}/seq_length_intervals.csv", index=False)
    print("\n=== A8: sequence-length prediction intervals ===")
    print(out.round(4).to_string(index=False))
    return out


# Internal detector-file names (raw feature_name values as stored in the CSVs).
# Note: MSP is stored internally as 'softmax'; utils.DSDS uses 'msp' as the file token.
DSDS_INTERNAL = ["knn", "grad_magnitude", "cross_entropy", "energy", "typicality", "softmax"]


def run_reviewer_experiments(batch_size=1):
    """
    Run every review-response experiment and write all artifacts under figures/.

    Loads the raw per-sample scores and the per-method prediction frames ONCE and
    threads them through, so the ~660-file load_all() and the LOSO fits are not
    repeated across functions.
    """
    raw = load_all(batch_size=batch_size, shift="")
    ours = shift_type_loo_predictions(batch_size=batch_size)
    atc = atc_predictions(batch_size=batch_size)
    pre = pre_predictions()
    reg = regression_baseline_predictions(batch_size=batch_size, raw=raw)

    equal_resource_baseline_table(batch_size, ours=ours, atc=atc, reg=reg)
    stress_test_analysis(batch_size)
    detector_quality_vs_coupling(batch_size)  # needs dr_gap_signed_correlation.csv (Fig 2)
    per_architecture_table(batch_size, ours=ours, atc=atc, pre=pre)
    threshold_sensitivity(batch_size, raw=raw)
    feature_type_ablation(batch_size, raw=raw)
    regressor_ablation(batch_size)
    detector_count_ablation(batch_size)
    polyp_continuous_iou(batch_size, raw=raw)
    sequence_length_intervals()


if __name__ == "__main__":
    run_reviewer_experiments()
