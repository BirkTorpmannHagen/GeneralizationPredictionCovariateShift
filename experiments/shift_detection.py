"""
UAE methods scored as OOD (shift) detectors — the reverse direction of the bridge.

Per-sample view: the per-sample scores used by UAE methods are already OOD detectors in
our set (AC / DoC / ATC-MC read max-softmax = the MSP detector; ATC-NE reads predictive
entropy = the Entropy detector), so their per-sample OOD quality is the detector's.

Set-level view (this module): batch statistics such as Nuclear Norm, COT and MDE have no
per-sample score, so we score every method on the task it can actually do: given an
unlabeled batch of SET_SIZE inputs, is it shifted?  For each (dataset, model):

  * negatives: N_NEG random batches from ind_test (in-distribution, held out)
  * positives: N_POS random batches from each shifted fold (synthetic or organic)
  * AUROC of the batch statistic, per shifted fold

Orientation (does a larger statistic mean "shifted"?) is fixed without target data: it is
the sign of (mean over synthetic calibration batches - mean over ind_val batches).

Methods: every detector's batch-mean score (= the estimator's raw-mean aggregate; for MSP
this is also AC, and DoC is a constant shift of it, so they share one entry), MDE, the ATC
soft rate, and Nuclear Norm / COT from figures/softmax_stats_subsets.csv when it exists
(produced on the GPU machine by collect_softmax_stats.py with the same batch size).

The detection AUROCs then go through the ranking machinery of ranking_analysis (as
1 - AUROC, so lower is better), and are correlated with each method's UAE error to test
whether shift-detection quality and accuracy-estimation quality go together.
"""
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from experiments.accuracy_prediction import _atc_threshold, _is_synthetic_shift_type, _parse_shift_type_from_fold
from experiments.ranking_analysis import CLASSIFICATION, CLASSIFIERS, MAX_SCOPES, analyse, restrict_targets
from utils import DSD_PRINT_LUT, DSDS, FEATURE_COLUMN_ALIAS, load_data

FIGDIR = "figures"
SET_SIZE = 100
N_NEG = 200
N_POS = 50
SEED = 0


def _intensity(fold):
    try:
        return float(fold.split("_")[-1])
    except ValueError:
        return np.nan


def _batch_stat(values, idx, kind):
    v = values[idx]  # (B, n)
    if kind == "mean":
        return v.mean(1)
    if kind == "mde":  # Peng et al. Eq. 5 on z = -energy, size-normalized
        z = -v
        m = z.max(1, keepdims=True)
        lse = (m + np.log(np.exp(z - m).sum(1, keepdims=True)))[:, 0]
        return lse - np.log(v.shape[1]) - z.mean(1)
    raise ValueError(kind)


def _batches(rng, n_items, n_sets):
    n = min(SET_SIZE, n_items)
    return np.stack([rng.choice(n_items, size=n, replace=False) for _ in range(n_sets)])


def _method_batches(df, rng, kind, transform=None):
    """{fold: array of batch statistics} for one per-sample score table."""
    out = {}
    for fold, g in df.groupby("fold"):
        if fold == "train":
            continue
        vals = g["feature"].to_numpy(dtype=float)
        if transform is not None:
            vals = transform(vals)
        n_sets = N_NEG if fold == "ind_test" else N_POS
        out[fold] = _batch_stat(vals, _batches(rng, len(vals), n_sets), kind)
    return out


def _score_folds(ds, model, method, batches):
    """AUROC per shifted fold, with orientation fixed from ind_val vs synthetic batches."""
    if "ind_test" not in batches or "ind_val" not in batches:
        return []
    synth = [f for f in batches if _is_synthetic_shift_type(_parse_shift_type_from_fold(f))]
    if not synth:
        return []
    sign = np.sign(np.mean(np.concatenate([batches[f] for f in synth])) - np.mean(batches["ind_val"])) or 1.0
    neg = sign * batches["ind_test"]
    rows = []
    for fold, b in batches.items():
        if fold in ("ind_val", "ind_test"):
            continue
        pos = sign * b
        y = np.r_[np.zeros(len(neg)), np.ones(len(pos))]
        auc = roc_auc_score(y, np.r_[neg, pos])
        st = _parse_shift_type_from_fold(fold)
        rows.append({"Dataset": ds, "Model": model, "Method": method, "feature_name": method,
                     "fold": fold, "shift_type": st,
                     "category": "Synthetic" if _is_synthetic_shift_type(st) else "Organic",
                     "intensity": _intensity(fold), "AUROC": auc, "MAE": 1.0 - auc})
    return rows


def detection_rows(datasets=CLASSIFICATION, models=CLASSIFIERS, softmax_subsets=f"{FIGDIR}/softmax_stats_subsets.csv"):
    rng = np.random.default_rng(SEED)
    rows = []
    for ds in datasets:
        for model in models:
            for det in DSDS:
                df = load_data(ds, det, batch_size=1, samples=1, model=model, shift="")
                if df.empty:
                    continue
                df = df[df["feature_name"] == FEATURE_COLUMN_ALIAS.get(det, det)]
                name = DSD_PRINT_LUT.get(det, det)
                if det == "msp":
                    name = "MSP (= AC, DoC)"
                rows += _score_folds(ds, model, name, _method_batches(df, rng, "mean"))
                if det == "energy":
                    rows += _score_folds(ds, model, "MDE", _method_batches(df, rng, "mde"))
                if det == "msp":
                    iv = df[df["fold"] == "ind_val"]
                    tau = _atc_threshold(iv["feature"].values, iv["correct_prediction"].values)
                    rows += _score_folds(ds, model, "ATC (soft rate)",
                                         _method_batches(df, rng, "mean", lambda v, t=tau: (v < t).astype(float)))
    if os.path.exists(softmax_subsets):
        s = pd.read_csv(softmax_subsets)
        for (ds, model), g in s.groupby(["Dataset", "Model"]):
            for method, col in [("NuclearNorm", "nuclear_norm"), ("COT", "cot_err")]:
                if col in g.columns:
                    b = {f: gg[col].to_numpy() for f, gg in g.groupby("fold")}
                    rows += _score_folds(ds, model, method, b)
    else:
        print(f"[shift_detection] {softmax_subsets} not found: NuclearNorm / COT not scored yet "
              f"(run collect_softmax_stats.py on the GPU machine)")
    return pd.DataFrame(rows)


def uae_vs_detection(det_rows, uae_rows):
    """Does a method's shift-detection quality predict its accuracy-estimation error?

    Joins per (Dataset, Model, method) the mean detection AUROC — over organic folds and the
    max-intensity fold of each synthetic family — with the mean UAE MAE of the same statistic
    in the accuracy-estimation setting, i.e. the LOSO protocol over all held-out folds and
    intensities.
    """
    d = restrict_targets(det_rows).groupby(["Dataset", "Model", "Method"], as_index=False)["AUROC"].mean()
    u = uae_rows.groupby(["Dataset", "Model", "Method"], as_index=False)["MAE"].mean()
    j = d.merge(u, on=["Dataset", "Model", "Method"])
    rho, p = spearmanr(j["AUROC"], j["MAE"]) if len(j) > 2 else (np.nan, np.nan)
    within = []
    for (ds, m), g in j.groupby(["Dataset", "Model"]):
        if len(g) > 2:
            within.append(spearmanr(g["AUROC"], g["MAE"])[0])
    return j, {"pooled_spearman": float(rho), "pooled_p": float(p), "n": len(j),
               "median_within_block_spearman": float(np.nanmedian(within)) if within else np.nan,
               "n_blocks": len(within)}


def _uae_rows_matching(det_methods):
    """UAE LOSO rows for the same statistics that were scored as detectors."""
    from experiments.ranking_analysis import _detector_rows, _perfold_tables, _reg_atc_rows
    from experiments.ablations import _softmax_stat_predictions, mde_predictions
    tables = _perfold_tables(CLASSIFICATION, CLASSIFIERS)
    r = _detector_rows(tables)
    r = r[r["Method"] == "Ours-mean"].copy()
    r["Method"] = r["feature_name"].replace({"MSP": "MSP (= AC, DoC)"})
    frames = [r, mde_predictions(1, CLASSIFICATION)]
    atc = _reg_atc_rows(CLASSIFICATION, CLASSIFIERS)
    if not atc.empty:
        atc["Method"] = "ATC (soft rate)"
        frames.append(atc)
    if os.path.exists(f"{FIGDIR}/softmax_stats.csv"):
        frames.append(_softmax_stat_predictions(f"{FIGDIR}/softmax_stats.csv"))
    u = pd.concat(frames, ignore_index=True)
    u = u[~u["fold"].isin(["train", "ind_val", "ind_test"])]
    return u[u["Method"].isin(det_methods)]


def run_shift_detection():
    os.makedirs(FIGDIR, exist_ok=True)
    rows = detection_rows()
    rows.to_csv(f"{FIGDIR}/shift_detection_rows.csv", index=False)

    targets = restrict_targets(rows)  # organic + max-intensity synthetic only
    summary = pd.concat([
        targets[targets["category"] == "Organic"].groupby("Method")["AUROC"].mean().rename("AUROC_organic"),
        targets[targets["category"] == "Synthetic"].groupby("Method")["AUROC"].mean().rename("AUROC_synthetic_max"),
        targets.groupby("Method")["AUROC"].mean().rename("AUROC_all"),
    ], axis=1).sort_values("AUROC_all", ascending=False)
    summary.round(4).to_csv(f"{FIGDIR}/shift_detection_summary.csv")
    print("\n=== Set-level shift detection AUROC (batch size %d) ===" % SET_SIZE)
    print(summary.round(3).to_string())

    # Ranking machinery on 1 - AUROC (lower = better).
    analyse(rows, "shiftdet", scopes=MAX_SCOPES, max_intensity_only=True)

    joined, corr = uae_vs_detection(rows, _uae_rows_matching(set(rows["Method"])))
    joined.to_csv(f"{FIGDIR}/shift_detection_vs_uae.csv", index=False)
    print("\n=== Shift-detection AUROC vs UAE MAE ===")
    print(corr)
    pd.DataFrame([corr]).to_csv(f"{FIGDIR}/shift_detection_vs_uae_corr.csv", index=False)
    return rows, summary, joined, corr


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    run_shift_detection()
