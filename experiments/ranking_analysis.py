"""
Multi-benchmark ranking analysis (Demšar, JMLR 2006; Benavoli et al., JMLR 2016).

Two questions, each answered with the same machinery:

  A. Which OOD detector makes the best accuracy estimator?  The eight post-hoc
     detectors are ranked against each other, separately for the two aggregate
     statistics the paper uses (raw mean score, and detection rate at the 0.95
     in-distribution quantile).
  B. How does the estimator rank against the UAE baselines?  The estimator enters
     with its detector chosen by *nested* leave-one-shift-family-out selection
     (the held-out family is never used to pick the detector), next to ATC, AC,
     DoC, Nuclear Norm, the COT-style score and the ATC-soft-rate regression.

Protocol
--------
* Block (the "dataset" of Demšar's setup) = one (Dataset, Model) pair.  Each block
  gets one score per method: the mean MAE over its held-out folds, under the
  leave-one-shift-family-out protocol of ``shift_type_loo_predictions``.  This is
  computed for three scopes: all held-out folds, synthetic only, organic only.
* Within a block, methods are ranked by MAE (1 = best; ties get the average rank).
* Reported per method: average rank, median rank, number of wins (rank 1), and the
  full rank histogram (how often each method lands at each rank position).
* Omnibus: Friedman chi-square, its Iman-Davenport F correction, and Kendall's W
  (rank agreement across blocks; W near 0 means the ordering is inconsistent).
* Post-hoc: Nemenyi critical difference (CD diagram), and pairwise two-sided
  Wilcoxon signed-rank tests on the block scores with Holm correction, which
  Benavoli et al. recommend over mean-rank post-hoc tests.
* Polyp is excluded from the main analysis: segmentation has no MSP, Mahalanobis,
  ReAct or ViM, and none of the softmax-based baselines, so its blocks are
  incomplete.  ``detector_ranking(include_polyp=True)`` ranks the five detectors
  that do exist there.

Selection never uses held-out labels: the nested selection for held-out family h
runs an inner leave-one-family-out over the remaining calibration families.

Outputs go to figures/ranking_*.csv and figures/ranking_cd_*.pdf.
"""
import itertools
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import f as f_dist
from scipy.stats import friedmanchisquare, rankdata, studentized_range, wilcoxon
from sklearn.linear_model import LinearRegression

from experiments.accuracy_prediction import _atc_threshold, atc_predictions
from experiments.ablations import (
    _loso_from_perfold,
    _perfold_dr,
    _perfold_stat,
    _softmax_stat_predictions,
    label_free_baselines,
    mde_predictions,
)
from utils import DSD_PRINT_LUT, DSDS, FEATURE_COLUMN_ALIAS, load_data

FIGDIR = "figures"
CLASSIFICATION = ["CCT", "OfficeHome", "Office31", "NICO", "Camelyon17", "IWildCam"]
CLASSIFIERS = ["resnet", "vit"]
SEGMENTORS = ["deeplabv3plus", "unet", "segformer"]
AGGREGATES = {
    "mean": lambda f: float(f["feature"].mean()),
    "dr95": None,  # detection rate at the 0.95 InD quantile, via _perfold_dr
}
# Accuracy-estimation analyses evaluate on every held-out fold (all synthetic intensities +
# organic). The OOD-detection analysis (shift_detection.py) instead uses only organic folds and
# the maximum-intensity fold of each synthetic family; see restrict_targets / max_intensity_only.
SCOPES = ("All", "Synthetic", "Organic")
MAX_SCOPES = ("All", "SyntheticMax", "Organic")


def _fold_intensity(fold):
    try:
        return float(str(fold).rsplit("_", 1)[-1])
    except ValueError:
        return np.nan


def restrict_targets(rows):
    """Keep organic folds and, per (Dataset, synthetic family), only the max-intensity fold."""
    rows = rows.copy()
    rows["intensity"] = rows["fold"].map(_fold_intensity)
    syn = rows["category"] == "Synthetic"
    max_int = rows[syn].groupby(["Dataset", "shift_type"])["intensity"].transform("max")
    keep = ~syn
    keep.loc[syn] = rows.loc[syn, "intensity"] == max_int
    return rows[keep]


# ----------------------------------------------------------------------------
# Per-fold tables and per-detector LOSO predictions
# ----------------------------------------------------------------------------
def _perfold_tables(datasets, models, detectors=DSDS):
    """Return {(dataset, model, detector, aggregate): per-fold table}.

    Loads one (dataset, model, detector) feature set at a time to bound memory.
    """
    tables = {}
    for ds, m, det in itertools.product(datasets, models, detectors):
        raw = load_data(ds, det, batch_size=1, samples=1, model=m, shift="")
        if raw is None or raw.empty:
            continue
        col = FEATURE_COLUMN_ALIAS.get(det, det)
        idx = {(ds, m, col): raw[raw["feature_name"] == col]}
        for agg, fn in AGGREGATES.items():
            if fn is None:
                pf = _perfold_dr(idx, ds, m, col, 0.95)
            else:
                pf = _perfold_stat(idx, ds, m, col, fn)
            if pf is not None and not pf.empty:
                tables[(ds, m, det, agg)] = pf[["fold", "shift_type", "category", "stat", "gap"]]
        del raw, idx
    return tables


def _detector_rows(tables):
    """LOSO predictions for every (dataset, model, detector, aggregate)."""
    rows = []
    for (ds, m, det, agg), pf in tables.items():
        for r in _loso_from_perfold(pf, ds, m, DSD_PRINT_LUT.get(det, det), f"Ours-{agg}"):
            rows.append(r)
    return pd.DataFrame(rows)


def _inner_loso_mae(pf):
    """Mean MAE of leave-one-family-out over the synthetic families in ``pf``."""
    fams = pf["shift_type"].unique()
    errs = []
    for g in fams:
        tr, te = pf[pf["shift_type"] != g], pf[pf["shift_type"] == g]
        if len(tr) < 2 or te.empty:
            continue
        reg = LinearRegression().fit(tr[["stat"]].values, tr["gap"].values)
        errs.append(np.abs(reg.predict(te[["stat"]].values) - te["gap"].values))
    return float(np.concatenate(errs).mean()) if errs else np.inf


def _nested_selection_rows(tables, agg):
    """Estimator with the detector chosen by nested LOSO; held-out labels never used."""
    rows = []
    keys = sorted({(ds, m) for (ds, m, _d, a) in tables if a == agg})
    for ds, m in keys:
        cands = {d: pf for (ds2, m2, d, a), pf in tables.items() if (ds2, m2, a) == (ds, m, agg)}
        pools = {d: pf[(pf["category"] == "Synthetic") & (pf["shift_type"] != "ind")] for d, pf in cands.items()}
        families = sorted(set.intersection(*[set(p["shift_type"]) for p in pools.values()]))

        def emit(r, pred, det, st, cat):
            rows.append({
                "Dataset": ds, "Model": m, "feature_name": DSD_PRINT_LUT.get(det, det),
                "fold": r["fold"], "shift_type": st, "category": cat,
                "observed_gap": float(r["gap"]), "predicted_gap": float(pred),
                "MAE": abs(float(pred) - float(r["gap"])), "Method": f"Ours-{agg} (nested sel.)",
            })

        for h in families:
            inner = {d: _inner_loso_mae(p[p["shift_type"] != h]) for d, p in pools.items()}
            det = min(inner, key=inner.get)
            p = pools[det]
            tr, te = p[p["shift_type"] != h], p[p["shift_type"] == h]
            reg = LinearRegression().fit(tr[["stat"]].values, tr["gap"].values)
            for (_, r), pred in zip(te.iterrows(), reg.predict(te[["stat"]].values)):
                emit(r, pred, det, h, "Synthetic")

        inner = {d: _inner_loso_mae(p) for d, p in pools.items()}
        det = min(inner, key=inner.get)
        p, pf = pools[det], cands[det]
        reg = LinearRegression().fit(p[["stat"]].values, p["gap"].values)
        org = pf[(pf["category"] == "Organic") & (~pf["fold"].isin(["train", "ind_val"]))]
        for _, r in org.iterrows():
            emit(r, reg.predict([[r["stat"]]])[0], det, "Organic", "Organic")
    return pd.DataFrame(rows)


def _reg_atc_rows(datasets, models):
    """ATC soft error rate regressed on the same calibration shifts (equal-resource ATC)."""
    rows = []
    for ds, m in itertools.product(datasets, models):
        raw = load_data(ds, "msp", batch_size=1, samples=1, model=m, shift="")
        if raw is None or raw.empty:
            continue
        iv = raw[raw["fold"] == "ind_val"]
        tau = _atc_threshold(iv["feature"].values, iv["correct_prediction"].values)
        if np.isnan(tau):
            continue
        idx = {(ds, m, "softmax"): raw}
        pf = _perfold_stat(idx, ds, m, "softmax", lambda f, t=tau: float((f["feature"].values < t).mean()))
        if pf is not None:
            rows += _loso_from_perfold(pf, ds, m, "softmax", "Reg-ATC")
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Generic ranking statistics
# ----------------------------------------------------------------------------
def block_scores(rows, scope="All", block=("Dataset", "Model"), max_intensity_only=False):
    """Blocks x methods matrix of mean MAE; only complete blocks are kept.

    ``max_intensity_only`` keeps organic folds and only the max-intensity fold of each
    synthetic family (used for the OOD-detection AUROC analysis).
    """
    df = rows[~rows["fold"].isin(["train", "ind_val", "ind_test"]) & (rows["shift_type"] != "ind")]
    if max_intensity_only:
        df = restrict_targets(df)
    if scope in ("Synthetic", "SyntheticMax"):
        df = df[df["category"] == "Synthetic"]
    elif scope != "All":
        df = df[df["category"] == scope]
    scores = df.pivot_table(index=list(block), columns="Method", values="MAE", aggfunc="mean")
    # Methods with no data at all (e.g. Entropy / ATC-NE before the GPU recollection) are
    # dropped as columns rather than emptying every block.
    scores = scores.dropna(axis=1, how="all")
    dropped = scores[scores.isna().any(axis=1)].index.tolist()
    if dropped:
        print(f"[block_scores:{scope}] dropping incomplete blocks: {dropped}")
    return scores.dropna()


def rank_matrix(scores):
    """Rank methods within each block (1 = lowest MAE, ties averaged)."""
    return scores.apply(lambda r: pd.Series(rankdata(r.values, method="average"), index=r.index), axis=1)


def rank_summary(ranks):
    k = ranks.shape[1]
    hist = pd.DataFrame(
        {m: [(np.floor(ranks[m]) == p).sum() for p in range(1, k + 1)] for m in ranks.columns},
        index=[f"rank_{p}" for p in range(1, k + 1)],
    ).T
    out = pd.DataFrame({
        "avg_rank": ranks.mean(),
        "median_rank": ranks.median(),
        "std_rank": ranks.std(ddof=1),
        "wins": (ranks == 1).sum(),
        "top3": (ranks <= 3).sum(),
    }).join(hist)
    return out.sort_values("avg_rank")


def friedman_tests(scores):
    """Friedman chi2, Iman-Davenport F, and Kendall's W."""
    N, k = scores.shape
    chi2, p = friedmanchisquare(*[scores[c].values for c in scores.columns])
    ff = (N - 1) * chi2 / (N * (k - 1) - chi2) if N * (k - 1) != chi2 else np.inf
    p_ff = float(f_dist.sf(ff, k - 1, (k - 1) * (N - 1)))
    return {
        "n_blocks": N, "n_methods": k,
        "friedman_chi2": float(chi2), "friedman_p": float(p),
        "iman_davenport_F": float(ff), "iman_davenport_p": p_ff,
        "kendalls_W": float(chi2 / (N * (k - 1))),
    }


def nemenyi_cd(k, N, alpha=0.05):
    q = studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)
    return float(q * np.sqrt(k * (k + 1) / (6.0 * N)))


def wilcoxon_holm(scores, alpha=0.05):
    """Pairwise two-sided Wilcoxon signed-rank on block scores, Holm-corrected."""
    rows = []
    for a, b in itertools.combinations(scores.columns, 2):
        x, y = scores[a].values, scores[b].values
        d = x - y
        p = 1.0 if np.all(d == 0) else float(wilcoxon(x, y, zero_method="wilcox").pvalue)
        rows.append({"method_1": a, "method_2": b, "n_blocks": len(d),
                     "mean_mae_1": x.mean(), "mean_mae_2": y.mean(),
                     "median_diff": float(np.median(d)), "wins_1": int((d < 0).sum()),
                     "wins_2": int((d > 0).sum()), "p_value": p})
    out = pd.DataFrame(rows).sort_values("p_value").reset_index(drop=True)
    m = len(out)
    adj = np.maximum.accumulate([min(1.0, (m - i) * p) for i, p in enumerate(out["p_value"])])
    out["p_holm"] = adj
    out["significant"] = out["p_holm"] < alpha
    return out


def cd_diagram(avg_ranks, cd, pairwise, title, path):
    """Critical-difference diagram. Bars join methods that Wilcoxon-Holm cannot separate."""
    avg_ranks = avg_ranks.sort_values()
    names, vals = list(avg_ranks.index), avg_ranks.values
    k = len(names)
    sig = {frozenset((r.method_1, r.method_2)) for r in pairwise.itertuples() if r.significant}
    # maximal cliques of consecutive methods with no significant pair inside
    cliques = []
    for i in range(k):
        j = i
        while j + 1 < k and all(frozenset((names[a], names[j + 1])) not in sig for a in range(i, j + 1)):
            j += 1
        if j > i and not any(c[0] <= i and c[1] >= j for c in cliques):
            cliques.append((i, j))

    half = (k + 1) // 2
    fig, ax = plt.subplots(figsize=(7.5, 1.0 + 0.22 * half))
    lo, hi = 1, k
    ax.set_xlim(lo - 0.5, hi + 0.5)
    ax.set_ylim(-half * 0.45 - 0.3, 2.3)
    ax.axis("off")
    ax.hlines(1, lo, hi, color="black", lw=1)
    for t in range(lo, hi + 1):
        ax.vlines(t, 1, 1.15, color="black", lw=1)
        ax.text(t, 1.35, str(t), ha="center", va="bottom", fontsize=8)
    ax.hlines(1.9, lo, lo + cd, color="black", lw=1.5)
    ax.text(lo + cd / 2, 1.95, f"CD (Nemenyi) = {cd:.2f}", ha="center", va="bottom", fontsize=8)
    for i, (n, v) in enumerate(zip(names, vals)):
        left = i < half
        y = -(i if left else (k - 1 - i)) * 0.45 - 0.3
        xt = lo - 0.3 if left else hi + 0.3
        ax.plot([v, v, xt], [1, y, y], color="0.3", lw=0.8)
        ax.text(xt + (-0.05 if left else 0.05), y, f"{n} ({v:.2f})", ha="right" if left else "left",
                va="center", fontsize=8)
    for c, (i, j) in enumerate(cliques):
        y = 0.75 - 0.12 * c
        ax.hlines(y, vals[i] - 0.03, vals[j] + 0.03, color="black", lw=3)
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


FINE_BLOCK = ("Dataset", "Model", "shift_type")


def analyse(rows, label, scopes=SCOPES, alpha=0.05, block=("Dataset", "Model"), max_intensity_only=False):
    """Run the full ranking analysis on a long-format row table; write CSVs and CD diagrams.

    ``block=FINE_BLOCK`` treats each (dataset, model, shift family) as a block — the unit
    used by accuracy_prediction.method_statistical_tests. It gives the Wilcoxon tests usable
    power but the blocks within one dataset are not independent, so report it as a
    sensitivity analysis next to the (Dataset, Model) design.
    """
    os.makedirs(FIGDIR, exist_ok=True)
    if tuple(block) == FINE_BLOCK:
        label = f"{label}_fine"
    summaries, omnibus = [], []
    for scope in scopes:
        scores = block_scores(rows, scope, block, max_intensity_only)
        if scores.shape[0] < 2 or scores.shape[1] < 3:
            print(f"[{label}/{scope}] too few complete blocks ({scores.shape[0]})")
            continue
        ranks = rank_matrix(scores)
        summ = rank_summary(ranks)
        summ.insert(0, "mean_MAE", scores.mean()[summ.index])
        fr = friedman_tests(scores)
        cd = nemenyi_cd(scores.shape[1], scores.shape[0], alpha)
        fr.update({"analysis": label, "scope": scope, "nemenyi_CD": cd})
        pw = wilcoxon_holm(scores, alpha)

        tag = f"{label}_{scope.lower()}"
        scores.round(4).to_csv(f"{FIGDIR}/ranking_{tag}_block_scores.csv")
        ranks.to_csv(f"{FIGDIR}/ranking_{tag}_ranks.csv")
        pw.to_csv(f"{FIGDIR}/ranking_{tag}_wilcoxon_holm.csv", index=False)
        cd_diagram(summ["avg_rank"], cd, pw,
                   f"{label} ({scope}; {scores.shape[0]} blocks, Friedman p={fr['friedman_p']:.2g}, "
                   f"W={fr['kendalls_W']:.2f})",
                   f"{FIGDIR}/ranking_cd_{tag}.pdf")
        summ.insert(0, "scope", scope)
        summaries.append(summ)
        omnibus.append(fr)

        print(f"\n=== {label} | {scope} | blocks={scores.shape[0]} methods={scores.shape[1]} ===")
        print(summ.drop(columns=["scope"]).round(3).to_string())
        print({k: round(v, 4) if isinstance(v, float) else v for k, v in fr.items()})
        print(pw[pw["significant"]][["method_1", "method_2", "median_diff", "p_holm"]].round(4).to_string())

    if summaries:
        pd.concat(summaries).round(4).to_csv(f"{FIGDIR}/ranking_{label}_summary.csv")
        pd.DataFrame(omnibus).to_csv(f"{FIGDIR}/ranking_{label}_omnibus.csv", index=False)
    return summaries, omnibus


# ----------------------------------------------------------------------------
# Entry points
# ----------------------------------------------------------------------------
def detector_ranking(tables=None, include_polyp=False):
    """A. Rank the detectors against each other, per aggregate statistic."""
    if tables is None:
        tables = _perfold_tables(CLASSIFICATION, CLASSIFIERS)
    det_rows = _detector_rows(tables)
    for agg in AGGREGATES:
        sub = det_rows[det_rows["Method"] == f"Ours-{agg}"].copy()
        sub["Method"] = sub["feature_name"]
        analyse(sub, f"detectors_{agg}")
        analyse(sub, f"detectors_{agg}", scopes=("All",), block=FINE_BLOCK)
    if include_polyp:
        pt = _perfold_tables(["Polyp"], SEGMENTORS)
        prow = _detector_rows(pt)
        for agg in AGGREGATES:
            sub = prow[prow["Method"] == f"Ours-{agg}"].copy()
            sub["Method"] = sub["feature_name"]
            analyse(sub, f"detectors_{agg}_polyp")
    return det_rows


def method_ranking(tables=None):
    """B. Rank the estimator (nested detector selection) against the UAE baselines."""
    if tables is None:
        tables = _perfold_tables(CLASSIFICATION, CLASSIFIERS)
    frames = [_nested_selection_rows(tables, agg) for agg in AGGREGATES]
    frames.append(atc_predictions(1))
    frames.append(label_free_baselines(1, datasets=CLASSIFICATION))
    frames.append(_reg_atc_rows(CLASSIFICATION, CLASSIFIERS))
    frames.append(mde_predictions(1, CLASSIFICATION))
    nn_path = f"{FIGDIR}/softmax_stats.csv"
    if os.path.exists(nn_path):
        # NuclearNorm, LabelShiftL1 (the old "COT"), and real COT / COT (native) once
        # collect_softmax_stats.py has been rerun on the GPU machine.
        frames.append(_softmax_stat_predictions(nn_path))
    rows = pd.concat([f for f in frames if f is not None and not f.empty], ignore_index=True)
    rows = rows[rows["Dataset"].isin(CLASSIFICATION)]
    analyse(rows, "methods")
    analyse(rows, "methods", scopes=("All",), block=FINE_BLOCK)
    return rows


def run_ranking_analysis(include_polyp=True):
    tables = _perfold_tables(CLASSIFICATION, CLASSIFIERS)
    det_rows = detector_ranking(tables, include_polyp=include_polyp)
    method_rows = method_ranking(tables)
    det_rows.to_csv(f"{FIGDIR}/ranking_detector_rows.csv", index=False)
    method_rows.to_csv(f"{FIGDIR}/ranking_method_rows.csv", index=False)
    return det_rows, method_rows


if __name__ == "__main__":
    matplotlib.use("Agg")
    run_ranking_analysis()
