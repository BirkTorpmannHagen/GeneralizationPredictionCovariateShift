"""
Calibration of the accuracy estimators: do predicted accuracies agree with observed ones?

Uses the per-fold rows written by ranking_analysis (figures/ranking_method_rows.csv) and the
same evaluation targets: every held-out fold (all synthetic intensities + organic).

Estimators are fitted on the generalization gap internally (gap = InD-val accuracy - fold
accuracy); with a fixed InD accuracy this is the same linear model as regressing accuracy
directly, and the absolute errors are identical. Everything reported here is in accuracy space:
observed_acc is the fold's accuracy and predicted_acc = observed_acc - (predicted_gap -
observed_gap), which keeps every method's signed error exactly.

Per method (pooled) and per (Dataset, Model) block:
  * bias        : mean(predicted_acc - observed_acc); > 0 overstates accuracy
  * slope       : OLS slope of predicted_acc on observed_acc (1 = ideal; < 1 = large accuracy
                  drops compressed toward InD accuracy, i.e. the monitor understates them)
  * intercept   : of the same fit
  * R2          : of the same fit
  * binned_CE   : ECE-style calibration error. Folds are binned by observed accuracy (width
                  0.1); sum over bins of (bin share) * |mean predicted - mean observed|
  * missed_rate : among folds whose accuracy dropped by more than LARGE_GAP below InD, the share
                  whose accuracy is overestimated by more than MISS_MARGIN (silent failures)

The per-block |slope - 1|, |bias| and binned_CE go through the ranking machinery
(Friedman / Kendall's W / Wilcoxon-Holm / CD diagrams).
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.ranking_analysis import CLASSIFICATION, CLASSIFIERS, SEGMENTORS, analyse
from utils import load_data

FIGDIR = "figures"
LARGE_GAP = 0.2
MISS_MARGIN = 0.1
BIN = 0.1
FOLD_ACC = f"{FIGDIR}/fold_accuracy.csv"


def fold_accuracy(refresh=False):
    """Observed accuracy of every (Dataset, Model, fold), from the cached per-sample data."""
    if os.path.exists(FOLD_ACC) and not refresh:
        return pd.read_csv(FOLD_ACC)
    recs = []
    for datasets, models in [(CLASSIFICATION, CLASSIFIERS), (["Polyp"], SEGMENTORS)]:
        for ds in datasets:
            for m in models:
                df = load_data(ds, "energy", batch_size=1, samples=1, model=m, shift="")
                if df.empty:
                    continue
                df = df[df["feature_name"] == "energy"]
                a = df.groupby("fold")["correct_prediction"].mean()
                recs += [{"Dataset": ds, "Model": m, "fold": f, "observed_acc": float(v)} for f, v in a.items()]
    out = pd.DataFrame(recs)
    out.to_csv(FOLD_ACC, index=False)
    return out


SOFTMAX_METHODS = ("NuclearNorm", "LabelShiftL1", "COT", "COT (native)")


def _ind_anchor():
    """InD-val accuracy per (Dataset, Model, source). Every estimator's gap is defined against
    the InD-val accuracy of its own data source; InD-val is identical across detector files,
    but stochastic shifts (e.g. fog) differ between collections, so shifted-fold accuracy must
    come from each row's own gap rather than from a shared per-fold table."""
    fa = fold_accuracy()
    a = fa[fa["fold"] == "ind_val"][["Dataset", "Model", "observed_acc"]].rename(columns={"observed_acc": "ind_acc"})
    a["source"] = "features"
    frames = [a]
    sm = f"{FIGDIR}/softmax_stats.csv"
    if os.path.exists(sm):
        s = pd.read_csv(sm)
        s = s[s["fold"] == "ind_val"].groupby(["Dataset", "Model"], as_index=False)["true_acc"].mean()
        frames.append(s.rename(columns={"true_acc": "ind_acc"}).assign(source="softmax"))
    return pd.concat(frames, ignore_index=True)


def with_accuracy(rows):
    """Add observed_acc / predicted_acc to per-fold estimator rows (signed errors preserved).

    Rows that already carry them (PRE) keep theirs; all others use
    observed_acc = InD-val accuracy of their source - observed_gap."""
    rows = rows.copy()
    if "observed_acc" not in rows.columns:
        rows["observed_acc"] = np.nan
    need = rows["observed_acc"].isna()
    rows["source"] = np.where(rows["Method"].isin(SOFTMAX_METHODS), "softmax", "features")
    anchored = rows.merge(_ind_anchor(), on=["Dataset", "Model", "source"], how="left")
    anchored.index = rows.index
    rows.loc[need, "observed_acc"] = (anchored["ind_acc"] - anchored["observed_gap"])[need]
    rows["predicted_acc"] = rows["observed_acc"] - (rows["predicted_gap"] - rows["observed_gap"])
    missing = rows["observed_acc"].isna() & rows["observed_gap"].notna()
    if missing.any():
        print(f"[with_accuracy] {int(missing.sum())} rows without an InD accuracy (dropped)")
        rows = rows[~missing]
    return rows.drop(columns="source")


def _calib_metrics(g):
    if "observed_acc" not in g.columns or g["observed_acc"].isna().any() or "predicted_acc" not in g.columns:
        g = with_accuracy(g)
    x = g["observed_acc"].to_numpy(float)
    y = g["predicted_acc"].to_numpy(float)
    drop = g["observed_gap"].to_numpy(float)
    res = y - x
    if len(x) > 2 and np.var(x) > 0:
        slope, intercept = np.polyfit(x, y, 1)
        r2 = 1 - np.sum((y - (slope * x + intercept)) ** 2) / np.sum((y - y.mean()) ** 2)
    else:
        slope = intercept = r2 = np.nan
    bins = np.floor(x / BIN)
    ce = sum((bins == b).mean() * abs(y[bins == b].mean() - x[bins == b].mean()) for b in np.unique(bins))
    large = drop > LARGE_GAP
    missed = float(((y - x)[large] > MISS_MARGIN).mean()) if large.any() else np.nan
    return pd.Series({"n": len(x), "MAE": np.abs(res).mean(), "bias": res.mean(), "slope": slope,
                      "intercept": intercept, "R2": r2, "binned_CE": ce,
                      "missed_rate": missed, "n_large": int(large.sum())})


def calibration_tables(rows):
    rows = rows[~rows["fold"].isin(["train", "ind_val", "ind_test"])]
    rows = with_accuracy(rows.dropna(subset=["observed_gap", "predicted_gap"]))
    pooled = {}
    for scope, sub in [("All", rows), ("Synthetic", rows[rows["category"] == "Synthetic"]),
                       ("Organic", rows[rows["category"] == "Organic"])]:
        pooled[scope] = sub.groupby("Method").apply(_calib_metrics)
    block = rows.groupby(["Dataset", "Model", "Method"]).apply(_calib_metrics).reset_index()
    return rows, pooled, block


def reliability_plot(rows, path, methods=None):
    methods = methods or sorted(rows["Method"].unique())
    k = len(methods)
    ncol = min(5, k)
    nrow = int(np.ceil(k / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.6 * ncol, 2.6 * nrow), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    lim = (0.0, 1.05)
    for ax, m in zip(axes, methods):
        g = rows[rows["Method"] == m]
        for cat, mk, col in [("Synthetic", "o", "0.55"), ("Organic", "^", "C3")]:
            gg = g[g["category"] == cat]
            ax.scatter(gg["observed_acc"], gg["predicted_acc"], s=9, marker=mk, color=col, alpha=0.7,
                       label="synthetic" if cat == "Synthetic" else "natural")
        ax.plot(lim, lim, "k--", lw=0.8)
        met = _calib_metrics(g)
        ax.set_title(f"{m}\nslope {met['slope']:.2f}, bias {met['bias']:+.3f}", fontsize=8)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
    for ax in axes[k:]:
        ax.axis("off")
    axes[0].legend(fontsize=6, frameon=False, loc="upper left")
    fig.supxlabel("observed accuracy", fontsize=9)
    fig.supylabel("predicted accuracy", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def run_calibration_analysis(rows_path=f"{FIGDIR}/ranking_method_rows.csv"):
    rows = pd.read_csv(rows_path)
    rows, pooled, block = calibration_tables(rows)
    for scope, t in pooled.items():
        t.round(4).to_csv(f"{FIGDIR}/calibration_{scope.lower()}_pooled.csv")
        print(f"\n=== Calibration, pooled ({scope}) ===")
        print(t.sort_values("binned_CE").round(3).to_string())
    block.round(4).to_csv(f"{FIGDIR}/calibration_per_block.csv", index=False)

    # Rank methods per block on calibration criteria (lower is better).
    for metric, fn in [("slope_dev", lambda b: (b["slope"] - 1).abs()),
                       ("abs_bias", lambda b: b["bias"].abs()),
                       ("binned_CE", lambda b: b["binned_CE"])]:
        r = block[["Dataset", "Model", "Method"]].copy()
        r["MAE"] = fn(block)
        r["fold"], r["shift_type"], r["category"] = "block", "block", "Organic"
        analyse(r, f"calib_{metric}", scopes=("All",))
    reliability_plot(rows, f"{FIGDIR}/calibration_reliability.pdf")
    return pooled, block


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    run_calibration_analysis()
