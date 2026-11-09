"""
Calibration of the accuracy estimators: do predicted accuracy gaps agree with observed ones?

Uses the per-fold rows written by ranking_analysis (figures/ranking_method_rows.csv) and the
same evaluation targets: every held-out fold (all synthetic intensities + organic).

Per method (pooled) and per (Dataset, Model) block:
  * bias        : mean(predicted_gap - observed_gap); > 0 overstates degradation
  * slope       : OLS slope of predicted_gap on observed_gap (1 = ideal; < 1 = large gaps
                  compressed toward zero, i.e. the monitor understates large drops)
  * intercept   : of the same fit
  * R2          : of the same fit
  * binned_CE   : ECE-style calibration error. Folds are binned by observed gap (width 0.1);
                  sum over bins of (bin share) * |mean predicted - mean observed|
  * missed_rate : among folds with observed gap > LARGE_GAP, the share whose gap is
                  underestimated by more than MISS_MARGIN (silent failures for a monitor)

The per-block |slope - 1|, |bias| and binned_CE go through the ranking machinery
(Friedman / Kendall's W / Wilcoxon-Holm / CD diagrams).
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.ranking_analysis import analyse

FIGDIR = "figures"
LARGE_GAP = 0.2
MISS_MARGIN = 0.1
BIN = 0.1


def _calib_metrics(g):
    x = g["observed_gap"].to_numpy(float)
    y = g["predicted_gap"].to_numpy(float)
    res = y - x
    if len(x) > 2 and np.var(x) > 0:
        slope, intercept = np.polyfit(x, y, 1)
        r2 = 1 - np.sum((y - (slope * x + intercept)) ** 2) / np.sum((y - y.mean()) ** 2)
    else:
        slope = intercept = r2 = np.nan
    bins = np.floor(x / BIN)
    ce = sum((bins == b).mean() * abs(y[bins == b].mean() - x[bins == b].mean()) for b in np.unique(bins))
    large = x > LARGE_GAP
    missed = float(((y - x)[large] < -MISS_MARGIN).mean()) if large.any() else np.nan
    return pd.Series({"n": len(x), "MAE": np.abs(res).mean(), "bias": res.mean(), "slope": slope,
                      "intercept": intercept, "R2": r2, "binned_CE": ce,
                      "missed_rate": missed, "n_large": int(large.sum())})


def calibration_tables(rows):
    rows = rows[~rows["fold"].isin(["train", "ind_val", "ind_test"])]
    rows = rows.dropna(subset=["observed_gap", "predicted_gap"])
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
    lim = (-0.15, 1.0)
    for ax, m in zip(axes, methods):
        g = rows[rows["Method"] == m]
        for cat, mk, col in [("Synthetic", "o", "0.55"), ("Organic", "^", "C3")]:
            gg = g[g["category"] == cat]
            ax.scatter(gg["observed_gap"], gg["predicted_gap"], s=9, marker=mk, color=col, alpha=0.7,
                       label="synthetic" if cat == "Synthetic" else "natural")
        ax.plot(lim, lim, "k--", lw=0.8)
        met = _calib_metrics(g)
        ax.set_title(f"{m}\nslope {met['slope']:.2f}, bias {met['bias']:+.3f}", fontsize=8)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
    for ax in axes[k:]:
        ax.axis("off")
    axes[0].legend(fontsize=6, frameon=False, loc="upper left")
    fig.supxlabel("observed accuracy gap", fontsize=9)
    fig.supylabel("predicted accuracy gap", fontsize=9)
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
