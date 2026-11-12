"""
Paper tables (booktabs) and the detection-vs-UAE figure, generated from the analysis outputs.

Run after ranking_analysis, calibration_analysis and shift_detection have written their CSVs:

    python -m experiments.paper_artifacts

Writes:
    paper_drafting/tables/tab_uae_ranking.tex          accuracy-estimation ranks (all intensities)
    paper_drafting/tables/tab_detection_vs_uae.tex     per-statistic detection AUROC and UAE MAE
    paper_drafting/tables/tab_detection_uae_corr.tex   AUROC-vs-MAE association statistics
    paper_drafting/tables/tab_calibration.tex          calibration, all intensities and natural
    paper_drafting/figures/detection_vs_uae.pdf        scatter + per-block correlations

Detection AUROC: organic folds + the max-intensity fold of each synthetic family.
UAE MAE: the accuracy-estimation setting, i.e. every held-out fold and intensity.
"""
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr, wilcoxon

from experiments.ranking_analysis import restrict_targets

FIG = "figures"
OUT_TAB = "OODDetectorsAreUAE/tables"
OUT_FIG = "OODDetectorsAreUAE/figures"

METHOD_TEX = {
    "Ours-dr95 (nested sel.)": "Ours (DR)",
    "Ours-mean (nested sel.)": "Ours (mean score)",
    "NuclearNorm": "Nuclear Norm",
    "LabelShiftL1": "Label-marginal $L_1$",
    "COT": "COT",
    "COT (native)": "COT (native)",
    "MaNo": "MaNo",
    "Dispersion": "Dispersion",
    "Reg-ATC": "ATC (regressed)",
    "ATC-MC": "ATC-MC",
    "ATC-NE": "ATC-NE",
    "MDE": "MDE",
    "DoC": "DoC",
    "AC": "AC",
}
STAT_TEX = {"MSP (= AC, DoC)": "MSP / AC, DoC", "ATC (soft rate)": "ATC soft rate",
            "UniformCE": "Uniform-CE", "NuclearNorm": "Nuclear Norm"}
# Which literature a statistic was proposed in. MSP is the standard OOD-detection baseline
# (Hendrycks & Gimpel) and is also the score AC and DoC average for UAE; predictive entropy is
# likewise an OOD baseline and the score ATC-NE thresholds.
ORIGIN = {
    "kNN": "OOD detector", "ViM": "OOD detector", "ReAct": "OOD detector",
    "Mahalanobis": "OOD detector", "Energy": "OOD detector", "GradNorm": "OOD detector",
    "UniformCE": "OOD detector",
    "MSP (= AC, DoC)": "Both", "Entropy": "Both",
    "ATC (soft rate)": "UAE method", "MDE": "UAE method",
    "NuclearNorm": "UAE method", "COT": "UAE method", "MaNo": "UAE method", "Dispersion": "UAE method",
}
SIGNAL = {
    "kNN": "feature distance", "Mahalanobis": "feature distance", "ViM": "feature + logit",
    "ReAct": "clipped logit", "Energy": "logit", "GradNorm": "gradient", "UniformCE": "logit",
    "MSP (= AC, DoC)": "confidence", "Entropy": "confidence", "ATC (soft rate)": "confidence",
    "MDE": "logit (batch)", "NuclearNorm": "prediction matrix", "COT": "prediction matrix",
    "MaNo": "normalized logits", "Dispersion": "feature clusters",
}
ORIGIN_LABEL = {"OOD detector": "OOD detector", "UAE method": "UAE method",
                "Both": "Both (OOD baseline and UAE score)"}
# Validated categorical slots 1-3 (all-pairs pass, light surface); marker shape is the
# secondary encoding.
FAMILY_STYLE = {
    "OOD detector": ("#2a78d6", "o"),
    "UAE method": ("#eb6834", "s"),
    "Both": ("#1baf7a", "^"),
}
FAMILY = ORIGIN
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _fmt_p(p):
    return "<0.001" if p < 1e-3 else f"{p:.3f}"


def _tex_p(p):
    return "$<$0.001" if p < 1e-3 else f"{p:.3f}"


def _write(path, body):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(body)
    print(f"wrote {path}")


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def detection_vs_uae_frame():
    """Per (Dataset, Model, statistic): detection AUROC (restricted targets) and UAE MAE
    (all held-out folds), plus the UAE MAE split by synthetic / organic."""
    det = pd.read_csv(f"{FIG}/shift_detection_rows.csv")
    auroc = restrict_targets(det).groupby(["Dataset", "Model", "Method"])["AUROC"].mean()

    d = pd.read_csv(f"{FIG}/ranking_detector_rows.csv")
    d = d[d["Method"] == "Ours-mean"].copy()
    d["Method"] = d["feature_name"].replace({"MSP": "MSP (= AC, DoC)"})
    m = pd.read_csv(f"{FIG}/ranking_method_rows.csv")
    m = m[m["Method"].isin(["MDE", "Reg-ATC", "NuclearNorm", "COT"])].copy()
    m["Method"] = m["Method"].replace({"Reg-ATC": "ATC (soft rate)"})
    uae = pd.concat([d, m], ignore_index=True)
    uae = uae[~uae["fold"].isin(["train", "ind_val", "ind_test"])]

    keys = ["Dataset", "Model", "Method"]
    out = pd.concat([
        auroc,
        uae.groupby(keys)["MAE"].mean().rename("MAE"),
        uae[uae["category"] == "Synthetic"].groupby(keys)["MAE"].mean().rename("MAE_synthetic"),
        uae[uae["category"] == "Organic"].groupby(keys)["MAE"].mean().rename("MAE_organic"),
    ], axis=1).dropna(subset=["AUROC", "MAE"]).reset_index()

    rest = restrict_targets(det)
    split = pd.concat([
        rest[rest["category"] == "Organic"].groupby("Method")["AUROC"].mean().rename("AUROC_natural"),
        rest[rest["category"] == "Synthetic"].groupby("Method")["AUROC"].mean().rename("AUROC_synmax"),
    ], axis=1)
    return out, split


def association_stats(j):
    rows = []
    for label, col in [("All held-out folds", "MAE"), ("Synthetic folds", "MAE_synthetic"),
                       ("Natural folds", "MAE_organic")]:
        jj = j.dropna(subset=[col])
        within = np.array([spearmanr(g["AUROC"], g[col])[0]
                           for _, g in jj.groupby(["Dataset", "Model"]) if len(g) > 2])
        rho, p = spearmanr(jj["AUROC"], jj[col])
        rows.append({"scope": label, "median_rho": np.median(within),
                     "q25": np.percentile(within, 25), "q75": np.percentile(within, 75),
                     "n_neg": int((within < 0).sum()), "n_blocks": len(within),
                     "wilcoxon_p": wilcoxon(within).pvalue, "pooled_rho": rho, "pooled_p": p,
                     "n": len(jj), "within": within})
    return rows


# ----------------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------------
def table_detection_vs_uae(j, split):
    per = j.groupby("Method")[["AUROC", "MAE"]].mean().join(split)
    per = per.sort_values("AUROC", ascending=False)
    per["r_auc"] = per["AUROC"].rank(ascending=False).astype(int)
    per["r_mae"] = per["MAE"].rank().astype(int)
    rho, p = spearmanr(per["AUROC"], per["MAE"])
    tau = kendalltau(per["AUROC"], per["MAE"])[0]
    best_auc, best_mae = per["AUROC"].max(), per["MAE"].min()
    lines = []
    fam_prev = None
    for name, r in per.iterrows():
        a = f"{r['AUROC']:.3f}"
        e = f"{r['MAE']:.3f}"
        a = rf"\textbf{{{a}}}" if r["AUROC"] == best_auc else a
        e = rf"\textbf{{{e}}}" if r["MAE"] == best_mae else e
        lines.append(f"{STAT_TEX.get(name, name)} & {ORIGIN.get(name, '')} & {SIGNAL.get(name, '')} & {r['AUROC_natural']:.3f} & "
                     f"{r['AUROC_synmax']:.3f} & {a} & {int(r['r_auc'])} & {e} & {int(r['r_mae'])} \\\\")
    body = rf"""\begin{{table}}[t]
\centering
\caption{{Set-level shift detection and unsupervised accuracy estimation for the same statistics.
Detection AUROC separates batches of 100 shifted inputs from in-distribution batches, over natural
shifts and the maximum-intensity fold of each synthetic family. UAE MAE is the error of the same
statistic, linearly calibrated on synthetic shifts, over all held-out folds: for an OOD detector this is
our estimator with that detector held fixed (mean-score aggregate), for a UAE method its own
statistic. Proposed as: the literature that introduced the statistic. MSP is the standard OOD
baseline and also the score that AC and DoC average for UAE. Means over
{j[['Dataset', 'Model']].drop_duplicates().shape[0]} dataset--architecture blocks; ranks are 1 = best.
Across statistics, Spearman $\rho={rho:.2f}$ ($p={_fmt_p(p)}$), Kendall $\tau={tau:.2f}$.}}
\label{{tab:detection_vs_uae}}
\small
\resizebox{{\linewidth}}{{!}}{{%
\begin{{tabular}}{{lllccccrc}}
\toprule
 & & & \multicolumn{{4}}{{c}}{{Shift detection (AUROC $\uparrow$)}} & \multicolumn{{2}}{{c}}{{UAE (MAE $\downarrow$)}} \\
\cmidrule(lr){{4-7}}\cmidrule(lr){{8-9}}
Statistic & Proposed as & Signal & Natural & Synth.\ max & Both & Rank & MAE & Rank \\
\midrule
{chr(10).join(lines)}
\bottomrule
\end{{tabular}}}}
\end{{table}}
"""
    _write(f"{OUT_TAB}/tab_detection_vs_uae.tex", body)
    return per, rho, p, tau


def table_association(stats):
    lines = [f"{s['scope']} & {s['median_rho']:.2f} & [{s['q25']:.2f}, {s['q75']:.2f}] & "
             f"{s['n_neg']}/{s['n_blocks']} & {_tex_p(s['wilcoxon_p'])} & {s['pooled_rho']:.2f} & "
             f"{_tex_p(s['pooled_p'])} \\\\" for s in stats]
    body = rf"""\begin{{table}}[t]
\centering
\caption{{Association between a statistic's shift-detection AUROC (natural and maximum-intensity
synthetic shifts) and its UAE error, by the folds on which UAE error is measured. Within-block:
Spearman $\rho$ across statistics inside each dataset--architecture block; negative values mean
better detectors are better accuracy estimators. The Wilcoxon signed-rank test asks whether the
block-level $\rho$ are centred on zero. Pooled: Spearman $\rho$ over all (block, statistic) pairs,
which mixes between-dataset differences.}}
\label{{tab:detection_uae_corr}}
\small
\begin{{tabular}}{{lcccccc}}
\toprule
 & \multicolumn{{4}}{{c}}{{Within block}} & \multicolumn{{2}}{{c}}{{Pooled}} \\
\cmidrule(lr){{2-5}}\cmidrule(lr){{6-7}}
UAE error on & Median $\rho$ & IQR & $\rho<0$ & Wilcoxon $p$ & $\rho$ & $p$ \\
\midrule
{chr(10).join(lines)}
\bottomrule
\end{{tabular}}
\end{{table}}
"""
    _write(f"{OUT_TAB}/tab_detection_uae_corr.tex", body)


def table_uae_ranking():
    s = pd.read_csv(f"{FIG}/ranking_methods_summary.csv", index_col=0)
    o = pd.read_csv(f"{FIG}/ranking_methods_omnibus.csv").set_index("scope")
    piv = s.pivot_table(index=s.index, columns="scope", values="avg_rank")
    mae = s[s["scope"] == "All"]["mean_MAE"]
    wins = s[s["scope"] == "All"]["wins"]
    piv = piv.join(mae.rename("MAE")).join(wins.rename("wins")).sort_values("All")
    best = {c: piv[c].min() for c in ["All", "Synthetic", "Organic", "MAE"]}

    def cell(v, c):
        t = f"{v:.2f}" if c != "MAE" else f"{v:.3f}"
        return rf"\textbf{{{t}}}" if np.isclose(v, best[c]) else t

    lines = [f"{METHOD_TEX.get(m, m)} & {cell(r['All'], 'All')} & {cell(r['Synthetic'], 'Synthetic')} & "
             f"{cell(r['Organic'], 'Organic')} & {cell(r['MAE'], 'MAE')} & {int(r['wins'])} \\\\"
             for m, r in piv.iterrows()]
    foot = "; ".join(f"{sc}: $p={_fmt_p(o.loc[sc, 'friedman_p'])}$, $W={o.loc[sc, 'kendalls_W']:.2f}$"
                     for sc in ["All", "Synthetic", "Organic"] if sc in o.index)
    n = int(o.loc["All", "n_blocks"])
    body = rf"""\begin{{table}}[t]
\centering
\caption{{Unsupervised accuracy estimation: average rank of each method across {n}
dataset--architecture blocks (1 = lowest MAE), under leave-one-shift-family-out evaluation over all
held-out folds and intensities. Our estimator selects its detector by nested cross-validation on the
calibration families. Friedman tests and Kendall's $W$ (rank agreement across blocks): {foot}.
Wins = blocks in which the method has the lowest MAE.}}
\label{{tab:uae_ranking}}
\small
\begin{{tabular}}{{lccccc}}
\toprule
 & \multicolumn{{3}}{{c}}{{Average rank $\downarrow$}} & & \\
\cmidrule(lr){{2-4}}
Method & All & Synthetic & Natural & MAE $\downarrow$ & Wins \\
\midrule
{chr(10).join(lines)}
\bottomrule
\end{{tabular}}
\end{{table}}
"""
    _write(f"{OUT_TAB}/tab_uae_ranking.tex", body)


def table_calibration():
    a = pd.read_csv(f"{FIG}/calibration_all_pooled.csv", index_col=0)
    n = pd.read_csv(f"{FIG}/calibration_organic_pooled.csv", index_col=0)
    a = a.sort_values("binned_CE")
    lines = []
    for m, r in a.iterrows():
        q = n.loc[m] if m in n.index else None
        nat = (f"{q['bias']:+.3f} & {q['slope']:.2f} & {100 * q['missed_rate']:.0f}\\%" if q is not None
               else "-- & -- & --")
        lines.append(f"{METHOD_TEX.get(m, m)} & {r['bias']:+.3f} & {r['slope']:.2f} & {r['R2']:.2f} & "
                     f"{r['binned_CE']:.3f} & {100 * r['missed_rate']:.0f}\\% & {nat} \\\\")
    body = rf"""\begin{{table*}}[t]
\centering
\caption{{Calibration of predicted against observed accuracy. Bias: mean of predicted minus
observed accuracy (positive values overstate accuracy). Slope and $R^2$: least-squares fit of predicted
on observed accuracy (slope 1 is ideal; slopes below 1 compress large accuracy drops toward InD accuracy).
Binned CE: calibration error over observed-accuracy bins of width 0.1. Missed: share of accuracy drops
larger than 0.2 whose accuracy is overestimated by more than 0.1, the failure that hides degradation
from a monitor. All held-out folds ({int(a['n'].iloc[0])} per method)
and natural shifts only ({int(n['n'].iloc[0])} per method, {int(n['n_large'].iloc[0])} large drops).}}
\label{{tab:calibration}}
\small
\begin{{tabular}}{{lccccc@{{\hspace{{1.5em}}}}ccc}}
\toprule
 & \multicolumn{{5}}{{c}}{{All held-out folds}} & \multicolumn{{3}}{{c}}{{Natural shifts}} \\
\cmidrule(lr){{2-6}}\cmidrule(lr){{7-9}}
Method & Bias & Slope & $R^2$ & Binned CE & Missed & Bias & Slope & Missed \\
\midrule
{chr(10).join(lines)}
\bottomrule
\end{{tabular}}
\end{{table*}}
"""
    _write(f"{OUT_TAB}/tab_calibration.tex", body)


# ----------------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------------
def figure_detection_vs_uae(j, per, rho, p, stats):
    matplotlib.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": TEXT,
                                "xtick.color": MUTED, "ytick.color": MUTED, "text.color": TEXT})
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.9), gridspec_kw={"width_ratios": [2.3, 1]})

    # (a) per-statistic means, per-block points faint behind them
    for fam, (col, mk) in FAMILY_STYLE.items():
        blk = j[j["Method"].map(FAMILY) == fam]
        ax.scatter(blk["AUROC"], blk["MAE"], s=10, marker=mk, color=col, alpha=0.18, lw=0, zorder=1)
    for name, r in per.iterrows():
        col, mk = FAMILY_STYLE[FAMILY.get(name, "OOD detector")]
        ax.scatter(r["AUROC"], r["MAE"], s=46, marker=mk, color=col, edgecolor="white", lw=1.2, zorder=3)
    coef = np.polyfit(per["AUROC"], per["MAE"], 1)
    xs = np.linspace(per["AUROC"].min() - 0.02, per["AUROC"].max() + 0.02, 50)
    ax.plot(xs, np.polyval(coef, xs), color=MUTED, lw=1, ls="--", zorder=2)
    offsets = {"kNN": (5, 3), "ViM": (5, -10), "ReAct": (5, -2), "Mahalanobis": (6, 2), "Energy": (-6, 4),
               "ATC (soft rate)": (6, 1), "MSP (= AC, DoC)": (-6, 3), "MDE": (-6, 2),
               "GradNorm": (-6, 4), "UniformCE": (-6, -10), "NuclearNorm": (5, 3), "COT": (5, 3)}
    for name, r in per.iterrows():
        dx, dy = offsets.get(name, (5, 3))
        ax.annotate(STAT_TEX.get(name, name), (r["AUROC"], r["MAE"]), xytext=(dx, dy),
                    textcoords="offset points", fontsize=7, color=TEXT,
                    ha="right" if dx < 0 else "left")
    ax.set_xlabel("Shift-detection AUROC (natural + max-intensity synthetic) →")
    ax.set_ylabel("UAE MAE (all held-out folds) ↓")
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    hi = max(per["MAE"].max(), 0.2) * 1.15
    ax.set_ylim(0, min(hi, j["MAE"].quantile(0.97) * 1.05))
    ax.set_title(f"(a) Statistic means; faint: per block\nacross statistics Spearman ρ = {rho:.2f} "
                 f"(p = {_fmt_p(p)})", fontsize=8, loc="left")
    handles = [plt.Line2D([], [], marker=mk, ls="", color=col, markeredgecolor="white", markersize=6,
                          label=ORIGIN_LABEL[fam]) for fam, (col, mk) in FAMILY_STYLE.items()
               if (per.index.map(FAMILY) == fam).any()]
    ax.legend(handles=handles, frameon=False, fontsize=6.5, loc="lower left")

    # (b) within-block Spearman rho, per UAE scope
    labels = ["All", "Synthetic", "Natural"]
    rng = np.random.default_rng(0)
    for i, s in enumerate(stats):
        w = s["within"]
        ax2.scatter(np.full(len(w), i) + rng.uniform(-0.12, 0.12, len(w)), w, s=12, color="#2a78d6",
                    alpha=0.6, lw=0)
        ax2.hlines(s["median_rho"], i - 0.25, i + 0.25, color=TEXT, lw=1.6)
        ax2.annotate(f"{s['n_neg']}/{s['n_blocks']} < 0\np {'< 0.001' if s['wilcoxon_p'] < 1e-3 else '= ' + _fmt_p(s['wilcoxon_p'])}",
                     (i, 1.0), ha="center", va="top", fontsize=6.5, color=MUTED)
    ax2.axhline(0, color=MUTED, lw=0.8)
    ax2.set_xticks(range(3))
    ax2.set_xticklabels(labels)
    ax2.set_xlim(-0.6, 2.6)
    ax2.set_ylim(-1.05, 1.05)
    ax2.set_ylabel("Within-block Spearman ρ")
    ax2.set_xlabel("UAE error measured on")
    ax2.grid(axis="y", color=GRID, lw=0.6)
    ax2.set_axisbelow(True)
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)
    ax2.set_title("(b) Per block (bar: median)", fontsize=8, loc="left")
    fig.tight_layout()
    os.makedirs(OUT_FIG, exist_ok=True)
    fig.savefig(f"{OUT_FIG}/detection_vs_uae.pdf", bbox_inches="tight")
    fig.savefig(f"{FIG}/detection_vs_uae.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT_FIG}/detection_vs_uae.pdf")


def run():
    j, split = detection_vs_uae_frame()
    j.to_csv(f"{FIG}/detection_vs_uae_joined.csv", index=False)
    stats = association_stats(j)
    per, rho, p, tau = table_detection_vs_uae(j, split)
    table_association(stats)
    table_uae_ranking()
    table_calibration()
    figure_detection_vs_uae(j, per, rho, p, stats)


if __name__ == "__main__":
    matplotlib.use("Agg")
    run()
