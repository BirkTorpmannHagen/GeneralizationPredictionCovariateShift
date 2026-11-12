"""
Remaining paper figures and tables for OODDetectorsAreUAE/, from the cached per-sample scores and
the outputs of ranking_analysis / calibration_analysis / shift_detection.

    python -m experiments.paper_figures

Protocol (identical to ranking_analysis): leave-one-shift-family-out; the regression is fit on
synthetic calibration shifts only; our detector is chosen by nested LOSO on the calibration
families; the detection rate uses the InD 0.95-quantile threshold; evaluation covers every
held-out fold except ind_test.
"""
import os
import pickle

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression

from experiments.ablations import dr_calibration
from experiments.accuracy_prediction import pre_predictions
from experiments.calibration_analysis import _calib_metrics, with_accuracy
from experiments.ranking_analysis import (CLASSIFICATION, CLASSIFIERS, SEGMENTORS, _inner_loso_mae,
                                          _nested_selection_rows, _perfold_tables, cd_diagram,
                                          wilcoxon_holm, rank_matrix)
from utils import DSD_LUT, DSD_PRINT_LUT, FEATURE_COLUMN_ALIAS, load_data

FIG = "figures"
PAPER = "OODDetectorsAreUAE"
CACHE = f"{FIG}/perfold_tables.pkl"
DATASETS_ALL = CLASSIFICATION + ["Polyp"]
DS_TEX = {"CCT": "CCT", "OfficeHome": "OfficeHome", "Office31": "Office31", "NICO": "NICO",
          "Camelyon17": "Camelyon17", "IWildCam": "iWildCam", "Polyp": "Polyp"}
METHOD_TEX = {
    "Ours-dr95 (nested sel.)": "Ours (DR)", "Ours-mean (nested sel.)": "Ours (mean score)",
    "NuclearNorm": "Nuclear Norm", "LabelShiftL1": "Label-marginal $L_1$", "COT": "COT",
    "COT (native)": "COT (native)", "MaNo": "MaNo", "Dispersion": "Dispersion",
    "Reg-ATC": "ATC (regressed)", "ATC-MC": "ATC-MC",
    "ATC-NE": "ATC-NE", "MDE": "MDE", "DoC": "DoC", "AC": "AC", "PRE": "PRE", "PRE-0": "PRE-0",
}
METHOD_PLAIN = {k: v.replace("$L_1$", "L1") for k, v in METHOD_TEX.items()}
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]  # validated, adjacent pairs
LINESTYLES = ["-", "--", "-.", ":", (0, (5, 1))]
MARKERS = ["o", "s", "^", "D", "v"]
FAMILY = {"kNN": 0, "ViM": 0, "ReAct": 0, "Mahalanobis": 0, "Energy": 1, "GradNorm": 1,
          "UniformCE": 1, "Entropy": 2, "MSP": 2}
FAMILY_NAME = ["Feature distance / hybrid", "Logit score", "Confidence"]
FAMILY_MARK = ["o", "s", "^"]


def _style():
    matplotlib.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": TEXT,
                                "xtick.color": MUTED, "ytick.color": MUTED, "text.color": TEXT,
                                "axes.spines.top": False, "axes.spines.right": False})


def _grid(ax, axis="both"):
    ax.grid(axis=axis, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def _intensity(fold):
    try:
        return float(str(fold).rsplit("_", 1)[-1])
    except ValueError:
        return np.nan


def _write(path, body):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(body)
    print("wrote", path)


def _save(fig, name):
    os.makedirs(f"{PAPER}/figures", exist_ok=True)
    fig.savefig(f"{PAPER}/figures/{name}.pdf", bbox_inches="tight")
    fig.savefig(f"{FIG}/{name}.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("wrote", f"{PAPER}/figures/{name}.pdf")


def _facets(n, panel_w=1.8, panel_h=1.75, sharex=False, sharey=False):
    """Column-wrapped facet grid: <= 3 panels in one row, otherwise two rows with the shorter
    bottom row centred (4 -> 2+2, 6 -> 3+3, 7 -> 4+3). Returns (fig, axes, first_col) where
    first_col[i] says whether panel i starts a row (gets the y-label)."""
    ncol = n if n <= 3 else int(np.ceil(n / 2))
    nrow = int(np.ceil(n / ncol))
    fig = plt.figure(figsize=(panel_w * ncol, panel_h * nrow + 0.35))
    gs = fig.add_gridspec(nrow, 2 * ncol)
    axes, first_col = [], []
    for i in range(n):
        r, c = divmod(i, ncol)
        in_row = min(ncol, n - r * ncol)
        off = ncol - in_row  # centre a short last row (in half-column units)
        kw = {}
        if axes and sharex:
            kw["sharex"] = axes[0]
        if axes and sharey:
            kw["sharey"] = axes[0]
        axes.append(fig.add_subplot(gs[r, off + 2 * c: off + 2 * c + 2], **kw))
        first_col.append(c == 0)
    if sharey:
        for ax, fc in zip(axes, first_col):
            if not fc:
                ax.tick_params(labelleft=False)
    return fig, axes, first_col


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def tables():
    if os.path.exists(CACHE):
        with open(CACHE, "rb") as f:
            return pickle.load(f)
    t = _perfold_tables(CLASSIFICATION, CLASSIFIERS)
    t.update(_perfold_tables(["Polyp"], SEGMENTORS))
    with open(CACHE, "wb") as f:
        pickle.dump(t, f)
    return t


def method_rows(t):
    """All methods' per-fold rows on every dataset, Polyp included (ours + PRE only there)."""
    rows = pd.read_csv(f"{FIG}/ranking_method_rows.csv")
    polyp = {k: v for k, v in t.items() if k[0] == "Polyp"}
    extra = [_nested_selection_rows(polyp, agg) for agg in ("dr95", "mean")]
    pre = pre_predictions()
    pre = pre[pre["Method"] == "PRE"]
    pre["feature_name"] = pre["feature_name"].replace({"Entropy": "UniformCE"})
    rows = pd.concat([rows] + extra + [pre], ignore_index=True)
    return rows[~rows["fold"].isin(["train", "ind_val", "ind_test"]) & (rows["shift_type"] != "ind")]


# ----------------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------------
def table_mae_per_dataset(rows):
    order = ["Ours-dr95 (nested sel.)", "Ours-mean (nested sel.)", "MaNo", "Dispersion", "NuclearNorm", "COT", "LabelShiftL1",
             "Reg-ATC", "ATC-MC", "MDE", "DoC", "AC", "PRE"]
    order = [m for m in order if m in rows["Method"].unique()]
    # mean over architectures of the per-(dataset, architecture) mean MAE
    blk = rows.groupby(["Dataset", "Model", "Method"])["MAE"].mean().reset_index()
    piv = blk.groupby(["Method", "Dataset"])["MAE"].mean().unstack()
    piv = piv.reindex(index=order, columns=DATASETS_ALL)
    best = piv.min(axis=0)
    cls = piv[CLASSIFICATION].mean(axis=1)
    cls[piv[CLASSIFICATION].isna().any(axis=1)] = np.nan
    lines = []
    for m in order:
        cells = []
        for d in DATASETS_ALL:
            v = piv.loc[m, d]
            cells.append("--" if pd.isna(v) else (rf"\textbf{{{v:.3f}}}" if np.isclose(v, best[d]) else f"{v:.3f}"))
        c = cls[m]
        cells.append("--" if pd.isna(c) else (rf"\textbf{{{c:.3f}}}" if np.isclose(c, cls.min()) else f"{c:.3f}"))
        sep = "\\midrule\n" if m == next((x for x in ("MaNo", "Dispersion", "NuclearNorm") if x in order), None) else ""
        lines.append(f"{sep}{METHOD_TEX[m]} & " + " & ".join(cells) + r" \\")
    head = " & ".join(DS_TEX[d] for d in DATASETS_ALL)
    body = rf"""\begin{{tabular}}{{l{'c' * len(DATASETS_ALL)}c}}
\toprule
 & {head} & Mean (cls.) \\
\midrule
{chr(10).join(lines)}
\bottomrule
\end{{tabular}}
"""
    _write(f"{PAPER}/tables/mae_per_dataset.tex", body)
    return piv


def table_pairwise():
    w = pd.read_csv(f"{FIG}/ranking_methods_all_wilcoxon_holm.csv")
    ours = "Ours-dr95 (nested sel.)"
    w = w[(w["method_1"] == ours) | (w["method_2"] == ours)].copy()
    out = []
    for _, r in w.iterrows():
        flip = r["method_2"] == ours
        other = r["method_1"] if flip else r["method_2"]
        mo, mx = (r["mean_mae_2"], r["mean_mae_1"]) if flip else (r["mean_mae_1"], r["mean_mae_2"])
        wo, wx = (r["wins_2"], r["wins_1"]) if flip else (r["wins_1"], r["wins_2"])
        p = r["p_holm_control"] if "p_holm_control" in r and pd.notna(r["p_holm_control"]) else r["p_holm"]
        out.append((other, mo, mx, int(wo), int(wx), p))
    out.sort(key=lambda x: x[2])
    lines = [f"{METHOD_TEX.get(o, o)} & {mx:.3f} & {mo - mx:+.3f} & {wo}/{wo + wx} & "
             f"{'$<$0.001' if p < 1e-3 else f'{p:.3f}'}{'$^*$' if p < 0.05 else ''} \\\\"
             for o, mo, mx, wo, wx, p in out]
    body = rf"""\begin{{tabular}}{{lcccc}}
\toprule
Comparison method & MAE & $\Delta$MAE (ours $-$ method) & Ours wins & $p_{{\mathrm{{Holm}}}}$ \\
\midrule
{chr(10).join(lines)}
\bottomrule
\end{{tabular}}
"""
    _write(f"{PAPER}/tables/pairwise_tests.tex", body)


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
def fig_dr_vs_acc(t):
    """Per dataset: the (architecture, detector) whose DR correlates most strongly (negatively)
    with accuracy on the synthetic calibration folds; synthetic folds grey, natural shifts
    highlighted. (accuracy = InD-val accuracy - gap, so rho(DR, acc) = -rho(DR, gap).)"""
    _style()
    fig, axes, first_col = _facets(len(DATASETS_ALL), panel_w=1.8, panel_h=1.95)
    for ax, ds in zip(axes, DATASETS_ALL):
        best, best_r = None, -np.inf
        for (d, m, det, agg), pf in t.items():
            if d != ds or agg != "dr95":
                continue
            syn = pf[pf["category"] == "Synthetic"]
            if syn["stat"].nunique() > 1:
                r = spearmanr(syn["stat"], syn["gap"])[0]
                if r > best_r:
                    best, best_r = (m, det, pf), r
        m, det, pf = best
        # gap = InD-val accuracy - fold accuracy; recover raw accuracy for the y-axis
        raw = load_data(ds, det, batch_size=1, samples=1, model=m, shift="")
        raw = raw[raw["feature_name"] == FEATURE_COLUMN_ALIAS.get(det, det)]
        ind_acc = float(raw.loc[raw["fold"] == "ind_val", "correct_prediction"].mean())
        pf = pf[pf["fold"] != "ind_val"].assign(acc=lambda d: ind_acc - d["gap"])
        syn = pf[pf["category"] == "Synthetic"]
        org = pf[(pf["category"] == "Organic") & (pf["fold"] != "ind_test")]
        ax.scatter(syn["stat"], syn["acc"], s=6, color="#9a9890", alpha=0.6, lw=0, label="synthetic")
        ax.scatter(org["stat"], org["acc"], s=28, marker="D", color="#2a78d6", edgecolor="white",
                   lw=0.8, label="natural", zorder=3)
        ax.set_title(f"{DS_TEX[ds]}\n{m} / {DSD_PRINT_LUT.get(det, det)} (ρ={-best_r:.2f})", fontsize=7.5)
        ax.set_xlabel("Detection rate", fontsize=7.5)
        ax.tick_params(labelsize=7)
        _grid(ax)
    for ax, fc in zip(axes, first_col):
        if fc:
            ax.set_ylabel("Accuracy", fontsize=7.5)
    axes[0].legend(frameon=False, fontsize=6.5, loc="lower left")
    fig.tight_layout()
    _save(fig, "dr_vs_acc")


def fig_signed_rho(t):
    _style()
    recs = []
    for (d, m, det, agg), pf in t.items():
        if agg != "dr95":
            continue
        pf = pf[~pf["fold"].isin(["ind_val", "ind_test"])]
        if pf["stat"].nunique() > 1:
            recs.append({"Dataset": d, "Model": m, "det": DSD_PRINT_LUT.get(det, det),
                         "rho": -spearmanr(pf["stat"], pf["gap"])[0]})  # rho(DR, accuracy)
    r = pd.DataFrame(recs).dropna()
    r.to_csv(f"{FIG}/signed_rho_dr95.csv", index=False)
    fig, ax = plt.subplots(figsize=(6.5, 2.4))
    rng = np.random.default_rng(0)
    for i, ds in enumerate(DATASETS_ALL):
        g = r[r["Dataset"] == ds]
        for fam in range(3):
            gg = g[g["det"].map(FAMILY) == fam]
            ax.scatter(i + rng.uniform(-0.22, 0.22, len(gg)), gg["rho"], s=16, marker=FAMILY_MARK[fam],
                       color=SERIES[fam], edgecolor="white", lw=0.5, label=FAMILY_NAME[fam] if i == 0 else None)
        ax.hlines(g["rho"].median(), i - 0.3, i + 0.3, color=TEXT, lw=1.5)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xticks(range(len(DATASETS_ALL)))
    ax.set_xticklabels([DS_TEX[d] for d in DATASETS_ALL])
    ax.set_ylabel("Spearman ρ (DR vs. accuracy)")
    ax.set_ylim(-1.05, 1.05)
    _grid(ax, "y")
    ax.legend(frameon=False, fontsize=6.5, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    _save(fig, "signed_rho")
    return r


def fig_intensity_breakdown(rows):
    _style()
    methods = ["Ours-dr95 (nested sel.)", "NuclearNorm", "LabelShiftL1", "ATC-MC", "MDE"]
    rows = rows[rows["Dataset"].isin(CLASSIFICATION) & rows["Method"].isin(methods)].copy()
    rows["intensity"] = rows["fold"].map(_intensity)
    fig, axes, first_col = _facets(len(CLASSIFICATION), panel_w=2.3, panel_h=1.9, sharey=True)
    for ax, ds in zip(axes, CLASSIFICATION):
        g = rows[rows["Dataset"] == ds]
        for k, m in enumerate(methods):
            gm = g[g["Method"] == m]
            syn = gm[gm["category"] == "Synthetic"].groupby("intensity")["MAE"]
            mu = syn.mean()
            ax.plot(mu.index, mu.values, color=SERIES[k], ls=LINESTYLES[k], lw=1.6,
                    label=METHOD_PLAIN[m])
            org = gm[gm["category"] == "Organic"]["MAE"].mean()
            ax.scatter(0.56 + 0.012 * k, org, marker=MARKERS[k], color=SERIES[k], s=26,
                       edgecolor="white", lw=0.6, zorder=3)
        ax.axvline(0.535, color=MUTED, lw=0.7, ls=":")
        top = g.groupby(["Method", "intensity"])["MAE"].mean().max()
        if top > 0.6:
            ax.annotate(f"off-scale (max {top:.2f})", (0.3, 0.585), ha="center", fontsize=6, color=MUTED)
        ax.set_xticks([0.1, 0.3, 0.5, 0.585])
        ax.set_xticklabels(["0.1", "0.3", "0.5", "nat."])
        ax.set_title(DS_TEX[ds], fontsize=8)
        ax.set_xlabel("Synthetic intensity α")
        _grid(ax)
    for ax, fc in zip(axes, first_col):
        if fc:
            ax.set_ylabel("MAE")
    axes[0].set_ylim(0, 0.6)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=7, ncol=len(methods), loc="lower center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    _save(fig, "intensity_breakdown")


def fig_error_per_acc(rows):
    _style()
    ours = with_accuracy(rows[rows["Method"] == "Ours-dr95 (nested sel.)"])
    fig, axes, first_col = _facets(len(DATASETS_ALL), panel_w=1.9, panel_h=1.8, sharey=True)
    for ax, ds in zip(axes, DATASETS_ALL):
        g = ours[ours["Dataset"] == ds].sort_values("observed_acc")
        ax.scatter(g["observed_acc"], g["MAE"], s=4, color="#2a78d6", alpha=0.35, lw=0)
        roll = g["MAE"].rolling(25, center=True, min_periods=5).mean()
        ax.plot(g["observed_acc"], roll, color=TEXT, lw=1.2, label="running mean")
        naive = g["observed_gap"].abs().rolling(25, center=True, min_periods=5).mean()
        ax.plot(g["observed_acc"], naive, color=MUTED, lw=0.9, ls="--", label="naive (InD accuracy)")
        ax.set_title(DS_TEX[ds], fontsize=8)
        ax.set_xlabel("Observed accuracy")
        _grid(ax)
    for ax, fc in zip(axes, first_col):
        if fc:
            ax.set_ylabel("Absolute error")
    axes[0].set_ylim(0, 0.5)
    axes[0].legend(frameon=False, fontsize=6.5, loc="upper left")
    fig.tight_layout()
    _save(fig, "error_per_acc")


def seq_length(t, lengths=(4, 8, 16, 32, 64, 128, 256, 512), n_seq=30, seed=0):
    """MAE of our DR estimator on sequences of n samples drawn from each held-out fold (real
    per-sample detector verdicts and correctness; regression and detector as in the protocol)."""
    rng = np.random.default_rng(seed)
    out = []
    for ds in CLASSIFICATION:
        for m in CLASSIFIERS:
            pools = {det: pf[(pf["category"] == "Synthetic")]
                     for (d, mm, det, agg), pf in t.items() if (d, mm, agg) == (ds, m, "dr95")}
            if not pools:
                continue
            fams = sorted(set.intersection(*[set(p["shift_type"]) for p in pools.values()]))
            cache = {}

            def verdicts(det):
                if det not in cache:
                    raw = load_data(ds, det, samples=1, model=m, shift="")
                    if raw.empty:  # incomplete collection (see utils.load_data)
                        cache[det] = (None, None)
                        return cache[det]
                    raw = raw[raw["feature_name"] == FEATURE_COLUMN_ALIAS.get(det, det)]
                    hi, thr = dr_calibration(raw, 0.95)
                    v = (raw["feature"] > thr) if hi else (raw["feature"] < thr)
                    iv = raw[raw["fold"] == "ind_val"]["correct_prediction"].mean()
                    cache[det] = (raw.assign(v=v.astype(float)), iv)
                return cache[det]

            targets = [(h, [h]) for h in fams] + [(None, None)]
            for held, _ in targets:
                if held is None:
                    inner = {d: _inner_loso_mae(p) for d, p in pools.items()}
                    det = min(inner, key=inner.get)
                    tr = pools[det]
                else:
                    inner = {d: _inner_loso_mae(p[p["shift_type"] != held]) for d, p in pools.items()}
                    det = min(inner, key=inner.get)
                    tr = pools[det][pools[det]["shift_type"] != held]
                reg = LinearRegression().fit(tr[["stat"]].values, tr["gap"].values)
                raw, ind_acc = verdicts(det)
                if raw is None:
                    continue
                if held is None:
                    folds = [f for f in raw["fold"].unique() if f not in ("train", "ind_val", "ind_test")
                             and f.split("_")[0] not in fams]
                else:
                    folds = [f for f in raw["fold"].unique() if f.startswith(held + "_")]
                for f in folds:
                    g = raw[raw["fold"] == f]
                    v, c = g["v"].to_numpy(), g["correct_prediction"].to_numpy(float)
                    for n in lengths:
                        idx = rng.integers(0, len(v), size=(n_seq, n))
                        pred = np.clip(ind_acc - reg.predict(v[idx].mean(1).reshape(-1, 1)), 0.0, 1.0)
                        err = np.abs(pred - c[idx].mean(1))
                        out.append({"Dataset": ds, "Model": m, "fold": f, "n": n,
                                    "category": "Organic" if held is None else "Synthetic",
                                    "MAE": err.mean(), "MAE_sd": err.std()})
    s = pd.DataFrame(out)
    s.to_csv(f"{FIG}/seq_length_rows.csv", index=False)
    return s


def fig_seq_length(s):
    _style()
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    for k, (cat, lab) in enumerate([("Synthetic", "synthetic folds"), ("Organic", "natural shifts")]):
        g = s[s["category"] == cat].groupby("n")
        mu, sd = g["MAE"].mean(), g["MAE_sd"].mean()
        ax.errorbar(mu.index, mu.values, yerr=sd.values, color=SERIES[k], marker=MARKERS[k], ms=4,
                    lw=1.4, capsize=2, label=lab, ls=LINESTYLES[k])
    allm = s.groupby("n")["MAE"].mean()
    c = np.mean(allm.values * np.sqrt(allm.index.values))
    xs = np.array(sorted(s["n"].unique()))
    ax.plot(xs, c / np.sqrt(xs), color=MUTED, lw=0.9, ls=":", label=r"$c/\sqrt{n}$")
    ax.axhline(0.1, color=GRID, lw=1)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Sequence length n")
    ax.set_ylabel("MAE")
    _grid(ax)
    ax.legend(frameon=False, fontsize=6.5)
    fig.tight_layout()
    _save(fig, "seq_length")
    return allm


def fig_cd_methods():
    scores = pd.read_csv(f"{FIG}/ranking_methods_all_block_scores.csv", index_col=[0, 1])
    pw = pd.read_csv(f"{FIG}/ranking_methods_all_wilcoxon_holm.csv")
    om = pd.read_csv(f"{FIG}/ranking_methods_omnibus.csv").set_index("scope")
    ren = {k: METHOD_PLAIN.get(k, k) for k in scores.columns}
    scores = scores.rename(columns=ren)
    pw = pw.assign(method_1=pw["method_1"].map(lambda x: ren.get(x, x)),
                   method_2=pw["method_2"].map(lambda x: ren.get(x, x)))
    from experiments.ranking_analysis import nemenyi_cd
    ranks = rank_matrix(scores).mean().sort_values()
    os.makedirs(f"{PAPER}/figures", exist_ok=True)
    cd_diagram(ranks, nemenyi_cd(scores.shape[1], scores.shape[0]), pw,
               f"{scores.shape[0]} dataset–architecture blocks; Friedman p = {om.loc['All', 'friedman_p']:.1g}, "
               f"Kendall's W = {om.loc['All', 'kendalls_W']:.2f}", f"{PAPER}/figures/cd_methods.pdf")
    print("wrote", f"{PAPER}/figures/cd_methods.pdf")


def fig_calibration(rows):
    _style()
    methods = ["Ours-dr95 (nested sel.)", "NuclearNorm", "LabelShiftL1", "Reg-ATC", "ATC-MC", "MDE"]
    rows = with_accuracy(rows[rows["Dataset"].isin(CLASSIFICATION)])
    fig, axes, first_col = _facets(len(methods), panel_w=2.1, panel_h=2.1, sharex=True, sharey=True)
    lim = (0.0, 1.02)
    for ax, m in zip(axes, methods):
        g = rows[rows["Method"] == m]
        syn, org = g[g["category"] == "Synthetic"], g[g["category"] == "Organic"]
        ax.scatter(syn["observed_acc"], syn["predicted_acc"], s=3, color="#9a9890", alpha=0.35, lw=0,
                   label="synthetic")
        ax.scatter(org["observed_acc"], org["predicted_acc"], s=16, marker="D", color="#2a78d6",
                   edgecolor="white", lw=0.5, label="natural", zorder=3)
        ax.plot(lim, lim, color=TEXT, lw=0.8, ls="--")
        met = _calib_metrics(g)
        ax.set_title(f"{METHOD_PLAIN[m]}\nslope {met['slope']:.2f}, bias {met['bias']:+.3f}", fontsize=7.5)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel("Observed accuracy")
        _grid(ax)
    for ax, fc in zip(axes, first_col):
        if fc:
            ax.set_ylabel("Predicted accuracy")
    axes[0].legend(frameon=False, fontsize=6.5, loc="upper left")
    fig.tight_layout()
    _save(fig, "calibration")


def run():
    t = tables()
    rows = method_rows(t)
    rows.to_csv(f"{FIG}/paper_method_rows.csv", index=False)
    piv = table_mae_per_dataset(rows)
    print(piv.round(3).to_string())
    table_pairwise()
    fig_dr_vs_acc(t)
    r = fig_signed_rho(t)
    print("signed rho: median per dataset", r.groupby("Dataset")["rho"].median().round(2).to_dict(),
          "share wrong-sign (positive)", round((r["rho"] > 0).mean(), 3), "n", len(r))
    print("wrong-sign by detector:", r[r["rho"] > 0]["det"].value_counts().to_dict())
    fig_intensity_breakdown(rows)
    fig_error_per_acc(rows)
    fig_cd_methods()
    fig_calibration(rows)
    s = seq_length(t)
    allm = fig_seq_length(s)
    print("seq length MAE:", allm.round(3).to_dict())
    pol = rows[rows["Dataset"] == "Polyp"]
    print("Polyp MAE by method/category:\n", pol.groupby(["Method", "category"])["MAE"].mean().round(3))
    print("Polyp calib (ours):", _calib_metrics(pol[pol["Method"] == "Ours-dr95 (nested sel.)"]).round(3).to_dict())


if __name__ == "__main__":
    matplotlib.use("Agg")
    run()
