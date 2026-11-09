"""
Label-shift stress test for unsupervised accuracy estimators.

Real deployment batches rarely preserve the training label distribution (camera traps record
bursts of one species, disease prevalence differs between sites, streams arrive ordered by
class). This module re-batches every evaluation fold under controlled class mixes and asks how
well each estimator, *calibrated exactly as in the main protocol*, tracks the true accuracy of
each batch.

    python -m experiments.label_shift

Batches (size N_BATCH, N_REP per fold and scheme, drawn with replacement):
  iid        : uniform over the fold (original class mix; control)
  dir1, dir01: class weights ~ Dirichlet(1) / Dirichlet(0.1) over the fold's classes
  k5, k2, k1 : only k randomly chosen classes, equal weight (k1 = single-class batches)
  hard, easy : only the 25% of classes with the lowest / highest in-distribution accuracy
Classes with fewer than MIN_PER_CLASS samples in the fold are not eligible.

Folds: every held-out fold of the LOSO protocol (synthetic families held out one at a time,
organic shifts evaluated with the full calibration pool), plus the in-distribution test split
(category "InD"), where the class mix is the only shift.

Methods (all regressions are fit on unresampled calibration folds, exactly as in the main
protocol; nothing is re-fit on resampled data):
  Ours (DR) / Ours (mean)  nested detector selection, as in ranking_analysis
  DR:<detector>            each detector held fixed (DR aggregate), for the detector analysis
  ATC-MC, AC, DoC          native forms
  ATC (regressed), MDE     LOSO linear calibration
  NuclearNorm, COT, LabelShiftL1   from data/softmax/<model>/<dataset>/<fold>.npz when present
                           (written by collect_softmax_stats.py on the GPU machine)

Also computes, per detector, the class dependence of its in-distribution scores
(eta^2 = between-class variance / total variance on the InD test split): a score that does not
depend on the class should not react to the class mix.
"""
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from experiments.accuracy_prediction import _atc_threshold, _is_synthetic_shift_type, _parse_shift_type_from_fold
from experiments.ranking_analysis import CLASSIFICATION, CLASSIFIERS, _inner_loso_mae
from experiments.ablations import mde_statistic
from utils import DSD_PRINT_LUT, DSDS, FEATURE_COLUMN_ALIAS, load_data

FIG = "figures"
CACHE = f"{FIG}/perfold_tables.pkl"
SOFTMAX_DIR = "data/softmax"
N_BATCH = 100
N_REP = 20
CAP = 2000
MIN_PER_CLASS = 5
SCHEMES = ["iid", "dir1", "dir01", "k5", "k2", "k1", "hard", "easy"]
SCHEME_TEX = {"iid": "Original mix", "dir1": r"Dirichlet $\alpha{=}1$", "dir01": r"Dirichlet $\alpha{=}0.1$",
              "k5": "5 classes", "k2": "2 classes", "k1": "1 class", "hard": "Hardest classes",
              "easy": "Easiest classes"}
SEED = 0


# ----------------------------------------------------------------------------
# Per-sample data
# ----------------------------------------------------------------------------
def _load_block(ds, model, rng):
    """Wide per-sample frame for one (dataset, model): fold, idx, class, correct, and one
    oriented-threshold verdict + raw score per detector. Folds are capped at CAP samples."""
    wide, meta = None, {}
    for det in DSDS:
        raw = load_data(ds, det, batch_size=1, samples=1, model=model, shift="")
        if raw.empty:
            continue
        raw = raw[raw["feature_name"] == FEATURE_COLUMN_ALIAS.get(det, det)]
        # orientation + InD 0.95-quantile threshold, exactly as in ablations._perfold_dr
        ind, ood = raw[~raw["ood"]], raw[raw["ood"]]
        hi = ood["feature"].mean() > ind["feature"].mean()
        thr = float(ind["feature"].quantile(0.95 if hi else 0.05))
        meta[det] = (hi, thr)
        raw = raw[raw["fold"] != "train"]
        cols = raw[["fold", "idx", "class", "correct_prediction", "feature"]].rename(
            columns={"feature": f"s:{det}", "correct_prediction": "correct"})
        cols[f"v:{det}"] = ((cols[f"s:{det}"] > thr) if hi else (cols[f"s:{det}"] < thr)).astype(np.float32)
        if wide is None:
            # fix the per-fold sample subset once, from the first detector
            # in-distribution folds stay complete (accuracy and ATC threshold as in the main protocol)
            keep = cols.groupby("fold", group_keys=False).apply(
                lambda g: g if g.name in ("ind_val", "ind_test")
                else g.sample(min(len(g), CAP), random_state=int(rng.integers(1 << 31))))
            wide = keep
        else:
            wide = wide.merge(cols[["fold", "idx", f"s:{det}", f"v:{det}"]], on=["fold", "idx"], how="inner")  # drop_last differs between collection runs
    return wide, meta


def _class_weights(classes, scheme, rng, rank):
    """Probability of each eligible class for one batch under a scheme (None = sample uniformly)."""
    k = len(classes)
    if scheme == "iid":
        return None
    if scheme.startswith("dir"):
        a = 1.0 if scheme == "dir1" else 0.1
        return dict(zip(classes, rng.dirichlet(np.full(k, a))))
    if scheme in ("k5", "k2", "k1"):
        kk = min(int(scheme[1:]), k)
        chosen = rng.choice(classes, size=kk, replace=False)
        return {c: 1.0 / kk for c in chosen}
    if scheme in ("hard", "easy"):
        ranked = [c for c in rank if c in set(classes)]  # ascending InD accuracy
        q = max(1, int(round(0.25 * len(ranked))))
        chosen = ranked[:q] if scheme == "hard" else ranked[-q:]
        return {c: 1.0 / len(chosen) for c in chosen}
    raise ValueError(scheme)


def _draw(fold_df, scheme, rng, rank):
    """N_REP index arrays (into fold_df rows) of N_BATCH samples each."""
    cls = fold_df["class"].to_numpy()
    counts = pd.Series(cls).value_counts()
    eligible = list(counts[counts >= MIN_PER_CLASS].index)
    by_class = {c: np.flatnonzero(cls == c) for c in eligible}
    out = []
    for _ in range(N_REP):
        w = _class_weights(eligible, scheme, rng, rank) if eligible else None
        if w is None:
            out.append(rng.integers(0, len(cls), N_BATCH))
            continue
        cs = list(w.keys())
        ps = np.array(list(w.values()), dtype=float)
        counts_per_class = rng.multinomial(N_BATCH, ps / ps.sum())
        parts = [by_class[c][rng.integers(0, len(by_class[c]), n)] for c, n in zip(cs, counts_per_class) if n]
        batch = np.concatenate(parts)
        rng.shuffle(batch)
        out.append(batch)
    return np.stack(out)


# ----------------------------------------------------------------------------
# Calibrations (fit on unresampled folds, as in the main protocol)
# ----------------------------------------------------------------------------
def _corr(a, b):
    """Pearson correlation between predicted and true batch accuracy (NaN if either is constant)."""
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def _fit(pool):
    return LinearRegression().fit(pool[["stat"]].values, pool["gap"].values)


def _pools(tables, ds, model, agg):
    return {d: pf[(pf["category"] == "Synthetic") & (pf["shift_type"] != "ind")]
            for (dd, mm, d, a), pf in tables.items() if (dd, mm, a) == (ds, model, agg)}


def _select(pools, held):
    inner = {d: _inner_loso_mae(p if held is None else p[p["shift_type"] != held]) for d, p in pools.items()}
    return min(inner, key=inner.get)


def _perfold(wide_full, fn):
    """Fold-level statistic table (stat, gap, shift_type, category) from per-sample data."""
    ind_acc = wide_full.loc[wide_full["fold"] == "ind_val", "correct"].mean()
    recs = []
    for f, g in wide_full.groupby("fold"):
        st = _parse_shift_type_from_fold(f)
        recs.append({"fold": f, "shift_type": st,
                     "category": "Synthetic" if _is_synthetic_shift_type(st) else "Organic",
                     "stat": fn(g), "gap": ind_acc - g["correct"].mean()})
    return pd.DataFrame(recs)


def _softmax_block(ds, model):
    d = os.path.join(SOFTMAX_DIR, model, ds)
    if not os.path.isdir(d):
        return None
    out = {}
    for fn in os.listdir(d):
        if fn.endswith(".npz"):
            z = np.load(os.path.join(d, fn))
            out[fn[:-4]] = (z["p"].astype(np.float32), z["y"], z["idx"])
    return out or None


# ----------------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------------
PAIRING_MIN_CORR = 0.1


class ScrambledPairingError(RuntimeError):
    pass


def check_pairing(wide, ds, model):
    """Per-sample analyses need each detector score paired with *its own* sample's correctness
    and class. Data collected before the shuffle fix in testbeds/base.py are scrambled
    (correlation between max-softmax and correctness ~0). Refuse to run on such data."""
    iv = wide[wide["fold"] == "ind_val"]
    if "s:msp" not in iv.columns or iv["correct"].nunique() < 2:
        return
    r = np.corrcoef(iv["s:msp"].astype(float), iv["correct"].astype(float))[0, 1]
    if not r > PAIRING_MIN_CORR:
        raise ScrambledPairingError(
            f"{ds}/{model}: corr(MSP, correct) on ind_val = {r:.3f}; per-sample scores are not paired "
            f"with their own samples. Recollect features with the fixed (unshuffled) loaders, see GPU_RERUN.md.")


def run_block(ds, model, tables, rng):
    wide, meta = _load_block(ds, model, rng)
    check_pairing(wide, ds, model)
    dets = [d for d in DSDS if f"v:{d}" in wide.columns]
    ind = wide[wide["fold"] == "ind_val"]
    ind_acc = ind["correct"].mean()
    rank = list(ind.groupby("class")["correct"].mean().sort_values().index)

    # class dependence of InD scores (eta^2) per detector
    test = wide[wide["fold"] == "ind_test"]
    eta = {}
    for d in dets:
        s = test[f"s:{d}"].astype(float)
        tot = s.var(ddof=0)
        between = test.groupby("class")[f"s:{d}"].mean().reindex(test["class"]).to_numpy().var(ddof=0)
        eta[d] = float(between / tot) if tot > 0 else np.nan

    # confidence baselines
    has_msp = "s:msp" in wide.columns
    if has_msp:
        tau = _atc_threshold(ind["s:msp"].values, ind["correct"].values)
        ind_conf = ind["s:msp"].mean()
        reg_atc_pf = _perfold(wide, lambda g: float((g["s:msp"] < tau).mean()))
    if "s:energy" in wide.columns:
        # MDE depends on set size through its log-sum-exp term, so calibrate it at the batch size
        # it is evaluated at: mean MDE over random N_BATCH-sample subsets of each fold.
        def _mde_at_batch_size(g, r=np.random.default_rng(SEED)):
            v = g["s:energy"].to_numpy(float)
            return float(np.mean([mde_statistic(v[r.integers(0, len(v), N_BATCH)]) for _ in range(N_REP)]))
        mde_pf = _perfold(wide, _mde_at_batch_size)

    folds = [f for f in wide["fold"].unique() if f != "ind_val"]
    pools = {agg: _pools(tables, ds, model, agg) for agg in ("dr95", "mean")}
    families = sorted(set.intersection(*[set(p["shift_type"]) for p in pools["dr95"].values()]))

    rows = []
    for f in folds:
        st = _parse_shift_type_from_fold(f)
        if f == "ind_test":
            cat, held = "InD", None
        elif _is_synthetic_shift_type(st):
            cat, held = "Synthetic", st
            if st not in families:
                continue
        else:
            cat, held = "Organic", None
        g = wide[wide["fold"] == f].reset_index(drop=True)
        correct = g["correct"].to_numpy(float)

        # estimators: name -> function(batch index array) -> predicted accuracy
        est = {}
        for agg, tag in (("dr95", "Ours (DR)"), ("mean", "Ours (mean)")):
            d = _select(pools[agg], held)
            p = pools[agg][d]
            reg = _fit(p if held is None else p[p["shift_type"] != held])
            col = (f"v:{d}" if agg == "dr95" else f"s:{d}")
            vals = g[col].to_numpy(float)
            est[tag] = (lambda b, v=vals, r=reg: ind_acc - r.predict(v[b].mean(1).reshape(-1, 1)))
        for d in dets:
            p = pools["dr95"].get(d)
            if p is None:
                continue
            reg = _fit(p if held is None else p[p["shift_type"] != held])
            vals = g[f"v:{d}"].to_numpy(float)
            est[f"DR:{DSD_PRINT_LUT.get(d, d)}"] = (lambda b, v=vals, r=reg: ind_acc - r.predict(v[b].mean(1).reshape(-1, 1)))
        if has_msp:
            m = g["s:msp"].to_numpy(float)
            est["ATC-MC"] = lambda b, m=m: (m[b] >= tau).mean(1)
            est["AC"] = lambda b, m=m: m[b].mean(1)
            est["DoC"] = lambda b, m=m: ind_acc - (ind_conf - m[b].mean(1))
            pool = reg_atc_pf[reg_atc_pf["category"] == "Synthetic"]
            r = _fit(pool if held is None else pool[pool["shift_type"] != held])
            est["ATC (regressed)"] = lambda b, m=m, r=r: ind_acc - r.predict((m[b] < tau).mean(1).reshape(-1, 1))
        if "s:energy" in wide.columns:
            e = g["s:energy"].to_numpy(float)
            pool = mde_pf[mde_pf["category"] == "Synthetic"]
            r = _fit(pool if held is None else pool[pool["shift_type"] != held])
            est["MDE"] = lambda b, e=e, r=r: ind_acc - r.predict(np.array([mde_statistic(e[i]) for i in b]).reshape(-1, 1))

        # Label-blind oracle: the fold's true overall accuracy. No estimator that ignores the class
        # composition of a batch can do better; its error is the part of the batch-level accuracy
        # change that is caused by the class mix alone.
        fold_acc = correct.mean()
        est["Oracle (fold accuracy)"] = lambda b, a=fold_acc: np.full(len(b), a)

        for scheme in SCHEMES:
            b = _draw(g, scheme, rng, rank)
            true_acc = correct[b].mean(1)
            for name, fn in est.items():
                pred = np.asarray(fn(b), dtype=float)
                err = pred - true_acc
                rows.append({"Dataset": ds, "Model": model, "Method": name, "scheme": scheme, "fold": f,
                             "shift_type": cat if cat != "Synthetic" else st, "category": cat,
                             "MAE": float(np.abs(err).mean()), "bias": float(err.mean()),
                             "track_r": _corr(pred, true_acc),
                             "true_gap": float(ind_acc - true_acc.mean())})
    rows += _softmax_rows(ds, model, ind_acc, rank, families, rng)
    eta_rows = [{"Dataset": ds, "Model": model, "detector": DSD_PRINT_LUT.get(d, d), "eta2": v} for d, v in eta.items()]
    return rows, eta_rows


def _softmax_rows(ds, model, ind_acc_feat, rank, families, rng):
    """Nuclear Norm, COT and the label-marginal statistic on re-batched softmax matrices."""
    sm = _softmax_block(ds, model)
    if sm is None or "ind_val" not in sm:
        return []
    from collect_softmax_stats import cot_error, nuclear_norm
    Pv, yv, _ = sm["ind_val"]
    ind_acc = float((Pv.argmax(1) == yv).mean())
    pi = np.bincount(yv, minlength=Pv.shape[1]).astype(float)
    pi /= pi.sum()
    ref = Pv.mean(0)
    stats = {"NuclearNorm": nuclear_norm, "COT": lambda P: cot_error(P, pi, rng),
             "LabelShiftL1": lambda P: float(np.abs(P.mean(0) - ref).sum())}
    recs = []
    for f, (P, y, _) in sm.items():
        st = _parse_shift_type_from_fold(f)
        recs.append({"fold": f, "shift_type": st, "category": "Synthetic" if _is_synthetic_shift_type(st) else "Organic",
                     **{k: fn(P) for k, fn in stats.items()}, "gap": ind_acc - float((P.argmax(1) == y).mean())})
    pf = pd.DataFrame(recs)
    pool_all = pf[pf["category"] == "Synthetic"]
    rows = []
    for f, (P, y, _) in sm.items():
        if f == "ind_val":
            continue
        st = _parse_shift_type_from_fold(f)
        cat = "InD" if f == "ind_test" else ("Synthetic" if _is_synthetic_shift_type(st) else "Organic")
        held = st if cat == "Synthetic" else None
        if cat == "Synthetic" and st not in families:
            continue
        g = pd.DataFrame({"class": y})
        correct = (P.argmax(1) == y).astype(float)
        regs = {}
        for k in stats:
            pool = pool_all if held is None else pool_all[pool_all["shift_type"] != held]
            regs[k] = LinearRegression().fit(pool[[k]].values, pool["gap"].values)
        for scheme in SCHEMES:
            b = _draw(g, scheme, rng, rank)
            true_acc = correct[b].mean(1)
            for k, fn in stats.items():
                s = np.array([fn(P[i]) for i in b]).reshape(-1, 1)
                pred = ind_acc - regs[k].predict(s)
                err = pred - true_acc
                rows.append({"Dataset": ds, "Model": model, "Method": k, "scheme": scheme, "fold": f,
                             "shift_type": cat if cat != "Synthetic" else st, "category": cat,
                             "MAE": float(np.abs(err).mean()), "bias": float(err.mean()),
                             "track_r": _corr(pred.ravel(), true_acc),
                             "true_gap": float(ind_acc - true_acc.mean())})
            if "COT" in stats:  # native COT: predicted accuracy = 1 - transport cost
                pred = 1.0 - np.array([stats["COT"](P[i]) for i in b])
                err = pred - true_acc
                rows.append({"Dataset": ds, "Model": model, "Method": "COT (native)", "scheme": scheme, "fold": f,
                             "shift_type": cat if cat != "Synthetic" else st, "category": cat,
                             "MAE": float(np.abs(err).mean()), "bias": float(err.mean()),
                             "track_r": _corr(pred, true_acc),
                             "true_gap": float(ind_acc - true_acc.mean())})
    return rows


def run_label_shift(datasets=CLASSIFICATION, models=CLASSIFIERS):
    import pickle
    with open(CACHE, "rb") as fh:
        tables = pickle.load(fh)
    rng = np.random.default_rng(SEED)
    rows, eta = [], []
    for ds in datasets:
        for m in models:
            r, e = run_block(ds, m, tables, rng)
            rows += r
            eta += e
            print(f"[label_shift] {ds}/{m}: {len(r)} rows", flush=True)
    rows, eta = pd.DataFrame(rows), pd.DataFrame(eta)
    rows.to_csv(f"{FIG}/label_shift_rows.csv", index=False)
    eta.to_csv(f"{FIG}/label_shift_eta2.csv", index=False)
    return rows, eta


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    run_label_shift()


# ----------------------------------------------------------------------------
# Paper assets (run after run_label_shift on correctly paired data)
# ----------------------------------------------------------------------------
PAPER = "OODDetectorsAreUAE"
TABLE_METHODS = ["Oracle (fold accuracy)", "Ours (DR)", "Ours (mean)", "NuclearNorm", "COT", "COT (native)",
                 "ATC (regressed)", "ATC-MC", "MDE", "DoC", "AC"]
METHOD_TEX = {"Oracle (fold accuracy)": "Label-blind oracle", "Ours (DR)": "Ours (DR)",
              "Ours (mean)": "Ours (mean score)", "NuclearNorm": "Nuclear Norm", "COT": "COT",
              "COT (native)": "COT (native)", "ATC (regressed)": "ATC (regressed)", "ATC-MC": "ATC-MC",
              "MDE": "MDE", "DoC": "DoC", "AC": "AC"}


def paper_assets(rows=None, eta=None):
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr
    rows = pd.read_csv(f"{FIG}/label_shift_rows.csv") if rows is None else rows
    eta = pd.read_csv(f"{FIG}/label_shift_eta2.csv") if eta is None else eta
    blk = rows.groupby(["Dataset", "Model", "Method", "scheme"])[["MAE", "track_r"]].mean().reset_index()
    orc = blk[blk["Method"] == "Oracle (fold accuracy)"].set_index(["Dataset", "Model", "scheme"])["MAE"]
    blk["excess"] = blk["MAE"] - blk.set_index(["Dataset", "Model", "scheme"]).index.map(orc).values

    # Table: MAE by scheme (mean over blocks), methods present in the data
    meths = [m for m in TABLE_METHODS if m in set(blk["Method"])]
    piv = blk[blk["Method"].isin(meths)].pivot_table(index="Method", columns="scheme", values="MAE")
    piv = piv.reindex(index=meths, columns=SCHEMES)
    best = piv.drop(index="Oracle (fold accuracy)", errors="ignore").min()
    lines = []
    for m in meths:
        cells = [(rf"\textbf{{{v:.3f}}}" if m != "Oracle (fold accuracy)" and np.isclose(v, best[s]) else f"{v:.3f}")
                 for s, v in piv.loc[m].items()]
        sep = "\\midrule\n" if m == "Ours (DR)" else ""
        lines.append(f"{sep}{METHOD_TEX[m]} & " + " & ".join(cells) + r" \\")
    head = " & ".join(SCHEME_TEX[s] for s in SCHEMES)
    _ensure = os.makedirs(f"{PAPER}/tables", exist_ok=True)
    with open(f"{PAPER}/tables/label_shift.tex", "w") as fh:
        fh.write(rf"""\begin{{tabular}}{{l{'c' * len(SCHEMES)}}}
\toprule
 & {head} \\
\midrule
{chr(10).join(lines)}
\bottomrule
\end{{tabular}}
""")

    # Figure: (a) excess error over the label-blind oracle by scheme; (b) per-detector excess at
    # k=1 against the class dependence of the detector's scores
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.7), gridspec_kw={"width_ratios": [1.6, 1]})
    show = [m for m in ["Ours (DR)", "NuclearNorm", "COT", "ATC-MC", "MDE"] if m in set(blk["Method"])]
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
    styles = ["-", "--", "-.", ":", (0, (5, 1))]
    for k, m in enumerate(show):
        y = blk[blk["Method"] == m].groupby("scheme")["excess"].mean().reindex(SCHEMES)
        ax.plot(range(len(SCHEMES)), y.values, color=colors[k], ls=styles[k], marker="o", ms=3.5, lw=1.5,
                label=METHOD_TEX[m])
    ax.axhline(0, color="#52514e", lw=0.8)
    ax.set_xticks(range(len(SCHEMES)))
    ax.set_xticklabels([SCHEME_TEX[s].replace("$", "").replace(r"\alpha{=}", "α=") for s in SCHEMES],
                       rotation=35, ha="right")
    ax.set_ylabel("MAE − label-blind oracle")
    ax.grid(color="#e4e3df", lw=0.6)
    ax.legend(frameon=False, fontsize=6.5)
    ax.set_title("(a) Excess error by batch composition", fontsize=8, loc="left")

    d = blk[blk["Method"].str.startswith("DR:") & (blk["scheme"].isin(["iid", "k1"]))]
    d = d.pivot_table(index=["Dataset", "Model", "Method"], columns="scheme", values="excess").reset_index()
    d["detector"] = d["Method"].str[3:]
    j = d.merge(eta, on=["Dataset", "Model", "detector"])
    j["delta"] = j["k1"] - j["iid"]
    for det, g in j.groupby("detector"):
        ax2.scatter(g["eta2"], g["delta"], s=12, alpha=0.7, label=det)
    rho, p = spearmanr(j["eta2"], j["delta"])
    ax2.set_xlabel("Class dependence of InD scores (η²)")
    ax2.set_ylabel("Δ excess error, 1 class vs. original")
    ax2.grid(color="#e4e3df", lw=0.6)
    ax2.set_title(f"(b) Detectors; Spearman ρ = {rho:.2f} (p = {p:.2g})", fontsize=8, loc="left")
    ax2.legend(frameon=False, fontsize=5.5, ncol=2)
    fig.tight_layout()
    os.makedirs(f"{PAPER}/figures", exist_ok=True)
    fig.savefig(f"{PAPER}/figures/label_shift.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote", f"{PAPER}/tables/label_shift.tex", f"{PAPER}/figures/label_shift.pdf")
    return piv
