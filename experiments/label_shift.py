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
from experiments.ablations import dr_calibration, mde_statistic
from utils import DSD_PRINT_LUT, DSDS, FEATURE_COLUMN_ALIAS, load_data

FIG = "figures"
CACHE = f"{FIG}/perfold_tables.pkl"
SOFTMAX_DIR = "data/softmax"
FEATURE_DIR = "data/features"
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
        # orientation + ind_val 0.95-quantile threshold, exactly as in ablations._perfold_dr
        cal = dr_calibration(raw, 0.95)
        if cal is None:
            continue
        hi, thr = cal
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
    """{fold: (P, y, idx, Z, Hp)}; logits Z and projected features Hp are None when absent."""
    d = os.path.join(SOFTMAX_DIR, model, ds)
    if not os.path.isdir(d):
        return None
    fdir = os.path.join(FEATURE_DIR, model, ds)
    out = {}
    for fn in os.listdir(d):
        if fn.endswith(".npz"):
            z = np.load(os.path.join(d, fn))
            Z = z["z"].astype(np.float32) if "z" in z.files else None
            hp = os.path.join(fdir, fn)
            Hp = np.load(hp)["h"].astype(np.float32) if os.path.exists(hp) else None
            out[fn[:-4]] = (z["p"].astype(np.float32), z["y"], z["idx"], Z, Hp)
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
                pred = np.clip(np.asarray(fn(b), dtype=float), 0.0, 1.0)
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
    from uae_stats import cot_error, dispersion, mano, nuclear_norm
    Pv, yv = sm["ind_val"][0], sm["ind_val"][1]
    ind_acc = float((Pv.argmax(1) == yv).mean())
    K = Pv.shape[1]
    pi = np.bincount(yv, minlength=K).astype(float)
    pi /= pi.sum()
    ref = Pv.mean(0)
    # Each statistic maps (P, Z, Hp) of one batch to a scalar; Z = logits, Hp = projected features.
    stats = {"NuclearNorm": lambda P, Z, H: nuclear_norm(P),
             "COT": lambda P, Z, H: cot_error(P, pi, rng),
             "LabelShiftL1": lambda P, Z, H: float(np.abs(P.mean(0) - ref).sum())}
    if all(v[3] is not None for v in sm.values()):
        stats["MaNo"] = lambda P, Z, H: mano(Z)
    if all(v[4] is not None for v in sm.values()):
        stats["Dispersion"] = lambda P, Z, H: dispersion(H, P.argmax(1), K=K)

    def ev(fn, P, Z, H, i):
        return fn(P[i], None if Z is None else Z[i], None if H is None else H[i])

    # Calibrate at the batch size the statistics are evaluated at. Nuclear Norm and COT depend
    # on the number of samples (the normalized nuclear norm through min(N, C); COT through the
    # granularity of the empirical label marginal; Dispersion through its cluster-size weights),
    # so a map fitted on ~2000-sample folds would be misapplied to N_BATCH-sample batches. Each
    # fold's calibration statistic is therefore the mean over N_REP random N_BATCH-sample subsets
    # (uniform, i.e. original class mix), as for MDE.
    cal_rng = np.random.default_rng(SEED)
    recs = []
    for f, (P, y, _, Z, H) in sm.items():
        st = _parse_shift_type_from_fold(f)
        subs = [cal_rng.integers(0, len(P), N_BATCH) for _ in range(N_REP)]
        recs.append({"fold": f, "shift_type": st, "category": "Synthetic" if _is_synthetic_shift_type(st) else "Organic",
                     **{k: float(np.mean([ev(fn, P, Z, H, i) for i in subs])) for k, fn in stats.items()},
                     "gap": ind_acc - float((P.argmax(1) == y).mean())})
    pf = pd.DataFrame(recs)
    pool_all = pf[pf["category"] == "Synthetic"]
    rows = []
    for f, (P, y, _, Z, H) in sm.items():
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
                s = np.array([ev(fn, P, Z, H, i) for i in b]).reshape(-1, 1)
                pred = np.clip(ind_acc - regs[k].predict(s).ravel(), 0.0, 1.0)
                err = pred - true_acc
                rows.append({"Dataset": ds, "Model": model, "Method": k, "scheme": scheme, "fold": f,
                             "shift_type": cat if cat != "Synthetic" else st, "category": cat,
                             "MAE": float(np.abs(err).mean()), "bias": float(err.mean()),
                             "track_r": _corr(pred.ravel(), true_acc),
                             "true_gap": float(ind_acc - true_acc.mean())})
            if "COT" in stats:  # native COT: predicted accuracy = 1 - transport cost
                pred = np.clip(1.0 - np.array([ev(stats["COT"], P, Z, H, i) for i in b]), 0.0, 1.0)
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
TABLE_METHODS = ["Oracle (fold accuracy)", "Ours (DR)", "Ours (mean)", "MaNo", "Dispersion", "NuclearNorm", "COT",
                 "COT (native)", "LabelShiftL1", "ATC (regressed)", "ATC-MC", "MDE", "DoC", "AC"]
METHOD_TEX = {"Oracle (fold accuracy)": "Label-blind oracle", "Ours (DR)": "Ours (DR)",
              "Ours (mean)": "Ours (mean score)", "NuclearNorm": "Nuclear Norm", "COT": "COT",
              "MaNo": "MaNo", "Dispersion": "Dispersion",
              "COT (native)": "COT (native)", "LabelShiftL1": "Label-marginal $L_1$",
              "ATC (regressed)": "ATC (regressed)", "ATC-MC": "ATC-MC",
              "MDE": "MDE", "DoC": "DoC", "AC": "AC"}
OURS = "Ours (DR)"
COMPARE = ["MaNo", "Dispersion", "NuclearNorm", "COT", "COT (native)", "LabelShiftL1", "ATC (regressed)", "ATC-MC", "MDE", "DoC", "AC"]
BLOCK = ["Dataset", "Model"]
FINE_BLOCK = ["Dataset", "Model", "shift_type"]


def label_shift_tests(rows=None, alpha=0.05):
    """Per batch composition: two-sided Wilcoxon signed-rank tests of our estimator against each
    method on block-mean MAE, Holm-corrected over the comparisons against ours (as in the main
    analysis). Two designs: (Dataset, Model) blocks, and (Dataset, Model, shift family) blocks
    as a sensitivity analysis (not independent within a dataset)."""
    from scipy.stats import wilcoxon
    rows = pd.read_csv(f"{FIG}/label_shift_rows.csv") if rows is None else rows
    out = []
    for design, block in (("block", BLOCK), ("fine", FINE_BLOCK)):
        for sch in SCHEMES:
            sc = rows[rows["scheme"] == sch].groupby(block + ["Method"])["MAE"].mean().unstack()
            res = []
            for m in [c for c in COMPARE if c in sc.columns]:
                d = (sc[OURS] - sc[m]).dropna()
                p = float(wilcoxon(d).pvalue) if (d != 0).any() else 1.0
                res.append({"design": design, "scheme": sch, "method": m, "n_blocks": len(d),
                            "mae_ours": float(sc[OURS].mean()), "mae_method": float(sc[m].mean()),
                            "ours_wins": int((d < 0).sum()), "p": p})
            r = pd.DataFrame(res).sort_values("p").reset_index(drop=True)
            k = len(r)
            r["p_holm"] = np.maximum.accumulate([min(1.0, (k - i) * p) for i, p in enumerate(r["p"])])
            r["significant"] = r["p_holm"] < alpha
            out.append(r)
    out = pd.concat(out, ignore_index=True)
    out.to_csv(f"{FIG}/label_shift_tests.csv", index=False)
    return out


def paper_assets(rows=None):
    import matplotlib.pyplot as plt
    rows = pd.read_csv(f"{FIG}/label_shift_rows.csv") if rows is None else rows
    tests = label_shift_tests(rows)
    sig = tests[(tests["design"] == "block") & tests["significant"]
                & (tests["mae_ours"] < tests["mae_method"])].set_index(["method", "scheme"]).index
    blk = rows.groupby(["Dataset", "Model", "Method", "scheme"])["MAE"].mean().reset_index()

    # Table: MAE by batch composition (mean over blocks). Dagger: ours significantly better.
    meths = [m for m in TABLE_METHODS if m in set(blk["Method"])]
    piv = blk[blk["Method"].isin(meths)].pivot_table(index="Method", columns="scheme", values="MAE")
    piv = piv.reindex(index=meths, columns=SCHEMES)
    best = piv.drop(index="Oracle (fold accuracy)", errors="ignore").min()
    lines = []
    for m in meths:
        cells = []
        for sch, v in piv.loc[m].items():
            c = rf"\textbf{{{v:.3f}}}" if m != "Oracle (fold accuracy)" and np.isclose(v, best[sch]) else f"{v:.3f}"
            cells.append(c + (r"$^\dagger$" if (m, sch) in sig else ""))
        sep = "\\midrule\n" if m in ("Ours (DR)", next((x for x in ("MaNo", "Dispersion", "NuclearNorm") if x in meths), None)) else ""
        lines.append(f"{sep}{METHOD_TEX[m]} & " + " & ".join(cells) + r" \\")
    headline = " & ".join(SCHEME_TEX[s] for s in SCHEMES)
    os.makedirs(f"{PAPER}/tables", exist_ok=True)
    with open(f"{PAPER}/tables/label_shift.tex", "w") as fh:
        fh.write(rf"""\begin{{tabular}}{{l{'c' * len(SCHEMES)}}}
\toprule
 & {headline} \\
\midrule
{chr(10).join(lines)}
\bottomrule
\end{{tabular}}
""")

    # Figure: (a) MAE by batch composition; (b) per detector, original mix vs. single-class batches
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.7), gridspec_kw={"width_ratios": [1.6, 1]})
    show = [m for m in ["Ours (DR)", "NuclearNorm", "COT", "MaNo", "Dispersion", "ATC-MC", "MDE"]
            if m in set(blk["Method"])]
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#8a5cd6", "#3fb3c4", "#eda100", "#e87ba4"]
    styles = ["-", "--", "-.", (0, (3, 1, 1, 1)), (0, (1, 1)), ":", (0, (5, 1))]
    x = range(len(SCHEMES))
    orc = blk[blk["Method"] == "Oracle (fold accuracy)"].groupby("scheme")["MAE"].mean().reindex(SCHEMES)
    ax.fill_between(x, 0, orc.values, color="#e4e3df", lw=0, label="Label-blind oracle")
    for k, m in enumerate(show):
        y = blk[blk["Method"] == m].groupby("scheme")["MAE"].mean().reindex(SCHEMES)
        ax.plot(x, y.values, color=colors[k], ls=styles[k], marker="o", ms=3.5, lw=1.5,
                label=METHOD_TEX[m])
    ax.set_xticks(list(x))
    ax.set_xticklabels([SCHEME_TEX[s].replace("$", "").replace(r"\alpha{=}", "α=") for s in SCHEMES],
                       rotation=35, ha="right")
    ax.set_ylabel("MAE")
    ax.set_ylim(0, None)
    ax.grid(color="#e4e3df", lw=0.6)
    ax.legend(frameon=False, fontsize=6.5)
    ax.set_title("(a) Accuracy-estimation error by batch composition", fontsize=8, loc="left")

    d = blk[blk["Method"].str.startswith("DR:") & blk["scheme"].isin(["iid", "k1"])]
    d = d.groupby(["Method", "scheme"])["MAE"].mean().unstack().sort_values("k1")
    o = blk[blk["Method"] == "Oracle (fold accuracy)"].groupby("scheme")["MAE"].mean()
    d = pd.concat([pd.DataFrame({"iid": [o["iid"]], "k1": [o["k1"]]}, index=["DR:Label-blind oracle"]), d])
    feat = {"kNN", "Mahalanobis", "ViM"}
    for i, (m, r) in enumerate(d.iterrows()):
        col = "#9a9890" if "oracle" in m else ("#2a78d6" if m[3:] in feat else "#eb6834")
        ax2.plot([r["iid"], r["k1"]], [i, i], color=col, lw=1.4)
        ax2.scatter([r["iid"]], [i], color="white", edgecolor=col, s=18, zorder=3)
        ax2.scatter([r["k1"]], [i], color=col, s=18, zorder=3)
    ax2.set_yticks(range(len(d)))
    ax2.set_yticklabels([m[3:].replace("UniformCE", "Uniform-CE") for m in d.index])
    ax2.set_xlabel("MAE: ○ original mix, ● one class")
    ax2.grid(color="#e4e3df", lw=0.6, axis="x")
    ax2.set_title("(b) Per detector (DR)", fontsize=8, loc="left")
    fig.tight_layout()
    os.makedirs(f"{PAPER}/figures", exist_ok=True)
    fig.savefig(f"{PAPER}/figures/label_shift.pdf", bbox_inches="tight")
    fig.savefig(f"{FIG}/label_shift.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("wrote", f"{PAPER}/tables/label_shift.tex", f"{PAPER}/figures/label_shift.pdf")
    return piv, tests
