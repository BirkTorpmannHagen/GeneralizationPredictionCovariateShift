"""
Online / streaming accuracy estimation from cached per-sample detector scores.

Motivation (revision pivot). Every prior unsupervised-accuracy-estimation (UAE)
method — ATC, MaNo, Nuclear-Norm, ODP-Bench — estimates a *single* accuracy
number for a *static* i.i.d. test set. The streaming / concept-drift literature
(Gama, STUDD, MD3, Podkopaev-Ramdas, Amoukou 2024) works on non-stationary
streams but emits a *drift alarm*, not a calibrated accuracy value (and
Podkopaev-Ramdas needs labels). The intersection — a continuously-updated,
calibrated accuracy estimate over a non-stationary unlabeled stream, driven by
aggregate detector statistics — is unoccupied.

This module tests the empirical core of that pivot from cache alone:

  1. Build a *non-stationary stream* by ordering cached per-sample scores into
     gradual / sudden / recurring / mixed drift schedules. Every detector signal
     is aligned on the SAME rows (join on fold+idx) so estimators are compared
     head-to-head on one stream.
  2. Calibrate a single linear map (detection-rate -> accuracy gap) on a pool of
     synthetic shift families, holding out the families used to build the stream
     (leave-shift-type-out -> honest generalization to unseen shift types).
  3. Slide a window, emit a per-window accuracy estimate, measure rolling MAE vs
     ground-truth rolling accuracy.

Unifying view: OOD detectors, classic anomaly detectors, and confidence
statistics are all plugged into the same windowed-DR estimator. If they track
rolling accuracy within a tight MAE band, online monitoring is a property of the
detector family, not one detector.

Prior-art baselines (estimate-vs-detect delta):
  * online-ATC   — native confidence estimator (fair confidence competitor).
  * online-CBPE  — NannyML's confidence-calibration estimate; assumes no drift.
  * Amoukou-2024 — sequential flagged-proportion; a drift ALARM, not a calibrated
                   accuracy curve (its proxy-as-accuracy is left uncalibrated).

Cache-only: no retraining, no re-inference. Deterministic (fixed seed).
"""
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["text.usetex"] = False

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.isotonic import IsotonicRegression

FIGDIR = "figures"
SEED = 0

# Synthetic families used ONLY for calibration (never appear in the stream).
CALIB_FAMILIES = ["noise", "multnoise", "saltpepper", "jpeg", "smear"]
# Held-out synthetic families used to BUILD the stream (unseen shift types).
STREAM_FAMILIES = ["brightness", "contrast", "fog", "hue"]

# Detector groups for the unifying-view band. Missing detectors are skipped.
DETECTOR_GROUPS = {
    "OOD-latent": ["mahalanobis", "vim", "react"],
    "OOD-logit/feature": ["energy", "grad_magnitude", "knn"],
    "Anomaly": ["isolation_forest", "lof", "ocsvm", "pca_recon"],
    "Confidence": ["msp", "cross_entropy"],
}
ALL_DETECTORS = [d for ds in DETECTOR_GROUPS.values() for d in ds]
GROUP_OF = {d: g for g, ds in DETECTOR_GROUPS.items() for d in ds}

_INTENSITIES = ["0.05", "0.1", "0.15000000000000002", "0.2", "0.25",
                "0.30000000000000004", "0.35000000000000003", "0.4", "0.45", "0.5"]


# ---------------------------------------------------------------------------
# Loading: one merged per-sample frame carrying every detector signal
# ---------------------------------------------------------------------------
def _read_detector(dataset, model, feature):
    prefix = f"data/{model}/feature_data"
    paths = [p for p in glob.glob(os.path.join(prefix, f"{dataset}_*_{feature}.csv"))
             if os.path.basename(p).endswith(f"_{feature}.csv")
             and os.path.basename(p).startswith(f"{dataset}_")]
    frames = []
    for p in sorted(paths):
        try:
            frames.append(pd.read_csv(p))
        except Exception:
            continue
    return pd.concat(frames, ignore_index=True) if frames else None


def load_merged(dataset, model):
    """Merged per-sample frame: one row per (fold, idx) with a column per detector.

    Columns: fold, idx, family, intensity, correct_prediction, ood, and
    ``feat_<detector>`` for every detector that was collected. None if the anchor
    detector is missing.
    """
    base = None
    cols = {}
    for det in ALL_DETECTORS:
        d = _read_detector(dataset, model, det)
        if d is None:
            continue
        d = d[["fold", "idx", "feature", "acc", "loss"]].drop_duplicates(["fold", "idx"])
        if base is None:
            base = d.rename(columns={"feature": f"feat_{det}"})
        else:
            base = base.merge(d[["fold", "idx", "feature"]]
                              .rename(columns={"feature": f"feat_{det}"}),
                              on=["fold", "idx"], how="inner")
        cols[det] = True
    if base is None:
        return None, []

    def fam(f):
        return f.split("_")[0] if "_0." in f else f

    def inten(f):
        return f.split("_")[1] if "_0." in f else None

    base["family"] = base["fold"].apply(fam)
    base["intensity"] = base["fold"].apply(inten)
    if dataset == "Polyp":
        base["correct_prediction"] = base["loss"] < 0.5
    else:
        base["correct_prediction"] = base["acc"] == 1
    base["ood"] = ~base["fold"].isin(["train", "ind_val", "ind_test"])
    return base, list(cols.keys())


# ---------------------------------------------------------------------------
# Per-detector calibration (id_quantile DR threshold + LOSO-pool linear map)
# ---------------------------------------------------------------------------
def _dr(values, thr, higher_is_ood):
    v = np.asarray(values)
    return float((v > thr).mean()) if higher_is_ood else float((v < thr).mean())


def _calibrate_detector(df, col, q=0.95):
    """Return dict of calibration state for one detector column, or None."""
    iv = df[df["fold"] == "ind_val"]
    ood = df[df["ood"]]
    if iv.empty or ood.empty:
        return None
    higher_is_ood = ood[col].mean() > iv[col].mean()
    thr = float(iv[col].quantile(q)) if higher_is_ood else float(iv[col].quantile(1 - q))
    ind_acc = float(iv["correct_prediction"].mean())
    pool = df[df["family"].isin(CALIB_FAMILIES)]
    recs = []
    for _, fdf in pool.groupby("fold"):
        recs.append((_dr(fdf[col].values, thr, higher_is_ood),
                     ind_acc - float(fdf["correct_prediction"].mean())))
    if len(recs) < 2:
        return None
    reg = LinearRegression().fit(np.array([[r[0]] for r in recs]),
                                 np.array([r[1] for r in recs]))
    # ATC threshold on this signal (confidence = sign*feature, higher=more correct)
    sign = -1.0 if higher_is_ood else 1.0
    conf = sign * iv[col].values
    err = 1.0 - ind_acc
    atc_tau = float(np.quantile(conf, err)) if 0 < err < 1 else float(conf.min() - 1)
    return {"thr": thr, "hio": higher_is_ood, "reg": reg, "ind_acc": ind_acc,
            "sign": sign, "atc_tau": atc_tau}


# ---------------------------------------------------------------------------
# Confidence-based prior-art estimators (computed from the MSP column)
# ---------------------------------------------------------------------------
def _cbpe_calibrator(df, col="feat_msp"):
    """NannyML CBPE: isotonic map from confidence -> P(correct), fit on ind_val.

    Window accuracy estimate = mean per-sample P(correct). Assumes calibration
    transfers (i.e. no concept drift) — degrades under covariate shift.
    """
    iv = df[df["fold"] == "ind_val"]
    if iv.empty:
        return None
    m = IsotonicRegression(out_of_bounds="clip", increasing=True)
    m.fit(iv[col].values, iv["correct_prediction"].values.astype(float))
    return m


# ---------------------------------------------------------------------------
# Stream construction
# ---------------------------------------------------------------------------
def _schedule(kind, families):
    hi, mid = _INTENSITIES[-1], _INTENSITIES[len(_INTENSITIES) // 2]
    if kind == "gradual":
        fam = families[0]
        up = [(fam, i) for i in _INTENSITIES]
        return [(fam, None)] * 2 + up + up[::-1] + [(fam, None)] * 2
    if kind == "sudden":
        seq = []
        for k in range(6):
            seq += [(families[k % len(families)], None)] * 3
            seq += [(families[k % len(families)], hi)] * 3
        return seq
    if kind == "recurring":
        fam = families[0]
        return [(fam, None), (fam, mid), (fam, hi), (fam, mid)] * 5
    if kind == "mixed":
        rng = np.random.default_rng(SEED)
        seq, lvl = [], 0
        for _ in range(60):
            fam = families[int(rng.integers(len(families)))]
            lvl = int(np.clip(lvl + int(rng.integers(-1, 2)), 0, len(_INTENSITIES) - 1))
            seq.append((fam, None if lvl == 0 else _INTENSITIES[lvl]))
        return seq
    raise ValueError(kind)


def build_stream(df, kind, families=STREAM_FAMILIES, seg_len=150, seed=SEED):
    rng = np.random.default_rng(seed)
    parts = []
    for fam, inten in _schedule(kind, families):
        pool = df[df["fold"] == "ind_test"] if inten is None \
            else df[(df["family"] == fam) & (df["intensity"] == inten)]
        if pool.empty:
            continue
        idx = rng.choice(pool.index.values, size=seg_len, replace=len(pool) < seg_len)
        parts.append(df.loc[idx])
    return pd.concat(parts, ignore_index=True) if parts else None


# ---------------------------------------------------------------------------
# Evaluation over a stream: all estimators on the same windows
# ---------------------------------------------------------------------------
def evaluate_stream(stream, dets, cal, cbpe, W=200, S=50):
    """Return per-window true accuracy + every estimator's estimate series.

    ``cal`` maps detector -> calibration dict; ``cbpe`` is the CBPE isotonic map.
    Estimators: DR-map + ATC per detector; CBPE and Amoukou-proportion from MSP.
    """
    n = len(stream)
    correct = stream["correct_prediction"].values.astype(float)
    starts = list(range(0, max(1, n - W + 1), S))
    true_acc = np.array([correct[s:s + W].mean() for s in starts])
    centers = np.array([s + W // 2 for s in starts])

    est = {}  # name -> np.array of per-window estimates
    for det in dets:
        c = cal.get(det)
        if c is None:
            continue
        f = stream[f"feat_{det}"].values
        dr_pred, atc_pred = [], []
        for s in starts:
            w = f[s:s + W]
            dr = _dr(w, c["thr"], c["hio"])
            dr_pred.append(np.clip(c["ind_acc"] - c["reg"].predict([[dr]])[0], 0, 1))
            atc_pred.append(float((c["sign"] * w >= c["atc_tau"]).mean()))
        est[f"DR:{det}"] = np.array(dr_pred)
        est[f"ATC:{det}"] = np.array(atc_pred)

    # Confidence prior-art (from MSP): CBPE + Amoukou flagged-proportion
    if "feat_msp" in stream.columns:
        fm = stream["feat_msp"].values
        cm = cal.get("msp")
        if cbpe is not None:
            psample = cbpe.predict(fm)
            est["CBPE"] = np.array([psample[s:s + W].mean() for s in starts])
        if cm is not None:
            # Amoukou: running proportion flagged as high-error at the ATC proxy;
            # read as accuracy = 1 - flagged proportion (deliberately UNcalibrated).
            est["Amoukou-prop"] = np.array(
                [float((cm["sign"] * fm[s:s + W] >= cm["atc_tau"]).mean()) for s in starts])
    return centers, true_acc, est


def _mae(true_acc, series):
    return float(np.mean(np.abs(true_acc - series)))


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run(datasets=("CCT", "OfficeHome", "Office31", "NICO", "Camelyon17", "IWildCam"),
        models=("resnet", "vit"),
        schedules=("gradual", "sudden", "recurring", "mixed"),
        q=0.95, W=200, S=50, seg_len=150, make_figure=True):
    rows = []
    demo = None
    for dataset in datasets:
        for model in models:
            df, dets = load_merged(dataset, model)
            if df is None or not dets:
                continue
            cal = {d: _calibrate_detector(df, f"feat_{d}", q) for d in dets}
            cbpe = _cbpe_calibrator(df) if "feat_msp" in df.columns else None
            for sched in schedules:
                stream = build_stream(df, sched, seg_len=seg_len)
                if stream is None or len(stream) < W:
                    continue
                centers, true_acc, est = evaluate_stream(stream, dets, cal, cbpe, W, S)
                for name, series in est.items():
                    kind, _, det = name.partition(":")
                    rows.append({
                        "Dataset": dataset, "Model": model, "Schedule": sched,
                        "Estimator": kind, "Detector": det or "-",
                        "Group": GROUP_OF.get(det, kind),
                        "MAE": _mae(true_acc, series),
                    })
                if demo is None and dataset == "CCT" and model == "resnet":
                    demo = (dataset, model, df, dets, cal, cbpe, W, S, seg_len)
    out = pd.DataFrame(rows)
    os.makedirs(FIGDIR, exist_ok=True)
    out.to_csv(f"{FIGDIR}/streaming_eval.csv", index=False)
    _summaries(out)
    if make_figure and demo is not None:
        _drift_figure(*demo)
    return out


def _summaries(out):
    if out.empty:
        print("no streaming rows produced")
        return
    dr = out[out["Estimator"] == "DR"]
    print("\n=== Unifying-view band: rolling MAE per detector group (DR estimator) ===")
    print(dr.groupby("Group")["MAE"].agg(["mean", "std", "min", "count"]).to_string())
    print("\n=== Best rolling MAE per dataset: our DR vs prior-art baselines ===")
    piv = {}
    piv["Ours (best DR detector)"] = dr.groupby("Dataset")["MAE"].min()
    for est in ["ATC", "CBPE", "Amoukou-prop"]:
        sub = out[out["Estimator"] == est]
        if not sub.empty:
            label = {"ATC": "online-ATC (best)", "CBPE": "online-CBPE",
                     "Amoukou-prop": "Amoukou-prop (uncal.)"}[est]
            piv[label] = sub.groupby("Dataset")["MAE"].min()
    print(pd.DataFrame(piv).round(4).to_string())


def _drift_figure(dataset, model, df, dets, cal, cbpe, W, S, seg_len):
    """2x2 grid: true rolling accuracy vs our estimate, CBPE, Amoukou-prop."""
    # representative detector = best-calibrated OOD-latent available
    pref = [d for d in ["vim", "mahalanobis", "knn", "energy"] if cal.get(d)]
    if not pref:
        return
    det = pref[0]
    scheds = ["gradual", "sudden", "recurring", "mixed"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 6), sharey=True)
    for ax, sched in zip(axes.ravel(), scheds):
        stream = build_stream(df, sched, seg_len=seg_len)
        if stream is None:
            continue
        centers, true_acc, est = evaluate_stream(stream, [det], cal, cbpe, W, S)
        ax.plot(centers, true_acc, "k-", lw=2.2, label="true accuracy")
        ax.plot(centers, est[f"DR:{det}"], "C1--", lw=1.8,
                label=f"ours ({det} DR)  MAE={_mae(true_acc, est[f'DR:{det}']):.3f}")
        if "CBPE" in est:
            ax.plot(centers, est["CBPE"], "C0:", lw=1.8,
                    label=f"CBPE  MAE={_mae(true_acc, est['CBPE']):.3f}")
        if "Amoukou-prop" in est:
            ax.plot(centers, est["Amoukou-prop"], "C2-.", lw=1.3,
                    label=f"Amoukou-prop  MAE={_mae(true_acc, est['Amoukou-prop']):.3f}")
        ax.set_title(f"{sched} drift", fontsize=10)
        ax.set_ylim(0, 1)
        ax.legend(loc="lower left", fontsize=7)
        ax.set_xlabel("stream position")
    axes[0, 0].set_ylabel("accuracy")
    axes[1, 0].set_ylabel("accuracy")
    fig.suptitle(f"Online accuracy tracking under non-stationary drift "
                 f"({dataset}/{model})", fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/streaming_drift_tracking.pdf")
    plt.close(fig)


if __name__ == "__main__":
    run()
