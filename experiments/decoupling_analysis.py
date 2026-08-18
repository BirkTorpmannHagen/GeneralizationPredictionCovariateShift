"""
Feasibility probe for the adversarial-monitoring follow-up (SECURITY_FOLLOWUP.md).

The whole follow-up paper hinges on ONE empirical question (the "knife-edge"):

  When the model is WRONG, do the different detector signals fire *independently*
  or *together* — and are they all explained by a single confidence factor?

  * Correlated / confidence-reducible  -> monitor-evasion ~= misclassification;
    ensembling adds nothing; the paper is a restatement of "attacks work". REJECT.
  * Genuinely decoupled                 -> a standard (confidence-fooling) attack
    stays off-manifold and trips the feature-space detectors; a joint attack is
    needed and is much harder -> negative-transfer result -> a real paper.

This is cache-only (per-sample detector scores). It does NOT build attacks; it
tells us whether building them is worthwhile. Metrics, per (dataset, model):

  A. Pairwise Spearman correlation of oriented per-sample detector responses,
     conditioned on MISCLASSIFIED shifted samples. Low |corr| => decoupled.
  B. PCA on those responses: variance explained by PC1 (the single latent
     factor). High PC1 => one factor (likely confidence) => redundant.
  C. Confidence-reducibility: R^2 of predicting each detector from MSP+Energy.
     Low R^2 => the detector carries information beyond confidence.
  D. THE KNIFE-EDGE: among "confidently wrong" samples (misclassified AND NOT
     flagged by MSP), what fraction are still flagged OOD by a feature-space
     detector (Mahalanobis / kNN)? High => standard attacks won't transfer =>
     the joint attack is real.

Run: python -m experiments.decoupling_analysis   (writes figures/decoupling_*.csv)
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

from experiments.streaming_eval import load_merged

FIGDIR = "figures"
CONF_DETECTORS = ["msp", "cross_entropy", "energy"]      # confidence / logit family
FEATURE_DETECTORS = ["mahalanobis", "knn", "vim", "react"]  # feature/density family


def _orient_standardize(merged, dets):
    """Add z_<det> columns oriented so higher = more anomalous, standardized on ind_val."""
    iv = merged[merged["fold"] == "ind_val"]
    ood = merged[merged["ood"]]
    zcols = []
    for d in dets:
        c = f"feat_{d}"
        sign = 1.0 if ood[c].mean() > iv[c].mean() else -1.0
        ref = sign * iv[c]
        mu, sd = float(ref.mean()), float(ref.std()) + 1e-9
        merged[f"z_{d}"] = (sign * merged[c] - mu) / sd
        zcols.append(f"z_{d}")
    return zcols


def _flag_threshold(merged, det, q=0.95):
    """95th-percentile ind_val threshold on the oriented z score (higher=OOD)."""
    iv = merged[merged["fold"] == "ind_val"]
    return float(iv[f"z_{det}"].quantile(q))


def analyze(dataset, model):
    merged, dets = load_merged(dataset, model)
    if merged is None or len(dets) < 3:
        return None
    _orient_standardize(merged, dets)

    mis = merged[(~merged["correct_prediction"]) & merged["ood"]]
    if len(mis) < 200:
        return None
    zcols = [f"z_{d}" for d in dets]
    Z = mis[zcols].replace([np.inf, -np.inf], np.nan).dropna()
    if len(Z) < 200:
        return None

    # A. mean |pairwise Spearman| among detectors, and confidence-vs-feature block
    rho, _ = spearmanr(Z.values)
    rho = np.atleast_2d(rho)
    off = rho[~np.eye(len(zcols), dtype=bool)]
    mean_abs_corr = float(np.mean(np.abs(off)))
    # cross-family mean |corr| (confidence family vs feature family)
    cf = [d for d in dets if d in CONF_DETECTORS]
    ff = [d for d in dets if d in FEATURE_DETECTORS]
    cross = []
    for a in cf:
        for b in ff:
            r, _ = spearmanr(Z[f"z_{a}"], Z[f"z_{b}"])
            cross.append(abs(r))
    cross_corr = float(np.mean(cross)) if cross else np.nan

    # B. PCA variance explained by PC1 / PC1+PC2 (standardized)
    Zs = (Z - Z.mean()) / (Z.std() + 1e-9)
    p = PCA().fit(Zs.values)
    pc1 = float(p.explained_variance_ratio_[0])
    pc2 = float(p.explained_variance_ratio_[:2].sum())

    # C. confidence-reducibility: R^2 predicting each FEATURE detector from confidence family
    confX = mis[[f"z_{d}" for d in cf]].values if cf else None
    red = {}
    if confX is not None and len(cf) >= 1:
        for d in ff:
            y = mis[f"z_{d}"].values
            m = np.isfinite(y) & np.all(np.isfinite(confX), axis=1)
            if m.sum() > 100:
                r2 = LinearRegression().fit(confX[m], y[m]).score(confX[m], y[m])
                red[d] = float(r2)
    feat_r2 = float(np.mean(list(red.values()))) if red else np.nan

    # D. knife-edge: confidently-wrong = misclassified & NOT flagged by MSP.
    #    fraction still caught by a feature-space detector.
    knife = {}
    if "msp" in dets:
        msp_thr = _flag_threshold(merged, "msp")
        conf_wrong = mis[mis["z_msp"] <= msp_thr]  # MSP thinks in-distribution
        knife["n_confidently_wrong"] = int(len(conf_wrong))
        knife["frac_of_errors_conf_wrong"] = float(len(conf_wrong) / max(1, len(mis)))
        for d in ff:
            if d in dets and len(conf_wrong):
                thr = _flag_threshold(merged, d)
                knife[f"caught_by_{d}"] = float((conf_wrong[f"z_{d}"] > thr).mean())

    return {
        "Dataset": dataset, "Model": model, "n_errors": int(len(mis)),
        "mean_abs_corr": mean_abs_corr, "cross_family_corr": cross_corr,
        "PC1_var": pc1, "PC1_2_var": pc2, "feature_R2_from_confidence": feat_r2,
        **knife,
    }


def run(datasets=("CCT", "OfficeHome", "Office31", "NICO", "Camelyon17", "IWildCam"),
        models=("resnet", "vit")):
    rows = []
    for d in datasets:
        for m in models:
            try:
                r = analyze(d, m)
            except Exception as e:
                print(f"[skip] {d}/{m}: {e}")
                r = None
            if r:
                rows.append(r)
                print(f"{d}/{m}: errors={r['n_errors']} "
                      f"|corr|={r['mean_abs_corr']:.2f} cross={r['cross_family_corr']:.2f} "
                      f"PC1={r['PC1_var']:.2f} featR2|conf={r['feature_R2_from_confidence']:.2f} "
                      f"conf-wrong={r.get('frac_of_errors_conf_wrong', float('nan')):.2f} "
                      f"maha-catch={r.get('caught_by_mahalanobis', float('nan')):.2f} "
                      f"knn-catch={r.get('caught_by_knn', float('nan')):.2f}")
    out = pd.DataFrame(rows)
    os.makedirs(FIGDIR, exist_ok=True)
    out.to_csv(f"{FIGDIR}/decoupling_analysis.csv", index=False)
    if not out.empty:
        print("\n=== DECOUPLING SUMMARY (means over datasets/models) ===")
        for c in ["mean_abs_corr", "cross_family_corr", "PC1_var",
                  "feature_R2_from_confidence", "frac_of_errors_conf_wrong",
                  "caught_by_mahalanobis", "caught_by_knn"]:
            if c in out.columns:
                print(f"  {c:32s} {out[c].mean():.3f}")
        print("\nReading: low cross-family corr + low PC1 + low feature_R2|confidence +"
              " HIGH maha/knn catch-rate on confidently-wrong => DECOUPLED => follow-up"
              " is a real paper. The opposite => restatement.")
    return out


if __name__ == "__main__":
    run()
