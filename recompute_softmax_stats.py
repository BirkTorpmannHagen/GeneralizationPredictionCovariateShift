"""Recompute figures/softmax_stats.csv and figures/softmax_stats_subsets.csv from the
per-sample softmax dumps (data/softmax/<model>/<dataset>/<fold>.npz), without a GPU.

Same statistics as collect_softmax_stats.py. Datasets without dumps keep their rows from the
existing softmax_stats.csv (older collector: "cot" is the label-marginal L1 distance).
"""
import glob
import os

import numpy as np
import pandas as pd

from uae_stats import cot_error, dispersion, mano, nuclear_norm

SOFTMAX_DIR = "data/softmax"
FEATURE_DIR = "data/features"
N_SUBSETS, SUBSET_SIZE, SEED = 50, 100, 0


def _stats(P, y, pi, ref_mean, rng, Z=None, Hp=None):
    out = {"true_acc": float((P.argmax(1) == y).mean()), "ac": float(P.max(1).mean()),
           "nuclear_norm": nuclear_norm(P), "cot_err": cot_error(P, pi, rng),
           "label_shift_l1": float(np.abs(P.mean(0) - ref_mean).sum()), "N": int(P.shape[0])}
    if Z is not None:
        out.update(mano=mano(Z), mano_softmax=mano(Z, mode="softmax"), mano_taylor=mano(Z, mode="taylor"))
    if Hp is not None:  # exact Dispersion needs the full features and is computed on the GPU machine
        out["dispersion_proj"] = dispersion(Hp, P.argmax(1), K=P.shape[1])
    return out


def recompute(out="figures/softmax_stats.csv", out_sub="figures/softmax_stats_subsets.csv", force=False):
    """Offline fallback for when the GPU machine's CSVs were not copied back. The GPU collector also
    computes the exact Dispersion score on full features; this function cannot (it only has the
    projected features), so it refuses to overwrite a CSV that already has it unless force=True."""
    if os.path.exists(out) and "dispersion" in pd.read_csv(out, nrows=1).columns and not force:
        raise RuntimeError(f"{out} has exact Dispersion from the GPU collector; not overwriting "
                           f"(pass force=True to recompute anyway)")
    rng = np.random.default_rng(SEED)
    rows, sub_rows, done = [], [], set()
    for d in sorted(glob.glob(f"{SOFTMAX_DIR}/*/*")):
        model, ds = d.split("/")[-2:]
        files = {os.path.basename(f)[:-4]: f for f in glob.glob(f"{d}/*.npz")}
        if "ind_val" not in files:
            print(f"[skip] {ds}/{model}: no ind_val dump")
            continue
        fdir = f"{FEATURE_DIR}/{model}/{ds}"

        def load(f):
            z = np.load(files[f])
            Z = z["z"].astype(np.float64) if "z" in z.files else None
            hp = os.path.join(fdir, f"{f}.npz")
            Hp = np.load(hp)["h"].astype(np.float64) if os.path.exists(hp) else None
            return z["p"].astype(np.float64), z["y"].astype(int), Z, Hp

        Pv, yv, _, _ = load("ind_val")
        pi = np.bincount(yv, minlength=Pv.shape[1]).astype(float)
        pi /= pi.sum()
        ref = Pv.mean(0)
        for fold in ["ind_val"] + sorted(f for f in files if f != "ind_val"):
            P, y, Z, Hp = load(fold)
            rows.append({"Dataset": ds, "Model": model, "fold": fold, **_stats(P, y, pi, ref, rng, Z, Hp)})
            n = min(SUBSET_SIZE, len(P))
            for b in range(N_SUBSETS):
                i = rng.choice(len(P), size=n, replace=False)
                sub_rows.append({"Dataset": ds, "Model": model, "fold": fold, "subset": b,
                                 **_stats(P[i], y[i], pi, ref, rng, None if Z is None else Z[i],
                                          None if Hp is None else Hp[i])})
        done.add((ds, model))
        print(f"[softmax] {ds}/{model}: {len(files)} folds", flush=True)
    new = pd.DataFrame(rows)
    if os.path.exists(out):  # keep datasets that have no dumps yet (e.g. NICO)
        old = pd.read_csv(out)
        old = old[[(d, m) not in done for d, m in zip(old["Dataset"], old["Model"])]]
        if "cot" in old.columns and "label_shift_l1" not in old.columns:
            old = old.rename(columns={"cot": "label_shift_l1"})
        new = pd.concat([new, old], ignore_index=True)
    new.to_csv(out, index=False)
    pd.DataFrame(sub_rows).to_csv(out_sub, index=False)
    return new


if __name__ == "__main__":
    recompute()
