"""Per-fold softmax-matrix statistics for the batch-level UAE baselines.

Run on the GPU machine (needs the trained classifiers + datasets). Writes two files, plus the
per-sample softmax matrices under data/softmax/<model>/<dataset>/<fold>.npz (float16, capped at
SAMPLE_CAP per fold) used by experiments/label_shift.py:

  figures/softmax_stats.csv          one row per (dataset, model, fold), computed on up to
                                     SAMPLE_CAP samples of the fold
  figures/softmax_stats_subsets.csv  one row per bootstrap subset (N_SUBSETS per fold, each
                                     SUBSET_SIZE samples), used to score the batch statistics
                                     as set-level shift detectors and at deployment batch sizes

Statistics (P = [N, C] softmax matrix of the fold, y_hat = argmax):
  * nuclear_norm : Deng et al., CVPR 2023.  ||P||_* / sqrt(min(N, C) * N)  (their Eq. for
                   the normalized nuclear norm; higher => higher accuracy).
  * cot_err      : Lu et al., NeurIPS 2023 (Confidence Optimal Transport).  The predicted
                   error is the optimal-transport cost between the empirical distribution of
                   the fold's softmax vectors and one-hot labels drawn from the source (InD-val)
                   label distribution, with cost c(p, e_y) = ||p - e_y||_inf.  With equal sample
                   counts this is an assignment problem, solved exactly.
  * label_shift_l1 : L1 distance between the fold's mean prediction and the InD-val mean
                   prediction.  This is what earlier versions of this script called "cot";
                   it is NOT COT and is kept only for continuity.
  * ac           : average max-softmax confidence (AC).
  * mano         : MaNo, Xie et al., NeurIPS 2024 (p = 4, eta = 5, normalization chosen per batch);
                   mano_softmax / mano_taylor with the normalization held fixed (diagnostics).
  * dispersion   : Dispersion score, Xie et al., NeurIPS 2023, on the full penultimate features
                   (the classifier-head input) clustered by predicted class.
  * dispersion_proj : the same on the stored 256-d random projection of the features, to check the
                   approximation used by the offline label-shift test.
  * true_acc     : accuracy on the same samples.

Per-sample files (for offline re-batching):
  data/softmax/<model>/<dataset>/<fold>.npz   p (softmax), z (logits), y, idx
  data/features/<model>/<dataset>/<fold>.npz  h (head-input features, projected to PROJ_DIM)
"""
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from eval_detectors import TESTBEDS
from features import _shared
from uae_stats import cot_error, dispersion, mano, nuclear_norm, projection_matrix
from utils import SYNTHETIC_SHIFTS

CLASSIF = ["CCT", "OfficeHome", "Office31", "NICO", "Camelyon17", "IWildCam"]
MODELS = ["resnet", "vit"]
SAMPLE_CAP = 2000
N_SUBSETS = 50
SUBSET_SIZE = 100
BATCH = 16
SEED = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT = "figures/softmax_stats.csv"
OUT_SUB = "figures/softmax_stats_subsets.csv"
SOFTMAX_DIR = "data/softmax"  # data/softmax/<model>/<dataset>/<fold>.npz
FEATURE_DIR = "data/features"  # data/features/<model>/<dataset>/<fold>.npz
PROJ_DIM = 256
_PROJ = {}


def _projection(d):
    if d not in _PROJ:
        _PROJ[d] = projection_matrix(d, PROJ_DIM, SEED)
    return _PROJ[d]


def _fold_matrix(net, loader):
    """Whole fold, then a seeded random subsample of SAMPLE_CAP samples. (Evaluation loaders are
    deliberately unshuffled, see testbeds/base.py, so taking the first SAMPLE_CAP samples would
    keep only the first few classes.) Returns softmax P, logits Z, head-input features H, labels
    y and dataset indices idx."""
    Ps, Zs, Hs, ys, ids = [], [], [], [], []
    for data in loader:
        x = data[0].to(DEVICE)
        s = _shared(net, x)  # one forward pass: logits + input of the final linear layer
        logits, h = s["logits"].float(), s["h"]
        Zs.append(logits.cpu().numpy())
        Ps.append(F.softmax(logits, dim=1).cpu().numpy())
        Hs.append(h.float().cpu().numpy() if h is not None else np.zeros((x.shape[0], 0), np.float32))
        ys.append(data[1].cpu().numpy())
        ids.append(np.asarray(data[2]) if len(data) > 2 else np.full(x.shape[0], -1))
    if not Ps:
        return (None,) * 5
    P, Z, H = np.concatenate(Ps), np.concatenate(Zs), np.concatenate(Hs)
    y, idx = np.concatenate(ys), np.concatenate(ids)
    if len(P) > SAMPLE_CAP:
        keep = np.sort(np.random.default_rng(SEED).choice(len(P), SAMPLE_CAP, replace=False))
        P, Z, H, y, idx = P[keep], Z[keep], H[keep], y[keep], idx[keep]
    return P, Z, H, y, idx


def _save_matrix(ds, model, fold, P, Z, H, y, idx):
    """Per-sample softmax, logits, labels and dataset indices, plus projected head-input
    features, for arbitrary re-batching offline (label-shift stress test). The idx column matches
    the 'idx' column of the feature CSVs."""
    d = os.path.join(SOFTMAX_DIR, model, ds)
    os.makedirs(d, exist_ok=True)
    np.savez_compressed(os.path.join(d, f"{fold}.npz"), p=P.astype(np.float16), z=Z.astype(np.float16),
                        y=y.astype(np.int32), idx=idx.astype(np.int64))
    if H.shape[1]:
        d = os.path.join(FEATURE_DIR, model, ds)
        os.makedirs(d, exist_ok=True)
        np.savez_compressed(os.path.join(d, f"{fold}.npz"),
                            h=(H @ _projection(H.shape[1])).astype(np.float16), idx=idx.astype(np.int64))


def _stats(P, y, pi, ref_mean, rng, Z=None, H=None):
    yhat = P.argmax(1)
    out = {
        "true_acc": float((yhat == y).mean()),
        "ac": float(P.max(1).mean()),
        "nuclear_norm": nuclear_norm(P),
        "cot_err": cot_error(P, pi, rng),
        "label_shift_l1": float(np.abs(P.mean(0) - ref_mean).sum()),
        "N": int(P.shape[0]),
    }
    if Z is not None:
        out.update(mano=mano(Z), mano_softmax=mano(Z, mode="softmax"), mano_taylor=mano(Z, mode="taylor"))
    if H is not None and H.shape[1]:
        out["dispersion"] = dispersion(H, yhat, K=P.shape[1])
        out["dispersion_proj"] = dispersion(H @ _projection(H.shape[1]), yhat, K=P.shape[1])
    return out


def _merge_save(new_rows, path, done):
    """Write new rows, keeping rows of (dataset, model) blocks not processed in this run."""
    new = pd.DataFrame(new_rows)
    if os.path.exists(path):
        old = pd.read_csv(path)
        old = old[[(d, m) not in done for d, m in zip(old["Dataset"], old["Model"])]]
        new = pd.concat([new, old], ignore_index=True)
    new.round(6).to_csv(path, index=False)


def collect(datasets=CLASSIF, models=MODELS):
    os.makedirs("figures", exist_ok=True)
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    rows, sub_rows = [], []
    for ds in datasets:
        ctor = TESTBEDS.get(ds)
        if ctor is None:
            continue
        for model in models:
            pi = ref_mean = None
            for mode in ["normal"] + SYNTHETIC_SHIFTS:
                try:
                    bench = ctor(model=model, mode=mode, batch_size=BATCH)
                except Exception as e:
                    print(f"[skip] {ds}/{model}/{mode}: {e}", flush=True)
                    continue
                net = bench.classifier.to(DEVICE).eval()
                loaders = {}
                if mode == "normal":
                    loaders.update(bench.ind_val_loader())  # must come first: defines pi
                    loaders.update(bench.ind_test_loader())
                loaders.update(bench.ood_loaders())
                for fold, loader in loaders.items():
                    P, Z, H, y, idx = _fold_matrix(net, loader)
                    if P is None:
                        continue
                    _save_matrix(ds, model, fold, P, Z, H, y, idx)
                    if fold == "ind_val":
                        C = P.shape[1]
                        pi = np.bincount(y, minlength=C).astype(float)
                        pi /= pi.sum()
                        ref_mean = P.mean(0)
                    if pi is None:
                        raise RuntimeError(f"{ds}/{model}: ind_val must be processed first")
                    rows.append({"Dataset": ds, "Model": model, "fold": fold,
                                 **_stats(P, y, pi, ref_mean, rng, Z, H)})
                    n = min(SUBSET_SIZE, len(P))
                    for b in range(N_SUBSETS):
                        idx = rng.choice(len(P), size=n, replace=False)
                        sub_rows.append({"Dataset": ds, "Model": model, "fold": fold, "subset": b,
                                         **_stats(P[idx], y[idx], pi, ref_mean, rng, Z[idx], H[idx])})
                print(f"[done] {ds}/{model}/{mode}", flush=True)
                done = {(r["Dataset"], r["Model"]) for r in rows}
                _merge_save(rows, OUT, done)  # incremental save; other datasets' rows are kept
                _merge_save(sub_rows, OUT_SUB, done)
    print(f"softmax stats -> {OUT} ({len(rows)} rows), {OUT_SUB} ({len(sub_rows)} rows)", flush=True)


if __name__ == "__main__":
    try:
        torch.multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass
    collect()
