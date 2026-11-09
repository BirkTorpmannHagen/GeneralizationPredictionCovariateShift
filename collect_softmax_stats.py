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
  * true_acc     : accuracy on the same samples.
"""
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from eval_detectors import TESTBEDS
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


def nuclear_norm(P):
    N, C = P.shape
    G = P.T @ P  # C x C, cheaper than an SVD of N x C
    eig = np.clip(np.linalg.eigvalsh(G), 0, None)
    return float(np.sqrt(eig).sum() / np.sqrt(min(N, C) * N))


def _labels_from_marginal(pi, n, rng):
    """n one-hot labels whose empirical distribution matches pi as closely as possible."""
    raw = pi * n
    counts = np.floor(raw).astype(int)
    rem = n - counts.sum()
    if rem > 0:
        frac = raw - counts
        # break ties randomly so no class is systematically favoured
        order = np.lexsort((rng.random(len(pi)), -frac))
        counts[order[:rem]] += 1
    return np.repeat(np.arange(len(pi)), counts)


def cot_error(P, pi, rng):
    """COT predicted error: W(P_pred, P_source_labels) with cost ||p - e_y||_inf."""
    N, C = P.shape
    labels = _labels_from_marginal(pi, N, rng)
    order = np.argsort(-P, axis=1)
    top1 = order[:, 0]
    m1 = P[np.arange(N), top1]
    m2 = P[np.arange(N), order[:, 1]] if C > 1 else np.zeros(N)
    # ||p - e_y||_inf = max(1 - p_y, max_{c != y} p_c)
    other_max = np.where(top1[:, None] == np.arange(C)[None, :], m2[:, None], m1[:, None])
    cost_by_class = np.maximum(1.0 - P, other_max)  # N x C
    cost = cost_by_class[:, labels]                  # N x N
    r, c = linear_sum_assignment(cost)
    return float(cost[r, c].mean())


def _fold_matrix(net, loader):
    """Whole fold, then a seeded random subsample of SAMPLE_CAP samples. (Evaluation loaders are
    deliberately unshuffled, see testbeds/base.py, so taking the first SAMPLE_CAP samples would
    keep only the first few classes.)"""
    Ps, ys, ids = [], [], []
    for data in loader:
        x = data[0].to(DEVICE)
        with torch.no_grad():
            logits = net(x)
            if isinstance(logits, list):
                logits = logits[1]
        Ps.append(F.softmax(logits.float(), dim=1).cpu().numpy())
        ys.append(data[1].cpu().numpy())
        ids.append(np.asarray(data[2]) if len(data) > 2 else np.full(x.shape[0], -1))
    if not Ps:
        return None, None, None
    P, y, idx = np.concatenate(Ps), np.concatenate(ys), np.concatenate(ids)
    if len(P) > SAMPLE_CAP:
        keep = np.sort(np.random.default_rng(SEED).choice(len(P), SAMPLE_CAP, replace=False))
        P, y, idx = P[keep], y[keep], idx[keep]
    return P, y, idx


def _save_matrix(ds, model, fold, P, y, idx):
    """Per-sample softmax, labels and dataset indices, for arbitrary re-batching offline
    (label-shift stress test). The idx column matches the 'idx' column of the feature CSVs."""
    d = os.path.join(SOFTMAX_DIR, model, ds)
    os.makedirs(d, exist_ok=True)
    np.savez_compressed(os.path.join(d, f"{fold}.npz"), p=P.astype(np.float16), y=y.astype(np.int32),
                        idx=idx.astype(np.int64))


def _stats(P, y, pi, ref_mean, rng):
    return {
        "true_acc": float((P.argmax(1) == y).mean()),
        "ac": float(P.max(1).mean()),
        "nuclear_norm": nuclear_norm(P),
        "cot_err": cot_error(P, pi, rng),
        "label_shift_l1": float(np.abs(P.mean(0) - ref_mean).sum()),
        "N": int(P.shape[0]),
    }


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
                    P, y, idx = _fold_matrix(net, loader)
                    if P is None:
                        continue
                    _save_matrix(ds, model, fold, P, y, idx)
                    if fold == "ind_val":
                        C = P.shape[1]
                        pi = np.bincount(y, minlength=C).astype(float)
                        pi /= pi.sum()
                        ref_mean = P.mean(0)
                    if pi is None:
                        raise RuntimeError(f"{ds}/{model}: ind_val must be processed first")
                    rows.append({"Dataset": ds, "Model": model, "fold": fold,
                                 **_stats(P, y, pi, ref_mean, rng)})
                    n = min(SUBSET_SIZE, len(P))
                    for b in range(N_SUBSETS):
                        idx = rng.choice(len(P), size=n, replace=False)
                        sub_rows.append({"Dataset": ds, "Model": model, "fold": fold, "subset": b,
                                         **_stats(P[idx], y[idx], pi, ref_mean, rng)})
                print(f"[done] {ds}/{model}/{mode}", flush=True)
                pd.DataFrame(rows).round(6).to_csv(OUT, index=False)  # incremental save
                pd.DataFrame(sub_rows).round(6).to_csv(OUT_SUB, index=False)
    print(f"softmax stats -> {OUT} ({len(rows)} rows), {OUT_SUB} ({len(sub_rows)} rows)", flush=True)


if __name__ == "__main__":
    try:
        torch.multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass
    collect()
