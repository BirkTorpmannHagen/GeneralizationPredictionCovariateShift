"""Per-fold softmax-matrix statistics for recent label-free accuracy-estimation
SOTA baselines (YLV1: "more recent, stronger baselines").

Produces, per (dataset, model, fold), a scalar score + the fold's true accuracy:
  * nuclear_norm : Deng et al., CVPR 2023 — nuclear norm of the [N,C] softmax
                   prediction matrix, size-normalized (higher dispersity/confidence
                   => higher accuracy). Computed via the C x C Gram matrix P^T P so
                   it scales to large folds.
  * cot          : a COT-style score (Lu et al., ICML 2023) — the L1 optimal-transport
                   cost between the fold's mean predicted class distribution and the
                   InD-val class distribution (0 when the label marginal is unchanged).

Each fold is subsampled to SAMPLE_CAP for tractability (a fold-level statistic needs
only enough samples to estimate it; this is NOT the evaluation data). The scores are
calibrated to accuracy with the same leave-one-shift-family-out linear map as our
estimator (see experiments.reviewer_experiments._softmax_stat_predictions), i.e. an
equal-resource comparison.

Writes figures/softmax_stats.csv. Requires the trained classifiers + datasets (GPU).
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from utils import SYNTHETIC_SHIFTS
from eval_detectors import TESTBEDS

CLASSIF = ["CCT", "OfficeHome", "Office31", "NICO", "Camelyon17", "IWildCam"]
MODELS = ["resnet", "vit"]
SAMPLE_CAP = 2000
BATCH = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT = "figures/softmax_stats.csv"


def _fold_stats(net, loader):
    C = None
    G = None
    N = 0
    correct = 0
    mean_p = None
    for data in loader:
        if N >= SAMPLE_CAP:
            break
        x = data[0].to(DEVICE)
        y = data[1].to(DEVICE)
        with torch.no_grad():
            logits = net(x)
            if isinstance(logits, list):
                logits = logits[1]
            p = F.softmax(logits, dim=1)
        if C is None:
            C = p.shape[1]
            G = torch.zeros(C, C, device=DEVICE)
            mean_p = torch.zeros(C, device=DEVICE)
        G += p.t() @ p
        mean_p += p.sum(0)
        correct += int((logits.argmax(1) == y).sum().item())
        N += p.shape[0]
    if N == 0:
        return None
    eig = torch.linalg.eigvalsh(G).clamp(min=0)
    nuclear = float(torch.sqrt(eig).sum().item()) / np.sqrt(N)
    mean_p = (mean_p / N).cpu().numpy()
    return {"true_acc": correct / N, "nuclear_norm": nuclear, "mean_p": mean_p, "N": N}


def collect(datasets=CLASSIF, models=MODELS):
    os.makedirs("figures", exist_ok=True)
    rows = []
    for ds in datasets:
        ctor = TESTBEDS.get(ds)
        if ctor is None:
            continue
        for model in models:
            ind_marginal = {}
            for mode in ["normal"] + SYNTHETIC_SHIFTS:
                try:
                    bench = ctor(model=model, mode=mode, batch_size=BATCH)
                except Exception as e:
                    print(f"[skip] {ds}/{model}/{mode}: {e}", flush=True)
                    continue
                net = bench.classifier.to(DEVICE).eval()
                loaders = {}
                if mode == "normal":
                    loaders.update(bench.ind_val_loader())
                    loaders.update(bench.ind_test_loader())
                loaders.update(bench.ood_loaders())
                for fold, loader in loaders.items():
                    st = _fold_stats(net, loader)
                    if st is None:
                        continue
                    if fold == "ind_val":
                        ind_marginal[(ds, model)] = st["mean_p"]
                    ref = ind_marginal.get((ds, model))
                    cot = float(np.abs(st["mean_p"] - ref).sum()) if ref is not None else np.nan
                    rows.append({"Dataset": ds, "Model": model, "fold": fold,
                                 "true_acc": round(st["true_acc"], 5),
                                 "nuclear_norm": round(st["nuclear_norm"], 5),
                                 "cot": round(cot, 5) if not np.isnan(cot) else np.nan,
                                 "N": st["N"]})
                print(f"[done] {ds}/{model}/{mode}", flush=True)
                pd.DataFrame(rows).to_csv(OUT, index=False)  # incremental save
    print(f"softmax stats -> {OUT} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    try:
        torch.multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass
    collect()
