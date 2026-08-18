"""
Hard mIoU / Dice segmentation metrics for the Polyp benchmark (Review YLV1 W10).

The cached feature_data stores only the *soft* Jaccard IoU (1 - JaccardLoss); the
continuous soft-IoU experiment (experiments/reviewer_experiments.polyp_continuous_iou)
uses that and needs no re-inference. This script instead recomputes the *hard*,
pixel-thresholded metrics that YLV1 asks about:

    * hard mIoU : iou_score(get_stats(preds, y, mode='binary', threshold=0.5))  (matches segmentor.py)
    * Dice/F1   : f1_score(...) with the same stats

It therefore REQUIRES the segmentation checkpoints under
``segmentation_logs/checkpoints/<model>/best.ckpt`` (and the Glow checkpoint that
PolypTestBed loads) plus the Polyp datasets under ``../../Datasets/Polyps``. Run it
on a machine where those are present:

    python -m experiments.segmentation_hard_metrics

It writes per-fold hard metrics to ``data/<model>/hard_metrics/Polyp_hardmetrics.csv``
and, by joining the fold-level metrics to the cached OOD detection rates, an MAE
table ``figures/polyp_hard_metrics.csv`` directly comparable to the soft-IoU and
binary-IoU>0.5 Polyp rows.
"""
import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from segmentation_models_pytorch.metrics import get_stats, iou_score, f1_score

from utils import SYNTHETIC_SHIFTS, SEG_MODELS
from testbeds.polyps import PolypTestBed

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _per_sample_hard_metrics(model, loader):
    """Return a DataFrame [idx, hard_iou, dice] for one fold loader."""
    recs = []
    model.eval()
    with torch.no_grad():
        for data in tqdm(loader, total=len(loader), leave=False):
            x = data[0].to(DEVICE)
            y = data[1].to(DEVICE)
            idx = data[2]
            preds = model(x)
            # threshold=0.5 on the raw model output, matching segmentor.py:81/92/101
            tp, fp, fn, tn = get_stats(preds, y.long(), mode="binary", threshold=0.5)
            iou = iou_score(tp, fp, fn, tn, reduction=None).squeeze(-1).cpu().numpy()
            dice = f1_score(tp, fp, fn, tn, reduction=None).squeeze(-1).cpu().numpy()
            idx = np.asarray(idx).reshape(-1)
            for j in range(len(iou)):
                recs.append({"idx": int(idx[j]), "hard_iou": float(iou[j]), "dice": float(dice[j])})
    return pd.DataFrame(recs)


def collect_hard_metrics(models=tuple(SEG_MODELS), batch_size=16):
    """Re-run each segmentor over every fold and cache per-sample hard IoU + Dice."""
    for model_name in models:
        out_dir = f"data/{model_name}/hard_metrics"
        os.makedirs(out_dir, exist_ok=True)
        all_rows = []
        for mode in ["normal"] + SYNTHETIC_SHIFTS:
            tb = PolypTestBed(mode=mode, model=model_name, batch_size=batch_size)
            model = tb.classifier.to(DEVICE)
            loaders = {}
            if mode == "normal":
                loaders.update(tb.ind_val_loader())
                loaders.update(tb.ind_test_loader())
            loaders.update(tb.ood_loaders())
            for fold, loader in loaders.items():
                df = _per_sample_hard_metrics(model, loader)
                if df.empty:
                    continue
                df["fold"] = fold
                df["Model"] = model_name
                df["Dataset"] = "Polyp"
                all_rows.append(df)
        if all_rows:
            out = pd.concat(all_rows, ignore_index=True)
            out.to_csv(f"{out_dir}/Polyp_hardmetrics.csv", index=False)
            print(f"[{model_name}] wrote {len(out)} rows -> {out_dir}/Polyp_hardmetrics.csv")


def hard_metric_table(models=tuple(SEG_MODELS)):
    """
    LOSO regression MAE with hard-IoU and Dice degradation as the target,
    reusing the exact protocol and per-fold DR from reviewer_experiments.
    Must be run after collect_hard_metrics().
    """
    from experiments.reviewer_experiments import (
        _perfold_dr, _loso_from_perfold, _best_config_pivot, DSDS_INTERNAL,
    )
    from utils import load_all

    raw = load_all(batch_size=1, shift="")
    raw = raw[raw["Dataset"] == "Polyp"]

    results = {}
    for target, col in [("hardIoU", "hard_iou"), ("Dice", "dice")]:
        rows = []
        for model_name in models:
            path = f"data/{model_name}/hard_metrics/Polyp_hardmetrics.csv"
            if not os.path.exists(path):
                print(f"[hard_metric_table] missing {path}; run collect_hard_metrics() first.")
                continue
            hm = pd.read_csv(path)
            # fold-level mean of the hard target
            fold_perf = hm.groupby("fold")[col].mean().to_dict()
            for ff in DSDS_INTERNAL:
                pf = _perfold_dr(raw, "Polyp", model_name, ff, 0.95)
                if pf is None or pf.empty:
                    continue
                pf = pf.copy()
                pf["Accuracy"] = pf["fold"].map(fold_perf)
                pf = pf.dropna(subset=["Accuracy"])
                if pf.empty:
                    continue
                ind_val_perf = fold_perf.get("ind_val", np.nan)
                if np.isnan(ind_val_perf):
                    continue
                pf["gap"] = ind_val_perf - pf["Accuracy"]
                rows += _loso_from_perfold(pf, "Polyp", model_name, ff, f"Ours-{target}")
        if rows:
            pivot, _ = _best_config_pivot(pd.DataFrame(rows), [f"Ours-{target}"])
            results[target] = float(pivot.loc["Polyp"].iloc[0])

    out = pd.DataFrame([results])
    os.makedirs("figures", exist_ok=True)
    out.round(4).to_csv("figures/polyp_hard_metrics.csv", index=False)
    print("\n=== Polyp hard-metric MAE (LOSO) ===")
    print(out.round(4).to_string(index=False))
    return out


if __name__ == "__main__":
    collect_hard_metrics()
    hard_metric_table()
