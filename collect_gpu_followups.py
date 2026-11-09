"""GPU-side collection for the hygiene / baseline fixes (see GPU_RERUN.md).

  1. Predictive-entropy detector (`features.entropy`) for every dataset and model.
     Needed for the Entropy detector, ATC-NE and Reg-Entropy; the old "Entropy" was the
     cross-entropy to an all-ones target (now reported as UniformCE).
  2. Softmax-matrix statistics: Nuclear Norm, real COT, AC, and bootstrap batches for the
     set-level shift-detection analysis (collect_softmax_stats.py).

NICO must be retrained first (seeded split; see GPU_RERUN.md), then recollected with
every detector; pass --nico to include that recollection here.
Idempotent for step 1: existing CSVs are skipped unless --overwrite.
"""
import argparse

import torch

import collect_softmax_stats
from eval_detectors import DETECTORS_ALL_NO_GLOW, collect_all
from features import entropy

CLASSIF = ["CCT", "OfficeHome", "Office31", "Camelyon17", "IWildCam"]
SEG = ["deeplabv3plus", "unet", "segformer"]


def main(nico=False, overwrite=False):
    datasets = CLASSIF + (["NICO"] if nico else [])
    if nico:
        print("\n### NICO: full recollection after retraining ###", flush=True)
        collect_all(detectors=DETECTORS_ALL_NO_GLOW, datasets=["NICO"], models=["resnet", "vit"],
                    batch_size=16, overwrite=True)
    print("\n### Entropy detector ###", flush=True)
    collect_all(detectors=[entropy], datasets=datasets, models=["resnet", "vit"],
                batch_size=16, overwrite=overwrite)
    collect_all(detectors=[entropy], datasets=["Polyp"], models=SEG, batch_size=16, overwrite=overwrite)
    print("\n### Softmax-matrix statistics (Nuclear Norm, COT, subsets) ###", flush=True)
    collect_softmax_stats.collect(datasets=datasets)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--nico", action="store_true", help="include NICO (only after retraining it)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    try:
        torch.multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass
    main(nico=args.nico, overwrite=args.overwrite)
