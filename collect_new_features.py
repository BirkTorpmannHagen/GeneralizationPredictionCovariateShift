"""Feature collection for the revision.

Two sweeps (see the reviewer discussion):
  1. NEW datasets (Camelyon17, iWildCam): compute ALL applicable detectors in one
     pass — the 5 existing post-hoc detectors (no Glow/Typicality) PLUS the new
     latent-space/hybrid detectors (Mahalanobis, ReAct, ViM).
  2. OLD classification datasets (CCT, OfficeHome, Office31, NICO): compute ONLY
     the new detectors (the existing detector features are already cached).

Polyp is skipped for the new detectors (they are classification-only; segmentation
has no per-image class/logit structure for Mahalanobis/ViM).

Evaluation folds are FULL (no subsampling); only the kNN/Maha/ViM reference is
capped inside FeatureSD (REFERENCE_CAP). Idempotent: existing per-detector CSVs are
skipped, so this can be re-run to resume.
"""
import torch

from eval_detectors import collect_all, DETECTORS_ALL_NO_GLOW, DETECTORS_NEW

BATCH_SIZE = 16


def main():
    # 1. New datasets: full detector set (no Glow).
    print("\n########## NEW DATASETS: all detectors ##########", flush=True)
    collect_all(
        detectors=DETECTORS_ALL_NO_GLOW,
        datasets=["Camelyon17", "IWildCam"],
        models=["resnet", "vit"],
        batch_size=BATCH_SIZE,
        overwrite=False,
    )

    # 2. Old classification datasets: new detectors only.
    print("\n########## OLD DATASETS: new detectors only ##########", flush=True)
    collect_all(
        detectors=DETECTORS_NEW,
        datasets=["CCT", "OfficeHome", "Office31", "NICO"],
        models=["resnet", "vit"],
        batch_size=BATCH_SIZE,
        overwrite=False,
    )
    print("\n########## FEATURE COLLECTION DONE ##########", flush=True)


if __name__ == "__main__":
    try:
        torch.multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass
    main()
