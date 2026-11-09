"""Collect anomaly-detector features (Isolation Forest, LOF, One-Class SVM,
PCA reconstruction) for the unifying-view experiment.

These are fit on the InD-train reference encodings and scored on every fold, on
all classification datasets (old 4 + WILDS). Reuses the same FeatureSD pipeline
and reference cap as the OOD detectors. Idempotent (existing CSVs skipped).
"""
import torch
from eval_detectors import collect_all, DETECTORS_AD

BATCH_SIZE = 16
DATASETS = ["CCT", "OfficeHome", "Office31", "NICO", "Camelyon17", "IWildCam"]


def main():
    print("\n########## ANOMALY DETECTORS: unifying-view collection ##########", flush=True)
    collect_all(
        detectors=DETECTORS_AD,
        datasets=DATASETS,
        models=["resnet", "vit"],
        batch_size=BATCH_SIZE,
        overwrite=False,
    )
    print("\n########## AD COLLECTION DONE ##########", flush=True)


if __name__ == "__main__":
    try:
        torch.multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass
    main()
