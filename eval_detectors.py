"""Collect post-hoc OOD-detector feature data for all (dataset, model, shift) cells.

Output convention (consumed by `utils.load_data`):
    data/{model}/feature_data/{dataset}_{mode}_{feature}.csv

Edit `DETECTORS` below to restrict which detectors are recomputed (e.g. only
`grad_magnitude` after fixing the GradNorm implementation).
"""

import os

import torch

from features import (
    cross_entropy,
    energy,
    grad_magnitude,
    knn,
    msp,
    typicality,
)
from ooddetectors import FeatureSD, convert_to_pandas_df, convert_to_pandas_df_no_ind
from testbeds import (
    CCTTestBed,
    NICOTestBed,
    Office31TestBed,
    OfficeHomeTestBed,
    PolypTestBed,
)
from utils import MODELS, SEG_MODELS, SYNTHETIC_SHIFTS

# ---------------------------------------------------------------------------
# Configure which detectors to (re)collect. Comment out any you want to skip.
# ---------------------------------------------------------------------------
DETECTORS = [
    grad_magnitude,
    cross_entropy,
    energy,
    knn,
    msp,
    typicality,
]

TESTBEDS = {
    "CCT": CCTTestBed,
    "OfficeHome": OfficeHomeTestBed,
    "Office31": Office31TestBed,
    "NICO": NICOTestBed,
    "Polyp": PolypTestBed,
}


def _feature_data_dir(model):
    return f"data/{model}/feature_data"


def _csv_path(model, dataset_name, mode, feature_name):
    return os.path.join(
        _feature_data_dir(model),
        f"{dataset_name}_{mode}_{feature_name}.csv",
    )


def _missing_detectors(detectors, model,  dataset_name, mode, overwrite):
    if overwrite:
        return list(detectors)
    missing = []
    for fn in detectors:
        path = _csv_path(model,dataset_name, mode, fn.__name__)
        if os.path.exists(path):
            print(f"  [skip] {path} exists")
        else:
            missing.append(fn)
    return missing


def _save(features_tuple, model, dataset_name, mode, feature_names, noind):
    out_prefix = os.path.join(
        _feature_data_dir(model), f"{dataset_name}_{mode}"
    )
    if noind:
        dfs = convert_to_pandas_df_no_ind(*features_tuple, feature_names)
    else:
        dfs = convert_to_pandas_df(*features_tuple, feature_names)
    for df, feature_name in zip(dfs, feature_names):
        df.to_csv(f"{out_prefix}_{feature_name}.csv")


def collect_features(
    testbed_constructor,
    dataset_name,
    detectors,
    mode="normal",
    model="resnet",
    batch_size=8,
    overwrite=False,
):
    """Collect feature CSVs for the requested detectors on a single (dataset, model, mode) cell."""
    todo = _missing_detectors(detectors, model, dataset_name, mode, overwrite)
    if not todo:
        print(f"  nothing to do for {dataset_name} / {model} / {mode}")
        return

    print(f"  collecting {[fn.__name__ for fn in todo]} for {dataset_name} / {model} / {mode}")
    os.makedirs(_feature_data_dir(model), exist_ok=True)

    bench = testbed_constructor(model=model, mode=mode, batch_size=batch_size)
    fsd = FeatureSD(bench.classifier, todo)
    fsd.register_testbed(bench)

    noind = mode != "normal"
    features_tuple = fsd.compute_pvals_and_loss(noind=noind)
    _save(features_tuple, model, dataset_name, mode,
          [fn.__name__ for fn in todo], noind=noind)


def collect_all(
    detectors=None,
    datasets=None,
    models=None,
    modes=None,
    batch_size=8,
    overwrite=False,
):
    """Sweep over (dataset, model, mode) cells, collecting only the listed detectors."""
    detectors = detectors if detectors is not None else DETECTORS
    datasets = datasets if datasets is not None else list(TESTBEDS.keys())
    models = models if models is not None else MODELS
    modes = modes if modes is not None else SYNTHETIC_SHIFTS + ["normal"]

    for dataset_name in datasets:
        constructor = TESTBEDS[dataset_name]
        for model in models:
            if dataset_name == "Polyp" and model not in SEG_MODELS:
                continue
            if dataset_name != "Polyp" and model in SEG_MODELS:
                continue
            for mode in modes:
                print(f"\n=== {dataset_name} | {model} | {mode} ===")

                collect_features(
                    constructor,
                    dataset_name,
                    detectors,
                    mode=mode,
                    model=model,
                    batch_size=batch_size,
                    overwrite=overwrite,
                )


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    collect_all(overwrite=True)
