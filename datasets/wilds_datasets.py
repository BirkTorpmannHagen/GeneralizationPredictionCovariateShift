"""WILDS dataset adapters (Camelyon17, iWildCam) for the covariate-shift testbeds.

Added in response to reviewer request for stronger real-world shift benchmarks
(67nM W4). These wrap the official WILDS domain-generalization splits into the
repo's (img, label, idx) dataset convention with `.num_classes`, so they plug into
the existing classifier training, testbeds, and FeatureSD pipeline.

Split mapping (domain = hospital for Camelyon17, camera location for iWildCam):
    ind_train  <- 'train'      (in-distribution domains)
    ind_val    <- 'id_val'     (held-out images, in-distribution domains)
    ind_test   <- 'id_test' if present else a deterministic half of 'id_val'
    ood_val    <- 'val'        (out-of-distribution domains)
    ood_test   <- 'test'       (out-of-distribution domains)

The synthetic-shift folds in the testbeds are derived (as elsewhere) by
transforming `ind_test`.
"""
import numpy as np
from torch.utils.data import Dataset

from wilds import get_dataset


class _WILDSFold(Dataset):
    """Wrap a WILDSSubset to yield (transformed_img, int_label, idx)."""

    def __init__(self, subset, num_classes):
        self.subset = subset
        self.num_classes = int(num_classes)

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, index):
        x, y, _metadata = self.subset[index]
        return x, int(y), index


# Distinct subclasses so classifier checkpoints land under
# classifier_logs/<model>/<ClassName>/ (train.py keys on type(train_set).__name__).
class Camelyon17(_WILDSFold):
    pass


class IWildCam(_WILDSFold):
    pass


def _resolve_splits(split_dict):
    """Map available WILDS split names to our five folds."""
    names = set(split_dict.keys())
    ind_val = "id_val" if "id_val" in names else ("val" if "val" in names else "train")
    ind_test = "id_test" if "id_test" in names else None  # else carved from ind_val
    ood_val = "val" if "val" in names else "test"
    ood_test = "test" if "test" in names else ood_val
    return ind_val, ind_test, ood_val, ood_test


def _slice_indices(subset, part, seed=42, cap=None):
    """Deterministically reduce a WILDSSubset's indices.

    part: 'all' | 'a' (first half) | 'b' (second half). cap: optional max size
    applied AFTER the half split. Used to (i) carve id_test from id_val when absent
    and (ii) subsample huge folds for tractable feature collection.
    """
    idx = np.array(subset.indices)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(idx))
    if part == "a":
        perm = perm[: len(idx) // 2]
    elif part == "b":
        perm = perm[len(idx) // 2:]
    if cap is not None and len(perm) > cap:
        perm = perm[:cap]
    subset.indices = idx[perm]
    return subset


# Default per-fold caps applied by the TESTBEDS (not by training) so that
# feature collection / synthetic-shift sweeps stay tractable on one GPU. Feature
# collection only needs enough samples to estimate a detection rate.
TESTBED_CAPS = {"train": 10000, "ind_val": 3000, "ind_test": 3000,
                "ood_val": 5000, "ood_test": 5000}


def _build(cls, name, root, train_transform, val_transform, caps=None):
    caps = caps or {}
    ds = get_dataset(dataset=name, root_dir=root, download=False)
    n = ds.n_classes
    ind_val_name, ind_test_name, ood_val_name, ood_test_name = _resolve_splits(ds.split_dict)

    train = cls(_slice_indices(ds.get_subset("train", transform=train_transform),
                               "all", cap=caps.get("train")), n)

    if ind_test_name is not None:
        ind_val = cls(_slice_indices(ds.get_subset(ind_val_name, transform=val_transform),
                                     "all", cap=caps.get("ind_val")), n)
        ind_test = cls(_slice_indices(ds.get_subset(ind_test_name, transform=val_transform),
                                      "all", cap=caps.get("ind_test")), n)
    else:
        # No id_test split: carve id_val in half deterministically.
        ind_val = cls(_slice_indices(ds.get_subset(ind_val_name, transform=val_transform),
                                     "a", cap=caps.get("ind_val")), n)
        ind_test = cls(_slice_indices(ds.get_subset(ind_val_name, transform=val_transform),
                                      "b", cap=caps.get("ind_test")), n)

    ood_val = cls(_slice_indices(ds.get_subset(ood_val_name, transform=val_transform),
                                 "all", cap=caps.get("ood_val")), n)
    ood_test = cls(_slice_indices(ds.get_subset(ood_test_name, transform=val_transform),
                                  "all", cap=caps.get("ood_test")), n)
    return train, ind_val, ind_test, ood_val, ood_test


def build_camelyon17_dataset(root, train_transform, val_transform, caps=None):
    return _build(Camelyon17, "camelyon17", root, train_transform, val_transform, caps)


def build_iwildcam_dataset(root, train_transform, val_transform, caps=None):
    return _build(IWildCam, "iwildcam", root, train_transform, val_transform, caps)
