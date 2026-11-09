"""CIFAR-10 family adapters for the covariate-shift testbeds (UAE benchmark parity).

Base in-distribution: CIFAR-10 (torchvision, auto-downloaded).
Organic OOD folds: CIFAR-10.1 (natural shift) and a selection of CIFAR-10-C
corruptions (standard synthetic-corruption suite). Synthetic-shift folds are, as
elsewhere, derived by transforming ind_test inside the testbed (mode != "normal").

Returns the repo convention: each fold yields (img, int_label, idx) and exposes
`.num_classes`; builders return (ind_train, ind_val, ind_test, ood_val, ood_test).

Expected on-disk layout under DATA_ROOT/cifar (see scripts/download_uae_datasets.sh):
    CIFAR-10-C/<corruption>.npy, CIFAR-10-C/labels.npy
    CIFAR-10.1/cifar10.1_v6_data.npy, .../cifar10.1_v6_labels.npy
    (base CIFAR-10 is fetched by torchvision into DATA_ROOT/cifar)
"""
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10 as _TVCIFAR10


class _ArrayDS(Dataset):
    """Wrap uint8 HWC arrays + int labels into (transformed_img, label, idx)."""

    def __init__(self, images, labels, transform, num_classes=10):
        self.images = images                      # (N, 32, 32, 3) uint8
        self.labels = np.asarray(labels).astype(int)
        self.transform = transform
        self.num_classes = int(num_classes)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img = Image.fromarray(self.images[i])
        return self.transform(img), int(self.labels[i]), i


# Distinct class so train checkpoints land under classifier_logs/<model>/CIFAR10/
class CIFAR10(_ArrayDS):
    pass


def _deterministic_halves(n, seed=42):
    perm = np.random.default_rng(seed).permutation(n)
    return perm[: n // 2], perm[n // 2:]


def build_cifar10_dataset(root, train_transform, val_transform, corruptions=("gaussian_noise", "fog", "brightness"),
                          severities=(3, 5)):
    """CIFAR-10 with CIFAR-10.1 + selected CIFAR-10-C corruptions as organic OOD."""
    cdir = os.path.join(root, "cifar")
    tv_train = _TVCIFAR10(cdir, train=True, download=True)
    tv_test = _TVCIFAR10(cdir, train=False, download=True)

    ind_train = CIFAR10(tv_train.data, tv_train.targets, train_transform)
    a, b = _deterministic_halves(len(tv_test.data))
    ind_val = CIFAR10(tv_test.data[a], np.array(tv_test.targets)[a], val_transform)
    ind_test = CIFAR10(tv_test.data[b], np.array(tv_test.targets)[b], val_transform)

    # organic: CIFAR-10.1 natural shift
    o = os.path.join(cdir, "CIFAR-10.1")
    c101_x = np.load(os.path.join(o, "cifar10.1_v6_data.npy"))
    c101_y = np.load(os.path.join(o, "cifar10.1_v6_labels.npy"))
    oa, ob = _deterministic_halves(len(c101_x))
    ood_val = _ArrayDS(c101_x[oa], c101_y[oa], val_transform)
    ood_test = _ArrayDS(c101_x[ob], c101_y[ob], val_transform)
    return ind_train, ind_val, ind_test, ood_val, ood_test


def cifar10c_folds(root, val_transform, corruptions=("gaussian_noise", "fog", "brightness",
                                                     "contrast", "motion_blur", "jpeg_compression"),
                   severities=(1, 3, 5)):
    """Return {name: dataset} of CIFAR-10-C corruption folds for use as extra organic
    OOD folds (override the testbed's get_ood_dict to include these)."""
    cc = os.path.join(root, "cifar", "CIFAR-10-C")
    labels = np.load(os.path.join(cc, "labels.npy"))
    out = {}
    for corr in corruptions:
        arr = np.load(os.path.join(cc, f"{corr}.npy"))     # (50000,32,32,3): 5 severities x 10000
        for sev in severities:
            sl = slice((sev - 1) * 10000, sev * 10000)
            out[f"{corr}_s{sev}"] = _ArrayDS(arr[sl], labels[sl], val_transform)
    return out
