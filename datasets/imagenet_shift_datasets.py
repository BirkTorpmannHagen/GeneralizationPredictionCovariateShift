"""ImageNet covariate-shift adapters (ImageNet-R / -A / -V2 / -C) for the testbeds.

No per-dataset training: the testbed uses an ImageNet-pretrained backbone directly,
so this only provides evaluation folds. Base in-distribution = the standard ImageNet
validation set; organic OOD = the shift variants.

KEY SUBTLETY (verify on-machine): ImageNet-R and ImageNet-A cover only 200 of the
1000 classes. Their folder wnids are remapped to the canonical 1000-class index via
the base val ImageFolder's class_to_idx (sorted-wnid order == torchvision pretrained
order). For those two folds, accuracy must be computed over the 200-class subset:
the builder returns the subset's 1000-class index list so the testbed can mask logits.
ImageNet-V2 and ImageNet-C cover all 1000 classes and need no masking.

Expected layout under DATA_ROOT/imagenet_shift (see scripts/download_uae_datasets.sh):
    imagenet-r/<wnid>/*.jpg          (200 classes)
    imagenet-a/<wnid>/*.jpg          (200 classes)
    imagenetv2-matched-frequency-format-val/<0..999>/*.jpeg
    imagenet-c/<corruption>/<severity>/<wnid>/*.JPEG   (optional, large)
Base ImageNet val (ILSVRC2012) is supplied separately via IMAGENET_VAL.
"""
import os
import numpy as np
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset


class _IdxWrap(Dataset):
    """ImageFolder -> (img, canonical_1000_label, idx), remapping folder labels."""

    def __init__(self, folder_ds, folder_to_canonical, transform, num_classes=1000):
        self.ds = folder_ds
        self.map = folder_to_canonical            # folder-local class idx -> 1000-class idx
        self.transform = transform
        self.num_classes = num_classes

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        path, local_label = self.ds.samples[i]
        img = self.ds.loader(path)
        return self.transform(img), int(self.map[local_label]), i


class ImageNet(_IdxWrap):
    pass


def _canonical_wnid_to_idx(imagenet_val_dir):
    """The canonical torchvision-pretrained class order = sorted wnid order of val."""
    base = ImageFolder(imagenet_val_dir)
    return base.class_to_idx, base                # wnid -> 0..999


def build_imagenet_shift_dataset(root, imagenet_val_dir, val_transform, variant="imagenet-r"):
    """Return (ind_train=None, ind_val, ind_test, ood_val, ood_test, subset_idx).

    ind_val/ind_test are halves of base ImageNet val (in-distribution). ood_* are the
    chosen shift variant. subset_idx is None for full-1000 variants, else the list of
    1000-class indices the fold covers (mask logits to these before argmax).
    """
    wnid_to_idx, base = _canonical_wnid_to_idx(imagenet_val_dir)
    n = len(base.samples)
    perm = np.random.default_rng(42).permutation(n)
    # in-distribution halves of base val
    ind_val = _IdxWrap(_Subset(base, perm[: n // 2]), {i: i for i in range(1000)}, val_transform)
    ind_test = _IdxWrap(_Subset(base, perm[n // 2:]), {i: i for i in range(1000)}, val_transform)

    vdir = {
        "imagenet-r": os.path.join(root, "imagenet-r"),
        "imagenet-a": os.path.join(root, "imagenet-a"),
        "imagenet-v2": os.path.join(root, "imagenetv2-matched-frequency-format-val"),
    }[variant]
    fold = ImageFolder(vdir)

    if variant in ("imagenet-r", "imagenet-a"):
        # folder classes are wnids -> map to canonical 1000-class idx; record subset
        local_to_canon = {li: wnid_to_idx[wnid] for wnid, li in fold.class_to_idx.items()}
        subset_idx = sorted(local_to_canon.values())
    else:
        # imagenet-v2 folders are 0..999 integer class ids already in canonical order
        local_to_canon = {i: i for i in range(1000)}
        subset_idx = None

    m = len(fold.samples)
    pv = np.random.default_rng(7).permutation(m)
    ood_val = _IdxWrap(_Subset(fold, pv[: m // 2]), local_to_canon, val_transform)
    ood_test = _IdxWrap(_Subset(fold, pv[m // 2:]), local_to_canon, val_transform)
    return None, ind_val, ind_test, ood_val, ood_test, subset_idx


class _Subset:
    """Lightweight index subset preserving .samples/.loader for _IdxWrap."""

    def __init__(self, folder_ds, indices):
        self.loader = folder_ds.loader
        self.samples = [folder_ds.samples[i] for i in indices]
