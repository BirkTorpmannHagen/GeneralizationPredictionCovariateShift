"""Train ResNet-101 and ViT classifiers on the WILDS additions (Camelyon17, iWildCam).

Fine-tunes the ImageNet-pretrained backbones on each dataset's in-distribution
train split, checkpointing the best (max val_acc) model to
    classifier_logs/<model>/<Camelyon17|IWildCam>/checkpoints/best.ckpt
which is exactly where the corresponding testbeds load from.

Usage:
    python train_wilds.py --smoke                 # data + 1-batch forward sanity check
    python train_wilds.py                          # train all 4 (dataset x arch)
    python train_wilds.py --datasets camelyon17 --models resnet
"""
import argparse
import warnings

warnings.filterwarnings("ignore")

from torchvision import transforms
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from utils import INPUT_SIZE
from datasets.wilds_datasets import build_camelyon17_dataset, build_iwildcam_dataset
from classifier.resnetclassifier import ResNetClassifier
from classifier.vit import ViTClassifier

WILDS_ROOT = "../../Datasets/wilds"

BUILDERS = {
    "camelyon17": (build_camelyon17_dataset, "Camelyon17"),
    "iwildcam": (build_iwildcam_dataset, "IWildCam"),
}


def make_transforms():
    train_t = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(90),
        transforms.ToTensor(),
    ])
    val_t = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
    ])
    return train_t, val_t


def build(dataset):
    builder, class_name = BUILDERS[dataset]
    train_t, val_t = make_transforms()
    train, ind_val, ind_test, ood_val, ood_test = builder(WILDS_ROOT, train_t, val_t)
    return train, ind_val, class_name


def train_one(dataset, model_type, batch_size=32, max_epochs=15, patience=3, num_workers=6):
    train_set, val_set, class_name = build(dataset)
    num_classes = train_set.num_classes
    print(f"[{dataset}/{model_type}] classes={num_classes} train={len(train_set)} val={len(val_set)}")

    if model_type == "resnet":
        model = ResNetClassifier(num_classes, 101, batch_size=batch_size, lr=1e-3)
    else:
        model = ViTClassifier(num_classes, batch_size=batch_size, lr=1e-3)

    logdir = f"classifier_logs/{model_type}/{class_name}"
    ckpt = ModelCheckpoint(
        dirpath=f"{logdir}/checkpoints", filename="best",
        save_top_k=1, monitor="val_acc", mode="max", verbose=True,
    )
    early = EarlyStopping(monitor="val_acc", mode="max", patience=patience)
    trainer = Trainer(
        max_epochs=max_epochs, accelerator="gpu", devices=1,
        logger=TensorBoardLogger(save_dir=logdir),
        callbacks=[ckpt, early],
        precision=16,
        # Bound epoch length so early-stopping acts within a reasonable wall-clock
        # (pretrained backbones fine-tune fast; full passes over 300k patches are
        # unnecessary). ~3000 steps * bs = ~96k images per epoch.
        limit_train_batches=3000,
        limit_val_batches=400,
    )
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, persistent_workers=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, persistent_workers=True)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    print(f"[{dataset}/{model_type}] best checkpoint -> {ckpt.best_model_path} (val_acc={ckpt.best_model_score})")


def smoke():
    import torch
    for dataset in BUILDERS:
        train_set, val_set, class_name = build(dataset)
        print(f"[smoke] {dataset}: class={class_name} num_classes={train_set.num_classes} "
              f"train={len(train_set)} val={len(val_set)}")
        x, y, idx = train_set[0]
        print(f"[smoke] {dataset}: x={tuple(x.shape)} y={y} idx={idx}")
        m = ResNetClassifier(train_set.num_classes, 101).eval()
        with torch.no_grad():
            out = m(x.unsqueeze(0))
        print(f"[smoke] {dataset}: resnet out {tuple(out.shape)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--datasets", nargs="+", default=list(BUILDERS.keys()))
    ap.add_argument("--models", nargs="+", default=["resnet", "vit"])
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()

    if args.smoke:
        smoke()
    else:
        for dataset in args.datasets:
            for model_type in args.models:
                print(f"\n===== TRAIN {dataset} / {model_type} =====", flush=True)
                train_one(dataset, model_type, batch_size=args.batch_size, max_epochs=args.epochs)
