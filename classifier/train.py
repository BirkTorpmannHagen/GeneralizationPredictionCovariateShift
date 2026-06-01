
from pytorch_lightning import Trainer
from datasets import *
from datasets.office31 import build_office31_dataset
from utils import INPUT_SIZE
from pytorch_lightning.callbacks import ModelCheckpoint

import warnings
from pytorch_lightning.loggers import TensorBoardLogger

from pytorch_lightning.tuner.tuning import Tuner  # works on PL 1.x/2.x
warnings.filterwarnings('ignore')

# torch and lightning imports
from torchvision import transforms
from torch.utils.data import DataLoader
from classifier.resnetclassifier import ResNetClassifier
from classifier.vit import ViTClassifier


# Here we define a new class to turn the ResNet model that we want to use as a feature extractor
# into a pytorch-lightning module so that we can take advantage of lightning's Trainer object.
# We aim to make it a little more general by allowing users to define the number of prediction classes.

def train_classifier(train_set, val_set, batch_size=16, load_from_checkpoint=None, model_type="vit"):
    num_classes =  train_set.num_classes
    if model_type=="resnet":
        model =  ResNetClassifier(num_classes, 101, batch_size=32, lr=1e-3).to("cuda")
    elif model_type=="vit":
        model =  ViTClassifier(num_classes, batch_size=batch_size, lr=1e-3).to("cuda")
    else:
        raise ValueError("model_type must be 'resnet' or 'vit'")
    if load_from_checkpoint:
        if model_type=="resnet":
            model = ResNetClassifier.load_from_checkpoint(load_from_checkpoint, num_classes=num_classes, resnet_version=101)
        elif model_type=="vit":
            print("loaded!")
            model = ViTClassifier.load_from_checkpoint(load_from_checkpoint, num_classes=num_classes, batch_size=32, lr=1e-3)
    # model = cifarrr
    tb_logger = TensorBoardLogger(save_dir=f"classifier_logs/{model_type}/{type(train_set).__name__}")
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"classifier_logs/{model_type}/{type(train_set).__name__}/checkpoints",
        save_top_k=3,
        verbose=True,
        monitor="val_acc",
        mode="max"
    )

    # ResNetClassifier.load_from_checkpoint("Imagenette_logs/checkpoints/epoch=82-step=24568.ckpt", resnet_version=101, nj
    trainer = Trainer(max_epochs=300, logger=tb_logger, accelerator="gpu",callbacks=checkpoint_callback)
    train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size, num_workers=4, persistent_workers=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=4, persistent_workers=True)
    tuner = Tuner(trainer)
    # lr_finder = tuner.lr_find(
    #     model,
    #     train_dataloaders=train_loader,
    #     val_dataloaders=val_loader,
    #     min_lr=1e-7,
    #     max_lr=10,
    #     num_training=100,     # ~100 steps is typical
    #     mode="exponential",            # classic LR range test
    # )
    # suggested_lr = float(lr_finder.suggestion())
    # print(f"Found lr: {suggested_lr}")
    # model.lr = suggested_lr
    trainer.fit(model, train_dataloaders=train_loader,
                val_dataloaders=val_loader)


if __name__ == '__main__':
    model_type="resnet"

    trans = transforms.Compose([transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
                        transforms.ToTensor(),
                            transforms.RandomHorizontalFlip(),
                        transforms.RandomVerticalFlip(),
                        transforms.RandomRotation(90),])
    val_trans = transforms.Compose([
                        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
                        transforms.ToTensor(), ])


    import torch
    torch.set_float32_matmul_precision('medium')
    train_set, val_set, test, ood_set = build_officehome_dataset("../../Datasets/OfficeHome", train_transform=trans, val_transform=val_trans)
    train_classifier(train_set, val_set, model_type=model_type)

    train_set, val_set, test_set, ood_set = build_nico_dataset( "../../Datasets/NICO++", trans, val_trans, ind_context="dim")
    train_classifier(train_set, val_set, model_type=model_type)

    train_set, val_set, test_set, ood_val_set, ood_test_set = build_office31_dataset("../../Datasets/office31", train_transform=trans, val_transform=val_trans )
    train_classifier(train_set, val_set, model_type=model_type)
    #
    train_set, val_set, test_set, ood_val_set, ood_test_set = build_cct_dataset("../../Datasets/CCT", train_transform=trans, val_transform=val_trans)
    train_classifier(train_set, val_set, model_type=model_type)

#train_set, val_set, test_set, ood_val_set, ood_test_set = build_cct_dataset("../../Datasets/CCT", trans, val_trans)
   # train_classifier(train_set, val_set, load_from_checkpoint="classifier_logs/CCT/checkpoints/epoch=60-step=50813.ckpt")
    # train_set, val_set, ood_set = build_officehome_dataset("../../Datasets/OfficeHome", train_transform=trans, val_transform=val_trans)
    # train_set, test_set,val_set, ood_set = get_pneumonia_dataset("../../Datasets/Pneumonia", trans, val_trans)
