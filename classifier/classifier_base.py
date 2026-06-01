

import sys
import warnings
from pathlib import Path
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings('ignore')

# torch and lightning imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.optim import SGD, Adam
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torchmetrics import Accuracy

class Classifier(pl.LightningModule):
    def __init__(self, num_classes,
                 optimizer='adam', lr=1e-3, batch_size=16, img_shape=224):
        super().__init__()

        self.__dict__.update(locals())
        self.num_classes = num_classes
        optimizers = {'adam': Adam, 'sgd': SGD}
        self.optimizer = optimizers[optimizer]
        # instantiate loss criterion
        self.criterion = nn.CrossEntropyLoss(reduction="none")
        # Using a pretrained ResNet backbone
        self.img_shape=img_shape
        self.model = []
        # Replace old FC layer with Identity so we can train our own
        # replace final layer for fine tuning

        self.acc = Accuracy(task="multiclass", num_classes=num_classes, top_k=1)


    def forward(self, X):
        return self.model(X)

    def get_encoding_size(self, depth):
        dummy = torch.randn((1,3,self.img_shape,self.img_shape)).to(self.device)
        return self.get_encoding(dummy).shape[-1]

    def get_encoding(self, X, depth=-2):

        with torch.no_grad():
            encoding =  torch.nn.Sequential(*list(self.model.children())[:-1])(X).flatten(1)
            return encoding

    def compute_loss(self, x, y):
        return self.criterion(self(x), y)

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,100, 2)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}


    def training_step(self, batch, batch_idx):
        x = batch[0]
        y = batch[1]
        preds = self(x)
        loss = self.criterion(preds, y).mean()

        acc = self.acc(preds, y)
        # perform logging
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch[0]
        y = batch[1]
        preds = self(x)
        loss = self.criterion(preds,y).mean()
        acc = self.acc(preds, y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True, logger=True)


    def test_step(self, batch, batch_idx):
        x = batch[0]
        y = batch[1]
        preds = self(x)
        loss = self.criterion(preds, y).mean()
        acc = self.acc(preds, y)

        # perform logging
        self.log("test_loss", loss, on_step=True, prog_bar=True, logger=True)
        self.log("test_acc", acc, on_step=True, prog_bar=True, logger=True)
