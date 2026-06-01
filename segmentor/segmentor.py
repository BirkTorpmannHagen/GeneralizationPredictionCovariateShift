
from segmentation_models_pytorch import DeepLabV3Plus, UnetPlusPlus, Segformer
from segmentation_models_pytorch.losses import JaccardLoss
from segmentation_models_pytorch.metrics import get_stats, iou_score
import warnings
from segmentation_models_pytorch.base import SegmentationHead

from utils import INPUT_SIZE

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



class SegmentationModel(pl.LightningModule):
    def __init__(self, transfer=True,
                 optimizer='adam', lr=1e-3, batch_size=16, model_name="deeplabv3plus"):
        super().__init__()

        self.__dict__.update(locals())
        optimizers = {'adam': Adam, 'sgd': SGD}
        self.optimizer = optimizers[optimizer]
        # instantiate loss criterion
        self.criterion = JaccardLoss(mode="binary")
        optimizers = {'adam': Adam, 'sgd': SGD}
        self.optimizer = optimizers[optimizer]
        # instantiate loss criterion
        # Using a pretrained ResNet backbone
        if model_name=="deeplabv3plus":
            self.segmentor = DeepLabV3Plus()
        elif model_name=="unet":
            self.segmentor = UnetPlusPlus()
        elif model_name=="segformer":
            self.segmentor = Segformer()


        self.encoder = self.segmentor.encoder
        # replace final layer for fine tuning
        self.decoder = self.segmentor.decoder
        self.latent_dim = self.get_encoding_size()
        print(self.latent_dim)



    def forward(self, X):
        return self.segmentor(X)

    def get_encoding_size(self):
        dummy = torch.zeros((1,3,INPUT_SIZE,INPUT_SIZE))
        return self.get_encoding(dummy).shape[-1]

    def get_encoding(self, X):
        code =  torch.mean(self.segmentor.encoder(X)[-2], [-1, -2]).flatten(1).squeeze(-1)
        return code

    def compute_loss(self, x, y, reduce=True):
        out = self.segmentor(x)
        if reduce: #s
            return self.criterion(out, y)
        else:
            return torch.Tensor([self.criterion(out[i], y[i]) for i in range(out.shape[0])])


    def configure_optimizers(self):
        return self.optimizer(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        x, y, = batch[0], batch[1]
        preds = self(x)
        loss = self.criterion(preds, y)
        acc = torch.mean(iou_score(*get_stats(preds,y, mode="binary", threshold=0.5)))
        # perform logging
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_iou", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss


    def validation_step(self, batch, batch_idx):
        x, y, = batch[0], batch[1]
        preds = self(x)
        loss = self.criterion(preds, y)
        acc = torch.mean(iou_score(*get_stats(preds,y, mode="binary", threshold=0.5)))
        # perform logging
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_iou", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        x, y, = batch[0], batch[1]
        preds = self(x)
        loss = self.criterion(preds, y)
        acc = torch.mean(iou_score(*get_stats(preds,y, mode="binary", threshold=0.5)))
        # perform logging
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_iou", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
