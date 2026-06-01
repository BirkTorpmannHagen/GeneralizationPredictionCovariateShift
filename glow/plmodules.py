import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch.nn as nn
from torch import log
from glow.model import *
from torch.optim import SGD, Adam

def calc_loss(log_p, logdet, image_size, n_bins):
    # log_p = calc_log_p([z_list])
    n_pixel = image_size * image_size * 3
    loss = -log(n_bins) * n_pixel
    loss = loss + logdet + log_p
    return (
        (-loss / (log(2) * n_pixel)).mean(),
        (log_p / (log(2) * n_pixel)).mean(),
        (logdet / (log(2) * n_pixel)).mean(),
    )


class GlowLoss(nn.Module):

    def __init__(self, image_size, n_bins):
        super().__init__()
        self.image_size = image_size
        self.n_bins = n_bins
        pass

    def forward(self, log_p, logdet):
        loss, log_p, log_det =  calc_loss(log_p, logdet, self.image_size, self.n_bins)
        return loss



class GlowPL(pl.LightningModule):
    def __init__(self, in_channel, n_flow, n_block, affine, conv_lu,
                 optimizer='adam',n_bins = 2.0 ** 5, lr=1e-6, img_size=32):
        super().__init__()
        
        self.__dict__.update(locals())
        optimizers = {'adam': Adam, 'sgd': SGD}
        self.optimizer = optimizers[optimizer]
        self.img_size = img_size
        # instantiate loss criterion
        self.criterion = GlowLoss(img_size, n_bins)
        #glow model
        self.glow = Glow(in_channel, n_flow, n_block, affine, conv_lu)

    def on_train_start(self):
        self.train()

    def forward(self, X):
        return self.glow(X)

    def get_encoding_size(self, depth):
        dummy = torch.zeros((1, 3,self.img_size, self.img_size))
        return self.glow.get_encoding(dummy).shape[-1]

    def get_encoding(self, X):
        return self.glow.get_encoding(X)

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 100, 2)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

    def training_step(self, batch, batch_idx):
        x = batch[0]
        image = x * 255
        image = torch.floor(image / 2 ** (8 - 5))
        image = image / self.n_bins - 0.5

        # Check for NaNs or Infs
        if torch.isnan(image).any() or torch.isinf(image).any():
            print("NaN or Inf detected in input image!")
            return None  # Skip step to avoid crash

        log_p, logdet, _ = self.glow(image + torch.rand_like(image) / self.n_bins)

        # Debugging: Check for NaN or Inf in log_p and logdet
        if torch.isnan(log_p).any() or torch.isinf(log_p).any():
            print("NaN or Inf detected in log_p!")
            print("log_p values:", log_p)
            return None

        if torch.isnan(logdet).any() or torch.isinf(logdet).any():
            print("NaN or Inf detected in logdet!")
            print("logdet values:", logdet)
            return None

        loss = self.criterion(log_p, logdet)

        # Debugging: Check for NaN or Inf in loss
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            print("NaN or Inf detected in loss!")
            print("Loss values:", loss)
            return None

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch[0]
        image = x * 255
        image = torch.floor(image / 2 ** (8 - 5))
        image = image / self.n_bins - 0.5

        log_p, logdet, _ = self.glow(image + torch.rand_like(x) / self.n_bins)
        loss = self.criterion(log_p, logdet)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        x = batch[0]

        image = x * 255
        image = torch.floor(image / 2 ** (8 - 5))
        image = image / self.n_bins - 0.5

        log_p, logdet, _ = self.glow(image + torch.rand_like(x) / self.n_bins)
        loss = self.criterion(log_p, logdet)
        self.log("test_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        # perform logging

    def estimate_log_likelihood(self, img):
        return self.glow.estimate_log_likelihood(img)