

import sys
import warnings
from pathlib import Path
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np

from classifier.classifier_base import Classifier

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

class ViTClassifier(Classifier):
    def __init__(self, num_classes,
                 optimizer='adam', lr=1e-6, batch_size=16,
                 img_shape=224):
        super().__init__(num_classes, optimizer, lr, batch_size, img_shape=224)

        self.__dict__.update(locals())

        self.num_classes = num_classes
        optimizers = {'adam': Adam, 'sgd': SGD}
        self.optimizer = optimizers[optimizer]
        self.criterion = nn.CrossEntropyLoss(reduction="none")

        self.model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)

        # Replace classifier head (ViT uses .heads, not .fc)
        in_features = self.model.heads.head.in_features
        self.model.heads.head = nn.Linear(in_features, num_classes)

        self.latent_dim = self.get_encoding_size(-1)
        print(self.latent_dim)
        self.acc = Accuracy(task="multiclass", num_classes=num_classes, top_k=1)

    def get_encoding(self, X, depth=-2):
        m = self.model
        x = m._process_input(X)  # [B, 196, 768] for 224×224, patch=16
        n = x.shape[0]
        cls = m.class_token.expand(n, -1, -1)  # [B, 1, 768]
        x = torch.cat((cls, x), dim=1)  # [B, 197, 768]
        x = m.encoder(x)  # adds pos emb, dropout, blocks
        x = m.encoder.ln(x)  # final LayerNorm
        return x[:, 0]