

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

class ResNetClassifier(Classifier):
    def __init__(self, num_classes, resnet_version,
                 optimizer='adam', lr=1e-6, batch_size=16):
        super().__init__(num_classes, optimizer, lr, batch_size)

        self.__dict__.update(locals())
        resnets = {
            18: models.resnet18, 34: models.resnet34,
            50: models.resnet50, 101: models.resnet101,
            152: models.resnet152
        }
        self.num_classes = num_classes
        optimizers = {'adam': Adam, 'sgd': SGD}
        self.optimizer = optimizers[optimizer]
        # instantiate loss criterion
        self.criterion = nn.CrossEntropyLoss(reduction="none")
        # Using a pretrained ResNet backbone
        self.model = resnets[resnet_version](pretrained=True)
        # Replace old FC layer with Identity so we can train our own
        linear_size = list(self.model.children())[-1].in_features
        # replace final layer for fine tuning
        self.model.fc = nn.Linear(linear_size, num_classes)

        self.latent_dim = self.get_encoding_size(-2)
        self.acc = Accuracy(task="multiclass", num_classes=num_classes, top_k=1)


