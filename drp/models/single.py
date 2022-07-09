"""DRP single layer."""

import torch.nn as nn
from torchvision import models

from drp import const
from drp.models.base import BaseDRP


class myModel(BaseDRP):
    """DRP Single Layer."""

    def __init__(self, num_classes=1000, **kwargs):
        super(myModel, self).__init__()
        model_dense = models.densenet161(pretrained=True)
        self.num_classes = num_classes

        self.features = nn.Sequential(
            *list(model_dense.features.children())[:-1],
        )
        self.features1 = nn.Sequential(
            *list(model_dense.features.children())[:-2],
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=2208,
                out_channels=1104,
                kernel_size=1,
                stride=1,
                padding=0
            ),
            nn.Dropout2d(p=0.5),
            nn.BatchNorm2d(1104),
        )
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.norm1 = nn.BatchNorm2d(4416)
        self.relu3 = nn.ReLU(inplace=True)
        self.classifier = nn.Linear((4416), self.num_classes)

        self.init_block1()
        self.init_block2()
        self.init_classifiers()

    def init_classifiers(
        self,
        dense_feature_dim1=const.BLOCK1_DENSE_FEATURE_DIM,
    ):
        self.classifier = nn.Linear((dense_feature_dim1), self.num_classes)

    def forward(self, x):
        out = self.features(x)
        identity = out
        ## Residual pooling layer ##
        ## 1. Residual encoding module ##
        identity = self.sigmoid(identity)
        out = self.conv1(out)
        out = self.relu1(out)
        out = out - identity
        ## 2. Aggregation module ##
        out = self.relu2(out)
        out = self.norm1(out)
        out = self.relu3(out)
        out = nn.functional.adaptive_avg_pool2d(
            out, (1, 1)).view(out.size(0), -1)
        x = self.classifier(out)
        return x
