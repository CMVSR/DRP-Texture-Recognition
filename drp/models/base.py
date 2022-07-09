"""Base model class for all models."""

import torch.nn as nn

from drp import const


class BaseDRP(nn.Module):
    """Base model class for all models."""

    def init_block1(
        self,
        dense_feature_dim=const.BLOCK1_DENSE_FEATURE_DIM,
        dropout_ratio=const.DROPOUT_RATIO,
    ):
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=dense_feature_dim,
                out_channels=dense_feature_dim,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.Dropout2d(p=dropout_ratio),
            nn.BatchNorm2d(dense_feature_dim),
        )
        self.sigmoid = nn.Sigmoid()
        self.relu1 = nn.Sigmoid()
        self.relu2 = nn.ReLU(inplace=True)
        self.norm1 = nn.BatchNorm2d(dense_feature_dim)
        self.relu3 = nn.ReLU(inplace=True)

    def init_block2(
        self,
        dense_feature_dim=const.BLOCK2_DENSE_FEATURE_DIM,
        dropout_ratio=const.DROPOUT_RATIO,
    ):
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=dense_feature_dim,
                out_channels=dense_feature_dim,
                kernel_size=1,
                stride=1,
                padding=0
            ),
            nn.Dropout2d(p=dropout_ratio),
            nn.BatchNorm2d(dense_feature_dim),
        )
        self.sigmoid1 = nn.Sigmoid()
        self.relu4 = nn.Sigmoid()
        self.relu5 = nn.ReLU(inplace=True)
        self.norm2 = nn.BatchNorm2d(dense_feature_dim)
        self.relu6 = nn.ReLU(inplace=True)
