import torch
import os
import torch.nn as nn


class TransferNet(nn.Module):
    def __init__(self, num_residual):
        super(TransferNet, self).__init__()

        in_channels = 3
        out_channels = 3
        feature_cfg = [32, 64, 128]
        self.num_residual = num_residual
        expand_cfg = [64, 32, 3]

        self.upsample_model = []
        self.residual_model = []
        self.expand_model = []
        for layer in feature_cfg:
            out_channels = layer
            self.upsample_model += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1),
                           nn.InstanceNorm2d(out_channels, affine=True)]
            in_channels = out_channels

        for i in num_residual:
            self.residual_model += [ResidualNetwork(out_channels)]

        for layer in expand_cfg:
            out_channels = layer
            self.expand_model += [nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1)]

        self.upsample_model = nn.Sequential(*self.upsample_model)
        self.residual_model = nn.Sequential(*self.residual_model)
        self.expand_model = nn.Sequential(*self.expand_model)

    def forward(self, x):
        x = self.upsample_model(x)
        x = self.residual_model(x)
        x = self.expand_model(x)
        return x


class ResidualNetwork(nn.Module):
    def __init__(self, channels):
        super(ResidualNetwork, self).__init__()
        block = []
        for i in range(2):
            block += [nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1),
                           nn.InstanceNorm2d(channels),
                           nn.ReLU()]
        self.model = nn.Sequential(*(block[:-1]))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.model(x)
        out = self.relu(x + out)
        return out

