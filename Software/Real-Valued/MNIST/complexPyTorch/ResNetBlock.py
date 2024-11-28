import torch
import torch.nn as nn
import torch.nn.functional as F

from complexLayers import ComplexBatchNorm2d, ComplexConv2d, ComplexLinear, ComplexReLU
from complexFunctions import complex_relu, complex_max_pool2d

class ResidualBlock(nn.Module):
    def init(self, in_channels, featmaps, filter_size, stage, block, shortcut, spectral_pool_scheme, model):
        super(ResidualBlock, self).__init__()
        self.model = model
        self.spectral_pool_scheme = spectral_pool_scheme
        self.shortcut = shortcut
        nb_fmaps1, nb_fmaps2 = featmaps
        convArgs = {}  # Define convolution arguments here
        bnArgs = {}  # Define batch normalization arguments here

        if self.model == "real":
            self.bn1 = nn.BatchNorm2d(in_channels, **bnArgs)
            self.conv1 = nn.Conv2d(in_channels, nb_fmaps1, filter_size, padding=(filter_size // 2, filter_size // 2), **convArgs)
        elif self.model == "complex":
            self.bn1 = ComplexBatchNorm2d(in_channels, **bnArgs)
            self.conv1 = ComplexConv2d(in_channels, nb_fmaps1, filter_size, **convArgs)

        # Similar initialization for the second set of layers
        if self.model == "real":
            self.bn2 = nn.BatchNorm2d(nb_fmaps1, **bnArgs)
            self.conv2 = nn.Conv2d(nb_fmaps1, nb_fmaps2, filter_size, padding=(filter_size // 2, filter_size // 2), **convArgs)
        elif self.model == "complex":
            self.bn2 = ComplexBatchNorm2d(nb_fmaps1, **bnArgs)
            self.conv2 = ComplexConv2d(nb_fmaps1, nb_fmaps2, filter_size, **convArgs)

        if self.shortcut == 'projection':
            self.shortcut_conv = nn.Conv2d(in_channels, nb_fmaps2, 1, stride=2 if spectral_pool_scheme != "nodownsample" else 1)

    def forward(self, x):
        identity = x

        out = self.bn1(x)
        out = F.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = F.relu(out)
        out = self.conv2(out)

        if self.shortcut == 'regular':
            out += identity
        elif self.shortcut == 'projection':
            identity = self.shortcut_conv(identity)
            out += identity

        out = F.relu(out)
        return out

# Example usage
# residual_block = ResidualBlock(in_channels=64, featmaps=(64, 64), filter_size=3, stage=2, block='a', shortcut='regular', spectral_pool_scheme='nodownsample', model='real')
