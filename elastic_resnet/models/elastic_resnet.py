import torch
from torch import Tensor
from torch import linalg as LA
import torch.nn as nn
import torch.nn.functional as F

from elastic_resnet.nn import CapNorm2d, ElasticConv2d


def scale_elastic_channels(layer: Tensor, target_num_channels: Tensor) -> Tensor:
    """
    Weights each channel in the layer according to how early the channel is relative to the target_num_channels using a sigmoid.
    Channels before target_num_channels should have a weight close to 1, and after should have weight close to 0
    """
    num_channels = layer.shape[1]
    channel_weights = torch.sigmoid(target_num_channels - torch.arange(num_channels))
    return channel_weights[None, :, None, None] * layer


EXTRA_BLOCK_CHANNELS = 3


class ElasticBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels: float = 2,
        stride=1,
        lock_in_and_out_dims: bool = True,
    ):
        super().__init__()
        self.hidden_channels = nn.Parameter(hidden_channels)
        self.conv1 = ElasticConv2d(
            in_channels,
            int(hidden_channels) + EXTRA_BLOCK_CHANNELS,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = CapNorm2d(hidden_channels + EXTRA_BLOCK_CHANNELS)
        self.conv2 = ElasticConv2d(
            hidden_channels + EXTRA_BLOCK_CHANNELS,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = CapNorm2d(hidden_channels)

        self.shortcut = None
        if stride != 1 or in_channels != out_channels:
            assert (
                not lock_in_and_out_dims
            ), "in_channels and out_channels must match and stride must be 1 unless lock_in_and_out_dims is False"
            self.shortcut = nn.Sequential(
                ElasticConv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                CapNorm2d(out_channels),
            )

    def expand(self):
        num_hidden_channels = int(self.hidden_channels) + EXTRA_BLOCK_CHANNELS
        self.conv1.update_channels(out_channels=num_hidden_channels)
        self.bn1.update_num_features(num_hidden_channels)
        self.conv2.update_channels(in_channels=num_hidden_channels)

    def get_conv_weight_penalty(self):
        # for the hidden channel, conv1 weight dim 0 and conv2 weight dim 1 is penalized.
        num_hidden_channels = self.conv1.weight.shape[0]
        # this weighting is the inverse of the channel weight, so we penalize layers more the further out they are
        channel_penalty_scaling = torch.sigmoid(
            torch.arange(num_hidden_channels) - self.hidden_channels
        )

        # hidden_channels correspond to weight dim 0 for conv1
        conv1_channel_norms = LA.vector_norm(self.conv1.weight, dims=(1, 2, 3))
        conv1_penalty = torch.sum(conv1_channel_norms * channel_penalty_scaling)

        # hidden_channels correspond to weight dim 1 for conv2
        conv2_channel_norms = LA.vector_norm(self.conv2.weight, dims=(0, 2, 3))
        conv2_penalty = torch.sum(conv2_channel_norms * channel_penalty_scaling)

        return conv1_penalty + conv2_penalty
    
    def expand(self):
        for block in self.blocks:
            block.expand()

    def forward(self, x):
        conv1 = scale_elastic_channels(self.conv1(x), self.hidden_channels)
        out = F.relu(self.bn1(conv1))
        out = self.bn2(self.conv2(out))
        if self.shortcut:
            out += self.shortcut(x)
        else:
            out += x
        out = F.relu(out)
        return out


class ElasticResNet(nn.Module):
    def __init__(self, num_blocks, num_classes=10):
        super().__init__()
        self.in_channels = 64
        self.blocks = []

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            block = ElasticBlock(self.in_channels, channels, stride)
            layers.append(block)
            self.blocks.append(block)
            self.in_channels = channels
        return nn.Sequential(*layers)

    def get_conv_weight_penalty(self):
        penalties = [block.get_conv_weight_penalty() for block in self.blocks]
        return torch.sum(penalties)

    def get_hidden_channels_penalty(self):
        # directly penalize the number of hidden channels in the block
        penalties = [block.hidden_channels for block in self.blocks]
        return torch.sum(penalties)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ElasticResNet18():
    return ElasticResNet([2, 2, 2, 2])
