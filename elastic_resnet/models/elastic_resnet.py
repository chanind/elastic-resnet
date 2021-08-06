import torch
from torch import Tensor
from torch import linalg as LA
import torch.nn as nn
import torch.nn.functional as F

from elastic_resnet.nn import CapNorm2d, ElasticConv2d


EXTRA_BLOCK_CHANNELS = 5


class ElasticBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        initial_hidden_channels: float = 2.0,
        stride=1,
    ):
        super().__init__()
        self.hidden_channels = nn.Parameter(
            torch.tensor(initial_hidden_channels, dtype=torch.float)
        )
        expanded_hidden_channels = int(initial_hidden_channels) + EXTRA_BLOCK_CHANNELS
        self.conv1 = ElasticConv2d(
            in_channels,
            expanded_hidden_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = CapNorm2d(expanded_hidden_channels)
        self.conv2 = ElasticConv2d(
            expanded_hidden_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = CapNorm2d(out_channels)

        self.shortcut = None
        if stride != 1 or in_channels != out_channels:
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

    def resize(self) -> bool:
        """
        Adjust block size, and return a bool indicating whether any actual resizing was necessary
        """
        num_hidden_channels = int(self.hidden_channels) + EXTRA_BLOCK_CHANNELS
        return any(
            [
                self.conv1.update_channels(
                    out_channels=num_hidden_channels, incoming_channels_scale=0.01
                ),
                self.bn1.update_num_features(num_hidden_channels),
                self.conv2.update_channels(
                    in_channels=num_hidden_channels, incoming_channels_scale=0.01
                ),
            ]
        )

    def get_hidden_channel_caps(self):
        num_channels = self.bn1.num_features
        channel_caps = torch.sigmoid(
            self.hidden_channels
            - torch.arange(num_channels, device=self.hidden_channels.device)
        )
        return channel_caps

    def forward(self, x):
        conv1 = self.conv1(x)
        out = F.relu(self.bn1(conv1))
        out = self.bn2(self.conv2(out))
        if self.shortcut:
            out += self.shortcut(x)
        else:
            out += x
        out = F.relu(out)
        return out

    def get_conv_weight_penalty(self):
        channel_caps = self.get_hidden_channel_caps()
        conv1_penalty = self.conv1.get_out_channel_weight_penalty(channel_caps)
        conv2_penalty = self.conv2.get_in_channel_weight_penalty(channel_caps)
        return conv1_penalty + conv2_penalty


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
            block = ElasticBlock(self.in_channels, channels, stride=stride)
            layers.append(block)
            self.blocks.append(block)
            self.in_channels = channels
        return nn.Sequential(*layers)

    def get_hidden_channels_penalty(self):
        # directly penalize the number of hidden channels in the block
        penalties = torch.stack([block.hidden_channels for block in self.blocks])
        return torch.sum(penalties)

    def resize(self) -> bool:
        """
        Adjust the network size and return a bool indicating whether anything actually changed
        """
        return any([block.resize() for block in self.blocks])

    def clip_weights(self):
        [block.clip_channel_weights() for block in self.blocks]

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

    def get_conv_weight_penalty(self):
        penalties = torch.stack(
            [block.get_conv_weight_penalty() for block in self.blocks]
        )
        return torch.sum(penalties)


def ElasticResNet18():
    return ElasticResNet([2, 2, 2, 2])
