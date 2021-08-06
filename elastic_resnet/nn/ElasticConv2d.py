import math
from typing import Optional
import torch
from torch import Tensor
from torch.nn import Conv2d, init, Parameter, ReLU


class ElasticConv2d(Conv2d):
    # This feels slightly sketchy, since pytorch might change the way Conv2d is implemented in a future version...
    def forward(
        self,
        input: Tensor,
        in_channel_caps: Optional[Tensor] = None,
        out_channel_caps: Optional[Tensor] = None,
    ) -> Tensor:
        weight = self.weight
        if in_channel_caps is not None:
            weight = torch.minimum(weight, in_channel_caps[None, :, None, None])
        if out_channel_caps is not None:
            weight = torch.minimum(weight, out_channel_caps[:, None, None, None])
        return self._conv_forward(input, weight, self.bias)

    def get_in_channel_weight_penalty(
        self,
        channel_caps: Tensor,
    ):
        weight = self.weight
        # only penalize differences > 0, so use relu to cut off the negatives
        excess_weights = ReLU(weight - channel_caps[None, :, None, None], inplace=True)
        return torch.sum(excess_weights)

    def get_out_channel_weight_penalty(
        self,
        channel_caps: Tensor,
    ):
        weight = self.weight
        # only penalize differences > 0, so use relu to cut off the negatives
        excess_weights = ReLU(weight - channel_caps[:, None, None, None], inplace=True)
        return torch.sum(excess_weights)

    def update_channels(
        self,
        in_channels: int = None,
        out_channels: int = None,
        incoming_channels_scale: float = 1.0,
    ) -> bool:
        """
        Updates the number of in and out channels, and returns a bool indicating whether any changes were necessary
        """
        new_in_channels = in_channels if in_channels else self.in_channels
        new_out_channels = out_channels if out_channels else self.out_channels
        if new_in_channels % self.groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if new_out_channels % self.groups != 0:
            raise ValueError("out_channels must be divisible by groups")
        if self.transposed:
            raise ValueError(
                "Updating the number of channels for transposed convolution isn't current supported"
            )

        if (
            new_in_channels == self.in_channels
            and new_out_channels == self.out_channels
        ):
            # no changes needed, exit here
            return False
        with torch.no_grad():
            device = self.weight.device
            if new_in_channels < self.in_channels:
                # no need to do weight init when shrinking dims
                self.weight = Parameter(
                    self.weight.data[:, : (new_in_channels // self.groups), :]
                    .detach()
                    .to(device),
                )
            elif new_in_channels > self.in_channels:
                num_new_channels = new_in_channels - self.in_channels
                new_weight_channels = torch.empty(
                    (self.weight.shape[0], num_new_channels, *self.weight.shape[2:]),
                    device=device,
                )
                # taken from https://pytorch.org/docs/stable/_modules/torch/nn/modules/conv.html#Conv2d
                init.kaiming_uniform_(new_weight_channels, a=math.sqrt(5))
                self.weight = Parameter(
                    torch.cat(
                        [
                            self.weight.data,
                            new_weight_channels * incoming_channels_scale,
                        ],
                        dim=1,
                    )
                    .detach()
                    .to(device),
                )

            if new_out_channels < self.out_channels:
                # no need to do weight init when shrinking dims
                self.weight = Parameter(
                    self.weight.data[:(new_out_channels), :, :].detach().to(device),
                )
                if self.bias is not None:
                    self.bias = Parameter(
                        self.bias.data[:(new_out_channels)].detach().to(device),
                    )
            elif new_out_channels > self.out_channels:
                num_new_channels = new_out_channels - self.out_channels
                new_weight_channels = torch.empty(
                    (num_new_channels, self.weight.shape[1], *self.weight.shape[2:]),
                    device=device,
                )
                # taken from https://pytorch.org/docs/stable/_modules/torch/nn/modules/conv.html#Conv2d
                init.kaiming_uniform_(new_weight_channels, a=math.sqrt(5))
                self.weight = Parameter(
                    torch.cat(
                        [
                            self.weight.data,
                            new_weight_channels * incoming_channels_scale,
                        ],
                        dim=0,
                    )
                    .detach()
                    .to(device),
                )
                if self.bias is not None:
                    fan_in, _ = init._calculate_fan_in_and_fan_out(new_weight_channels)
                    bound = 1 / math.sqrt(fan_in)
                    new_bias_channels = torch.empty((num_new_channels))
                    init.uniform_(new_bias_channels, -bound, bound)
                    self.bias = Parameter(
                        torch.cat(
                            [
                                self.bias.data,
                                new_bias_channels * incoming_channels_scale,
                            ]
                        )
                        .detach()
                        .to(device),
                    )

        self.in_channels = new_in_channels
        self.out_channels = new_out_channels
        return True
