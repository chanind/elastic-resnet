from typing import Optional
import torch
from torch import Tensor
from torch.nn import Module

from elastic_resnet.nn.functional import cap_norm

# code based on https://pytorch.org/docs/stable/_modules/torch/nn/modules/batchnorm.html#BatchNorm2d


class _CapNorm(Module):
    _version = 2
    __constants__ = ["track_running_stats", "momentum", "eps", "num_features", "affine"]
    num_features: int
    eps: float
    momentum: float
    affine: bool
    track_running_stats: bool
    # WARNING: weight and bias purposely not defined here.
    # See https://github.com/pytorch/pytorch/issues/39670

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        track_running_stats: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        self.factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.track_running_stats = track_running_stats
        self.setup_running_stats()
        self.reset_parameters()

    def reset_running_stats(self) -> None:
        if self.track_running_stats:
            # running_mean/running_var/num_batches... are registered at runtime depending
            # if self.track_running_stats is on
            self.running_mean.zero_()  # type: ignore[union-attr]
            self.running_var.fill_(1)  # type: ignore[union-attr]
            self.num_batches_tracked.zero_()  # type: ignore[union-attr,operator]

    def setup_running_stats(self) -> None:
        if self.track_running_stats:
            self.register_buffer(
                "running_mean", torch.zeros(self.num_features, **self.factory_kwargs)
            )
            self.register_buffer(
                "running_var", torch.ones(self.num_features, **self.factory_kwargs)
            )
            self.running_mean: Optional[Tensor]
            self.running_var: Optional[Tensor]
            self.register_buffer(
                "num_batches_tracked",
                torch.tensor(
                    0,
                    dtype=torch.long,
                    **{k: v for k, v in self.factory_kwargs.items() if k != "dtype"}
                ),
            )
            self.reset_running_stats()
        else:
            self.register_buffer("running_mean", None)
            self.register_buffer("running_var", None)
            self.register_buffer("num_batches_tracked", None)

    def reset_parameters(self) -> None:
        self.reset_running_stats()

    def _check_input_dim(self, input):
        raise NotImplementedError

    def extra_repr(self):
        return (
            "{num_features}, eps={eps}, momentum={momentum}, affine={affine}, "
            "track_running_stats={track_running_stats}".format(**self.__dict__)
        )

    def forward(self, input: Tensor) -> Tensor:
        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:  # type: ignore[has-type]
                self.num_batches_tracked = self.num_batches_tracked + 1  # type: ignore[has-type]
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        r"""
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        r"""
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        return cap_norm(
            input,
            # If buffers are not to be tracked, ensure that they won't be updated
            self.running_mean
            if not self.training or self.track_running_stats
            else None,
            self.running_var if not self.training or self.track_running_stats else None,
            training=bn_training,
            momentum=exponential_average_factor,
            eps=self.eps,
        )

    def update_num_features(self, num_features: int) -> None:
        """Change the number of features during training"""
        assert self.training, "Can only update the number of features in training mode"
        if num_features != self.num_features:
            # make sure device is set if present
            if self.running_mean is not None and self.factory_kwargs["device"] is None:
                self.factory_kwargs["device"] = self.running_mean.device
            self.num_features = num_features
            self.setup_running_stats()


class CapNorm2d(_CapNorm):
    r"""
    replaces each element with `z - mean / max(1, signma)` during training. During test, functionality is identical to BatchNorm2d sans weight and bias

    Philipp, George, and Jaime G. Carbonell. "Nonparametric neural networks." arXiv preprint arXiv:1712.05440 (2017).
    """

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError("expected 4D input (got {}D input)".format(input.dim()))
