from typing import Optional
import torch
from torch.nn.functional import batch_norm
from torch import Tensor


def cap_norm(
    input: Tensor,
    running_mean: Optional[Tensor] = None,
    running_var: Optional[Tensor] = None,
    training: bool = False,
    momentum: float = 0.1,
) -> Tensor:
    """
    Batch norm with no weight or bias, which only normalizes variance if it's greater than 1

    Philipp, George, and Jaime G. Carbonell. "Nonparametric neural networks." arXiv preprint arXiv:1712.05440 (2017).
    """
    remove_dims = [dim for dim in range(input.dim()) if dim != 1]

    var = running_var
    mean = running_mean

    if training or var is None:
        var = input.var(remove_dims, unbiased=False)
    if training or mean is None:
        mean = input.mean(remove_dims)

    if training:
        with torch.no_grad():
            # stats tracking inspired by https://github.com/ptrblck/pytorch_misc/blob/master/batch_norm_manual.py
            n = input.numel() / input.size(1)
            if running_mean is not None:
                running_mean.data = momentum * mean + (1 - momentum) * running_mean.data
            if running_var is not None:
                # update running_var with unbiased var
                running_var.data = (
                    momentum * var * n / (n - 1) + (1 - momentum) * running_var.data
                )

    capped_var = torch.clamp(var, min=1.0)
    return (input - mean[None, :, None, None]) / torch.sqrt(
        capped_var[None, :, None, None]
    )
