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
    eps: float = 1e-5,
) -> Tensor:
    """
    Batch norm with no weight or bias, which only normalizes variance if it's greater than 1

    Philipp, George, and Jaime G. Carbonell. "Nonparametric neural networks." arXiv preprint arXiv:1712.05440 (2017).
    """
    if training:
        remove_dims = [dim for dim in range(input.dim()) if dim != 1]
        target_shape = (1, -1, *[1 for _ in range(input.dim() - 2)])
        var = torch.var(input, dim=remove_dims, unbiased=False).view(target_shape)
        mean = torch.mean(input, dim=remove_dims).view(target_shape)

        capped_var = torch.clamp(var, min=1.0)
        return (input - mean) / torch.sqrt(capped_var)

    return batch_norm(
        input,
        running_mean,
        torch.clamp(running_var, min=1.0) if running_var else None,
        training=training,
        momentum=momentum,
        eps=eps,
    )
