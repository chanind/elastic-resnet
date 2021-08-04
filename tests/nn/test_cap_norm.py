import torch
from torch.nn.functional import batch_norm
from elastic_resnet.nn.functional import cap_norm


def test_cap_norm_with_high_variance_matches_batch_norm():
    high_var_input = torch.rand((5, 3, 16, 16)) * 100.0

    assert torch.allclose(
        batch_norm(high_var_input, None, None, training=True),
        cap_norm(high_var_input, training=True),
    )


def test_cap_norm_with_low_variance_is_smaller_than_batch_norm():
    low_var_input = torch.rand((5, 3, 8, 8)) * 0.05

    batch_norm_res = batch_norm(low_var_input, None, None, training=True)
    cap_norm_res = cap_norm(low_var_input, training=True)

    assert torch.norm(batch_norm_res) > torch.norm(cap_norm_res)
