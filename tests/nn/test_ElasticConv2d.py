import torch
from elastic_resnet.nn import ElasticConv2d


def test_elastic_conv2d_leave_channels_the_same():
    conv = ElasticConv2d(in_channels=10, out_channels=15, kernel_size=3, padding=1)
    have_dims_changed = conv.update_channels(in_channels=10, out_channels=15)
    assert not have_dims_changed


def test_elastic_conv2d_shrink_in_channels():
    conv = ElasticConv2d(in_channels=10, out_channels=15, kernel_size=3, padding=1)

    have_dims_changed = conv.update_channels(8, 15)

    assert have_dims_changed
    assert conv.weight.shape == torch.Size([15, 8, 3, 3])
    assert conv.bias.shape == torch.Size([15])
    assert conv.in_channels == 8
    assert conv.out_channels == 15


def test_elastic_conv2d_expand_in_channels():
    conv = ElasticConv2d(in_channels=10, out_channels=15, kernel_size=3, padding=1)

    have_dims_changed = conv.update_channels(in_channels=13)

    assert have_dims_changed
    assert conv.weight.shape == torch.Size([15, 13, 3, 3])
    assert conv.bias.shape == torch.Size([15])
    assert conv.in_channels == 13
    assert conv.out_channels == 15


def test_elastic_conv2d_shrink_out_channels():
    conv = ElasticConv2d(in_channels=10, out_channels=15, kernel_size=3, padding=1)

    have_dims_changed = conv.update_channels(8, 11)

    assert have_dims_changed
    assert conv.weight.shape == torch.Size([11, 8, 3, 3])
    assert conv.bias.shape == torch.Size([11])
    assert conv.in_channels == 8
    assert conv.out_channels == 11


def test_elastic_conv2d_expand_out_channels():
    conv = ElasticConv2d(in_channels=10, out_channels=15, kernel_size=3, padding=1)

    have_dims_changed = conv.update_channels(out_channels=16)

    assert have_dims_changed
    assert conv.weight.shape == torch.Size([16, 10, 3, 3])
    assert conv.bias.shape == torch.Size([16])
    assert conv.in_channels == 10
    assert conv.out_channels == 16
