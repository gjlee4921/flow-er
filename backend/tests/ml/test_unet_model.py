"""The U-Nets keep the 101 x 101 image size from input to logits."""

import pytest

torch = pytest.importorskip("torch")

from flow_er.ml.unet.model import UNet, pretrained_unet


def test_unito_unet_shape():
    model = UNet(in_channels=1).eval()
    with torch.no_grad():
        out = model(torch.zeros(1, 1, 101, 101))
    assert out.shape == (1, 1, 101, 101)


def test_unito_unet_matches_spec_widths():
    model = UNet()
    assert [b[0].out_channels for b in model.down] == [64, 128, 256, 512]
    assert model.bottom[0].out_channels == 1024
    assert all(b[0].bias is None for b in model.down)   # convs without bias


def test_pretrained_unet_shape_without_download():
    pytest.importorskip("segmentation_models_pytorch")
    model = pretrained_unet(in_channels=3, encoder_weights=None).eval()
    with torch.no_grad():
        out = model(torch.zeros(2, 3, 101, 101))
    assert out.shape == (2, 1, 101, 101)


@pytest.mark.parametrize("channels", [3, 2])
def test_shared_unet_shape(channels):
    pytest.importorskip("segmentation_models_pytorch")
    from flow_er.ml.unet.model import build, shared_arch

    model = build(shared_arch(channels, encoder_weights=None)).eval()
    with torch.no_grad():
        out = model(torch.zeros(1, channels, 101, 101))
    assert out.shape == (1, 1, 101, 101)


def test_build_unito_from_recipe_arch():
    from flow_er.ml.unet.model import build, unito_arch

    model = build(unito_arch((8, 16, 32, 64), 128)).eval()
    with torch.no_grad():
        assert model(torch.zeros(1, 1, 101, 101)).shape == (1, 1, 101, 101)
