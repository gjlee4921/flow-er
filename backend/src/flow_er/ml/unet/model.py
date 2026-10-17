"""U-Net definitions (docs/unito-spec.md).

UNet: the per-gate model of UNITO as published (spec §1), a standard U-Net (Ronneberger et al.
2015) implemented from the spec: encoder 64-128-256-512, bottleneck 1024, transposed-conv decoder
with skip connections, 1x1 conv to one channel of logits, input and output 101 x 101.

pretrained_unet: the flow-er "pretrained encoder" variant (spec §2), segmentation_models_pytorch
with an ImageNet ResNet-18 encoder; 101 x 101 inputs are padded to 128 x 128 and cropped back.
It is the shared template-refining U-Net: inputs [density, template, unstim] (or without unstim).

build(arch) rebuilds either from the "arch" dict stored in a model's recipe.
"""

import torch
import torch.nn.functional as F
from torch import nn

WIDTHS = (64, 128, 256, 512)
BOTTLENECK = 1024


def double_conv(cin: int, cout: int) -> nn.Sequential:
    """Two (3x3 conv, stride 1, padding 1, no bias -> BatchNorm -> ReLU)."""
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, in_channels: int = 1, widths: tuple[int, ...] = WIDTHS,
                 bottleneck: int = BOTTLENECK):
        super().__init__()
        self.down = nn.ModuleList()
        cin = in_channels
        for w in widths:
            self.down.append(double_conv(cin, w))
            cin = w
        self.pool = nn.MaxPool2d(2)
        self.bottom = double_conv(cin, bottleneck)
        self.up, self.dec = nn.ModuleList(), nn.ModuleList()
        cin = bottleneck
        for w in reversed(widths):
            self.up.append(nn.ConvTranspose2d(cin, w, 2, stride=2))
            self.dec.append(double_conv(2 * w, w))
            cin = w
        self.head = nn.Conv2d(cin, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for block in self.down:
            x = block(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottom(x)
        for up, dec, skip in zip(self.up, self.dec, reversed(skips)):
            x = up(x)
            if x.shape[-2:] != skip.shape[-2:]:     # 101 is odd: match the skip's size
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = dec(torch.cat([skip, x], dim=1))
        return self.head(x)


class PaddedSMPUnet(nn.Module):
    """smp.Unet on 128 x 128: pads the 101 x 101 input and crops the logits back."""

    SIZE = 128

    def __init__(self, in_channels: int, encoder: str = "resnet18",
                 encoder_weights: str | None = "imagenet"):
        super().__init__()
        import segmentation_models_pytorch as smp

        self.net = smp.Unet(encoder, encoder_weights=encoder_weights, in_channels=in_channels,
                            classes=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        x = F.pad(x, (0, self.SIZE - w, 0, self.SIZE - h))
        return self.net(x)[..., :h, :w]


def pretrained_unet(in_channels: int, encoder_weights: str | None = "imagenet") -> nn.Module:
    return PaddedSMPUnet(in_channels, encoder_weights=encoder_weights)


def build(arch: dict) -> nn.Module:
    """{"name": "unito", "in_channels", "widths", "bottleneck"} or
    {"name": "smp_unet", "in_channels", "encoder", "encoder_weights"}."""
    if arch["name"] == "unito":
        return UNet(arch.get("in_channels", 1), tuple(arch.get("widths", WIDTHS)),
                    arch.get("bottleneck", BOTTLENECK))
    if arch["name"] == "smp_unet":
        return PaddedSMPUnet(arch["in_channels"], arch.get("encoder", "resnet18"),
                             arch.get("encoder_weights"))
    raise ValueError(f"unknown architecture {arch['name']!r}")


def unito_arch(widths: tuple[int, ...] = WIDTHS, bottleneck: int = BOTTLENECK) -> dict:
    return {"name": "unito", "in_channels": 1, "widths": list(widths), "bottleneck": bottleneck}


def shared_arch(in_channels: int = 3, encoder_weights: str | None = "imagenet") -> dict:
    return {"name": "smp_unet", "in_channels": in_channels, "encoder": "resnet18",
            "encoder_weights": encoder_weights}
