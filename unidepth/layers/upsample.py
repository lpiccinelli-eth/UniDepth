"""
Author: Luigi Piccinelli
Licensed under the CC-BY NC 4.0 license (http://creativecommons.org/licenses/by-nc/4.0/)
"""

import torch
import torch.nn as nn
from einops import rearrange

from .convnext import CvnxtBlock


class ConvUpsample(nn.Module):
    def __init__(
        self,
        hidden_dim,
        num_layers: int = 2,
        expansion: int = 4,
        layer_scale: float = 1.0,
        kernel_size: int = 7,
        **kwargs,
    ):
        super().__init__()
        self.convs = nn.ModuleList([])
        for _ in range(num_layers):
            self.convs.append(
                CvnxtBlock(
                    hidden_dim,
                    kernel_size=kernel_size,
                    expansion=expansion,
                    layer_scale=layer_scale,
                )
            )
        self.up = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=1, padding=0),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(hidden_dim // 2, hidden_dim // 2, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor):
        for conv in self.convs:
            x = conv(x)
        x = self.up(x)
        x = rearrange(x, "b c h w -> b (h w) c")
        return x


class ConvUpsampleShuffle(nn.Module):
    def __init__(
        self,
        hidden_dim,
        num_layers: int = 2,
        expansion: int = 4,
        layer_scale: float = 1.0,
        kernel_size: int = 7,
        **kwargs,
    ):
        super().__init__()
        self.convs = nn.ModuleList([])
        for _ in range(num_layers):
            self.convs.append(
                CvnxtBlock(
                    hidden_dim,
                    kernel_size=kernel_size,
                    expansion=expansion,
                    layer_scale=layer_scale,
                )
            )
        self.up = nn.Sequential(
            nn.PixelShuffle(2),
            nn.Conv2d(hidden_dim // 4, hidden_dim // 2, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor):
        for conv in self.convs:
            x = conv(x)
        x = self.up(x)
        x = rearrange(x, "b c h w -> b (h w) c")
        return x


class ConvUpsampleShuffleResidual(nn.Module):
    def __init__(
        self,
        hidden_dim,
        num_layers: int = 2,
        expansion: int = 4,
        layer_scale: float = 1.0,
        kernel_size: int = 7,
        padding_mode: str = "zeros",
        **kwargs,
    ):
        super().__init__()
        self.convs = nn.ModuleList([])
        for _ in range(num_layers):
            self.convs.append(
                CvnxtBlock(
                    hidden_dim,
                    kernel_size=kernel_size,
                    expansion=expansion,
                    layer_scale=layer_scale,
                    padding_mode=padding_mode,
                )
            )
        self.up = nn.Sequential(
            nn.PixelShuffle(2),
            nn.Conv2d(
                hidden_dim // 4,
                hidden_dim // 4,
                kernel_size=7,
                padding=3,
                padding_mode=padding_mode,
                groups=hidden_dim // 4,
            ),
            nn.ReLU(),
            nn.Conv2d(
                hidden_dim // 4,
                hidden_dim // 2,
                kernel_size=3,
                padding=1,
                padding_mode=padding_mode,
            ),
        )
        self.residual = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=1, padding=0),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )

    def forward(self, x: torch.Tensor):
        for conv in self.convs:
            x = conv(x)
        x = self.up(x) + self.residual(x)
        x = rearrange(x, "b c h w -> b (h w) c")
        return x


class ResidualConvUnit(nn.Module):
    def __init__(
        self,
        dim,
        kernel_size: int = 3,
        padding_mode: str = "zeros",
        dilation: int = 1,
        layer_scale: float = 1.0,
        use_norm: bool = False,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=dilation * (kernel_size - 1) // 2,
            dilation=dilation,
            padding_mode=padding_mode,
        )
        self.conv2 = nn.Conv2d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=dilation * (kernel_size - 1) // 2,
            dilation=dilation,
            padding_mode=padding_mode,
        )
        self.activation = nn.LeakyReLU()
        self.gamma = (
            nn.Parameter(layer_scale * torch.ones(1, dim, 1, 1))
            if layer_scale > 0.0
            else 1.0
        )
        self.norm1 = nn.GroupNorm(dim // 16, dim) if use_norm else nn.Identity()
        self.norm2 = nn.GroupNorm(dim // 16, dim) if use_norm else nn.Identity()

    def forward(self, x):
        out = self.activation(x)
        out = self.conv1(out)
        out = self.norm1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.norm2(out)
        return self.gamma * out + x


class ResUpsampleBil(nn.Module):
    def __init__(
        self,
        hidden_dim,
        output_dim: int = None,
        num_layers: int = 2,
        kernel_size: int = 3,
        layer_scale: float = 1.0,
        padding_mode: str = "zeros",
        use_norm: bool = False,
        **kwargs,
    ):
        super().__init__()
        output_dim = output_dim if output_dim is not None else hidden_dim // 2
        self.convs = nn.ModuleList([])
        for _ in range(num_layers):
            self.convs.append(
                ResidualConvUnit(
                    hidden_dim,
                    kernel_size=kernel_size,
                    layer_scale=layer_scale,
                    padding_mode=padding_mode,
                    use_norm=use_norm,
                )
            )
        self.up = nn.Sequential(
            nn.Conv2d(
                hidden_dim,
                output_dim,
                kernel_size=1,
                padding=0,
                padding_mode=padding_mode,
            ),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        )

    def forward(self, x: torch.Tensor):
        for conv in self.convs:
            x = conv(x)
        x = self.up(x)
        return x
