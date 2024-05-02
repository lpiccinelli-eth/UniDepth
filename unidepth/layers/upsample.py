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
