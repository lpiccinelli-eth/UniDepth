import torch
import torch.nn as nn


class CvnxtBlock(nn.Module):
    def __init__(
        self,
        dim,
        kernel_size=7,
        layer_scale=1.0,
        expansion=4,
        dilation=1,
        padding_mode: str = "zeros",
    ):
        super().__init__()
        self.dwconv = nn.Conv2d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=dilation * (kernel_size - 1) // 2,
            groups=dim,
            dilation=dilation,
            padding_mode=padding_mode,
        )  # depthwise conv
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, expansion * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(expansion * dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale * torch.ones((dim))) if layer_scale > 0.0 else 1.0
        )

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

        x = self.gamma * x
        x = input + x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        return x
