from __future__ import absolute_import, division, print_function

import torch
import torch.nn.functional as F
from torch import nn

from ..functions import ExtractPatchesFunction


class RandomPatchExtractor(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

    def forward(
        self, tensor: torch.Tensor, centers: torch.Tensor, patch_size: tuple[int, int]
    ):
        device = tensor.device
        dtype = tensor.dtype
        patch_width, patch_height = patch_size
        pad_width = patch_width // 2
        pad_height = patch_height // 2
        dtype = tensor.dtype

        # Pad input to avoid out-of-bounds
        tensor_padded = F.pad(
            tensor,
            (pad_width, pad_width, pad_height, pad_height),
            mode="constant",
            value=0.0,
        )

        # Adjust edge coordinates to account for padding
        centers_padded = centers + torch.tensor(
            [pad_height, pad_width], dtype=dtype, device=device
        ).reshape(1, 1, 2)

        output = ExtractPatchesFunction.apply(
            tensor_padded.float(), centers_padded.int(), patch_height, patch_width
        )
        return output.to(dtype)
