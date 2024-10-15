import torch
import torch.nn as nn

from .utils import FNS, masked_mean


class ARel(nn.Module):
    def __init__(
        self,
        weight: float,
        output_fn: str = "sqrt",
        input_fn: str = "linear",
        eps: float = 1e-5,
    ):
        super().__init__()
        self.name: str = self.__class__.__name__
        self.weight: float = weight
        self.dims = [-2, -1]
        self.output_fn = FNS[output_fn]
        self.input_fn = FNS[input_fn]
        self.eps: float = eps

    @torch.autocast(device_type="cuda", enabled=False, dtype=torch.float32)
    def forward(
        self, input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        mask = mask.bool().clone()

        input = self.input_fn(input.float())
        target = self.input_fn(target.float())

        error = (input - target).norm(dim=1) / target.norm(dim=1).clip(min=0.05)
        mask = mask.squeeze(1)

        error_image = masked_mean(data=error, mask=mask, dim=self.dims).squeeze(1, 2)
        error_image = self.output_fn(error_image)
        return error_image

    @classmethod
    def build(cls, config):
        obj = cls(
            weight=config["weight"],
            output_fn=config["output_fn"],
            input_fn=config["input_fn"],
        )
        return obj
