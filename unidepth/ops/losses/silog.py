import torch
import torch.nn as nn

from .utils import (FNS, REGRESSION_DICT, masked_mean, masked_mean_var,
                    masked_quantile)


class SILog(nn.Module):
    def __init__(
        self,
        weight: float,
        input_fn: str = "linear",
        output_fn: str = "sqrt",
        integrated: float = 0.15,
        dims: list[int] = [-3, -2, -1],
        eps: float = 1e-5,
    ):
        super().__init__()
        self.name: str = self.__class__.__name__
        self.weight: float = weight

        self.dims = dims
        self.input_fn = FNS[input_fn]
        self.output_fn = FNS[output_fn]
        self.eps: float = eps
        self.integrated = integrated

    @torch.autocast(device_type="cuda", enabled=False, dtype=torch.float32)
    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        si: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        mask = mask.bool()
        error = self.input_fn(input.float()) - self.input_fn(target.float())
        mean_error, var_error = masked_mean_var(
            data=error, mask=mask, dim=self.dims, keepdim=False
        )
        if var_error.ndim > 1:
            var_error = var_error.mean(dim=-1)

        if self.integrated > 0.0:
            scale_error = mean_error**2
            var_error = var_error + self.integrated * scale_error * (1 - si.int())

        out_loss = self.output_fn(var_error)
        return out_loss

    @classmethod
    def build(cls, config):
        obj = cls(
            weight=config["weight"],
            dims=config["dims"],
            output_fn=config["output_fn"],
            input_fn=config["input_fn"],
            integrated=config.get("integrated", 0.15),
        )
        return obj
