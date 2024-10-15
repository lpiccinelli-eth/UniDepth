import torch
import torch.nn as nn

from .utils import FNS, REGRESSION_DICT, masked_mean, masked_quantile


class Regression(nn.Module):
    def __init__(
        self,
        weight: float,
        input_fn: str,
        output_fn: str,
        alpha: float,
        gamma: float,
        fn: str,
        dims: list[int] = [-1],
        quantile: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.name = self.__class__.__name__
        self.output_fn = FNS[output_fn]
        self.input_fn = FNS[input_fn]
        self.weight = weight
        self.dims = dims
        self.quantile = quantile
        self.alpha = alpha
        self.gamma = gamma
        self.fn = REGRESSION_DICT[fn]

    @torch.autocast(device_type="cuda", enabled=False, dtype=torch.float32)
    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if mask is not None:  # usually it is just repeated
            mask = mask[:, 0]

        input = self.input_fn(input.float())
        target = self.input_fn(target.float())
        error = self.fn(input - target, gamma=self.gamma, alpha=self.alpha).mean(dim=1)
        mean_error = masked_mean(data=error, mask=mask, dim=self.dims).squeeze(
            self.dims
        )
        mean_error = self.output_fn(mean_error)
        return mean_error

    @classmethod
    def build(cls, config):
        obj = cls(
            weight=config["weight"],
            output_fn=config["output_fn"],
            input_fn=config["input_fn"],
            dims=config.get("dims", (-1,)),
            alpha=config["alpha"],
            gamma=config["gamma"],
            fn=config["fn"],
        )
        return obj
