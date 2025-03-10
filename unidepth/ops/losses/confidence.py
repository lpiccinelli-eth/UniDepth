import torch
import torch.nn as nn

from .utils import FNS, masked_mean


class Confidence(nn.Module):
    def __init__(
        self,
        weight: float,
        output_fn: str = "sqrt",
        input_fn: str = "linear",
        rescale: bool = True,
        eps: float = 1e-5,
    ):
        super(Confidence, self).__init__()
        self.name: str = self.__class__.__name__
        self.weight = weight
        self.rescale = rescale
        self.eps = eps
        self.output_fn = FNS[output_fn]
        self.input_fn = FNS[input_fn]

    @torch.autocast(device_type="cuda", enabled=False, dtype=torch.float32)
    def forward(
        self,
        input: torch.Tensor,
        target_pred: torch.Tensor,
        target_gt: torch.Tensor,
        mask: torch.Tensor,
    ):
        B, C = target_gt.shape[:2]
        mask = mask.bool()
        target_gt = target_gt.float().reshape(B, C, -1)
        target_pred = target_pred.float().reshape(B, C, -1)
        input = input.float().reshape(B, -1)
        mask = mask.reshape(B, -1)

        if self.rescale:
            target_pred = torch.stack(
                [
                    p * torch.median(gt[:, m]) / torch.median(p[:, m])
                    for p, gt, m in zip(target_pred, target_gt, mask)
                ]
            )

        error = torch.abs(
            (self.input_fn(target_pred) - self.input_fn(target_gt)).norm(dim=1) - input
        )
        losses = masked_mean(error, dim=[-1], mask=mask).squeeze(dim=-1)
        losses = self.output_fn(losses)

        return losses

    @classmethod
    def build(cls, config):
        obj = cls(
            weight=config["weight"],
            output_fn=config["output_fn"],
            input_fn=config["input_fn"],
            rescale=config.get("rescale", True),
        )
        return obj
