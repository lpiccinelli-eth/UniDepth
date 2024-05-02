"""
Author: Luigi Piccinelli
Licensed under the CC-BY NC 4.0 license (http://creativecommons.org/licenses/by-nc/4.0/)
"""

from typing import Any, Optional, Dict, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


FNS = {
    "sqrt": torch.sqrt,
    "log": torch.log,
    "log1": lambda x: torch.log(x + 1),
    "linear": lambda x: x,
    "square": torch.square,
    "disp": lambda x: 1 / x,
}


FNS_INV = {
    "sqrt": torch.square,
    "log": torch.exp,
    "log1": lambda x: torch.exp(x) - 1,
    "linear": lambda x: x,
    "square": torch.sqrt,
    "disp": lambda x: 1 / x,
}


def masked_mean_var(data: torch.Tensor, mask: torch.Tensor, dim: List[int]):
    if mask is None:
        return data.mean(dim=dim, keepdim=True), data.var(dim=dim, keepdim=True)
    mask = mask.float()
    mask_sum = torch.sum(mask, dim=dim, keepdim=True)
    mask_mean = torch.sum(data * mask, dim=dim, keepdim=True) / torch.clamp(
        mask_sum, min=1.0
    )
    mask_var = torch.sum(
        mask * (data - mask_mean) ** 2, dim=dim, keepdim=True
    ) / torch.clamp(mask_sum, min=1.0)
    return mask_mean.squeeze(dim), mask_var.squeeze(dim)


def masked_mean(data: torch.Tensor, mask: torch.Tensor | None, dim: List[int]):
    if mask is None:
        return data.mean(dim=dim, keepdim=True)
    mask = mask.float()
    mask_sum = torch.sum(mask, dim=dim, keepdim=True)
    mask_mean = torch.sum(data * mask, dim=dim, keepdim=True) / torch.clamp(
        mask_sum, min=1.0
    )
    return mask_mean


def masked_mae(data: torch.Tensor, mask: torch.Tensor, dim: Tuple[int, ...]):
    if mask is None:
        return data.abs().mean(dim=dim, keepdim=True)
    mask = mask.float()
    mask_sum = torch.sum(mask, dim=dim, keepdim=True)
    mask_mean = torch.sum(data.abs() * mask, dim=dim, keepdim=True) / torch.clamp(
        mask_sum, min=1.0
    )
    return mask_mean


def masked_mse(data: torch.Tensor, mask: torch.Tensor, dim: Tuple[int, ...]):
    if mask is None:
        return (data**2).mean(dim=dim, keepdim=True)
    mask = mask.float()
    mask_sum = torch.sum(mask, dim=dim, keepdim=True)
    mask_mean = torch.sum((data**2) * mask, dim=dim, keepdim=True) / torch.clamp(
        mask_sum, min=1.0
    )
    return mask_mean


def masked_median(data: torch.Tensor, mask: torch.Tensor, dim: List[int]):
    ndim = data.ndim
    data = data.flatten(ndim - len(dim))
    mask = mask.flatten(ndim - len(dim))
    mask_median = torch.median(data[mask], dim=-1).values
    return mask_median


def masked_median_mad(data: torch.Tensor, mask: torch.Tensor):
    data = data.flatten()
    mask = mask.flatten()
    mask_median = torch.median(data[mask])
    n_samples = torch.clamp(torch.sum(mask.float()), min=1.0)
    mask_mad = torch.sum((data[mask] - mask_median).abs()) / n_samples
    return mask_median, mask_mad


def masked_weighted_mean_var(
    data: torch.Tensor, mask: torch.Tensor, weights: torch.Tensor, dim: Tuple[int, ...]
):
    if mask is None:
        return data.mean(dim=dim, keepdim=True), data.var(dim=dim, keepdim=True)
    mask = mask.float()
    mask_mean = torch.sum(data * mask * weights, dim=dim, keepdim=True) / torch.sum(
        mask * weights, dim=dim, keepdim=True
    ).clamp(min=1.0)
    # V1**2 - V2, V1: sum w_i, V2: sum w_i**2
    denom = torch.sum(weights * mask, dim=dim, keepdim=True).square() - torch.sum(
        (mask * weights).square(), dim=dim, keepdim=True
    )
    # correction is V1 / (V1**2 - V2), if w_i=1 => N/(N**2 - N) => 1/(N-1) (unbiased estimator of variance, cvd)
    correction_factor = torch.sum(mask * weights, dim=dim, keepdim=True) / denom.clamp(
        min=1.0
    )
    mask_var = correction_factor * torch.sum(
        weights * mask * (data - mask_mean) ** 2, dim=dim, keepdim=True
    )
    return mask_mean, mask_var


def masked_mean_var_q(data: torch.Tensor, mask: torch.Tensor, dim: List[int]):
    if mask is None:
        return data.mean(dim=dim, keepdim=True), data.var(dim=dim, keepdim=True)
    mask = mask.float()
    mask_sum = torch.sum(mask, dim=dim, keepdim=True)
    mask_mean = torch.sum(data * mask, dim=dim, keepdim=True) / torch.clamp(
        mask_sum, min=1.0
    )
    mask_var = torch.sum(
        mask * (data - mask_mean) ** 2, dim=dim, keepdim=True
    ) / torch.clamp(mask_sum, min=1.0)
    return mask_mean, mask_var


class SILog(nn.Module):
    def __init__(
        self,
        weight: float,
        scale_pred_weight: float = 0.15,
        output_fn: str = "sqrt",
        input_fn: str = "log",
        legacy: bool = False,
        abs_rel: bool = False,
        norm: bool = False,
        eps: float = 1e-5,
    ):
        super().__init__()
        assert output_fn in FNS
        self.name: str = self.__class__.__name__
        self.weight: float = weight

        self.scale_pred_weight: float = scale_pred_weight
        self.dims = (-4, -3, -2, -1) if legacy else (-2, -1)
        self.output_fn = FNS[output_fn]
        self.input_fn = FNS[input_fn]
        self.abs_rel = abs_rel
        self.norm = norm
        self.eps: float = eps

    @torch.cuda.amp.autocast(enabled=False)
    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        interpolate: bool = True,
        scale_inv: torch.Tensor | None = None,
        ss_inv: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if interpolate:
            input = F.interpolate(
                input, target.shape[-2:], mode="bilinear", align_corners=False
            )
        if mask is not None:
            mask = mask.to(torch.bool)
        if ss_inv is not None:
            ss_inv = ~ss_inv

        if input.shape[1] > 1:
            input_ = torch.cat(
                [input[:, :-1], self.input_fn(input[:, -1:].clamp(min=self.eps))], dim=1
            )
            target_ = torch.cat(
                [target[:, :-1], self.input_fn(target[:, -1:].clamp(min=self.eps))],
                dim=1,
            )
            error = torch.norm(input_ - target_, dim=1, keepdim=True)
        else:
            input_ = self.input_fn(input.clamp(min=self.eps))
            target_ = self.input_fn(target.clamp(min=self.eps))
            error = input_ - target_

        mean_error, var_error = masked_mean_var(data=error, mask=mask, dim=self.dims)

        # prevoiusly was inverted!!
        if self.abs_rel:
            scale_error = (input - target).abs()[:, -1:] / target[:, -1:].clip(
                min=self.eps
            )
            scale_error = masked_mean(data=scale_error, mask=mask, dim=self.dims)
        else:
            scale_error = mean_error**2

        if var_error.ndim > 1:
            var_error = var_error.sum(dim=1)
            scale_error = scale_error.sum(dim=1)

        # if scale inv -> mask scale error, if scale/shift, mask the full loss
        if scale_inv is not None:
            scale_error = (1 - scale_inv.int()) * scale_error
        scale_error = self.scale_pred_weight * scale_error
        loss = var_error + scale_error
        out_loss = self.output_fn(loss.clamp(min=self.eps))
        out_loss = masked_mean(data=out_loss, mask=ss_inv, dim=(0,))
        return out_loss.mean()

    @classmethod
    def build(cls, config: Dict[str, Any]):
        obj = cls(
            weight=config["weight"],
            legacy=config["legacy"],
            output_fn=config["output_fn"],
            input_fn=config["input_fn"],
            norm=config.get("norm", False),
            scale_pred_weight=config.get("gamma", 0.15),
            abs_rel=config.get("abs_rel", False),
        )
        return obj


class MSE(nn.Module):
    def __init__(
        self,
        weight: float = 1.0,
        input_fn: str = "linear",
        output_fn: str = "linear",
    ):
        super().__init__()
        self.name: str = self.__class__.__name__
        self.output_fn = FNS[output_fn]
        self.input_fn = FNS[input_fn]
        self.weight: float = weight
        self.eps = 1e-6

    @torch.cuda.amp.autocast(enabled=False)
    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
        batch_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        input = input[..., : target.shape[-1]]  # B N C or B H W C
        error = self.input_fn(input + self.eps) - self.input_fn(target + self.eps)
        abs_error = torch.square(error).sum(dim=-1)
        mean_error = masked_mean(data=abs_error, mask=mask, dim=(-1,)).mean(dim=-1)
        batched_error = masked_mean(
            self.output_fn(mean_error.clamp(self.eps)), batch_mask, dim=(0,)
        )
        return batched_error.mean(), mean_error.detach()

    @classmethod
    def build(cls, config: Dict[str, Any]):
        obj = cls(
            weight=config["weight"],
            output_fn=config["output_fn"],
            input_fn=config["input_fn"],
        )
        return obj


class SelfCons(nn.Module):
    def __init__(
        self,
        weight: float,
        scale_pred_weight: float = 0.15,
        output_fn: str = "sqrt",
        input_fn: str = "log",
        abs_rel: bool = False,
        norm: bool = False,
        eps: float = 1e-5,
    ):
        super().__init__()
        assert output_fn in FNS
        self.name: str = self.__class__.__name__
        self.weight: float = weight

        self.scale_pred_weight: float = scale_pred_weight
        self.dims = (-2, -1)
        self.output_fn = FNS[output_fn]
        self.input_fn = FNS[input_fn]
        self.abs_rel = abs_rel
        self.norm = norm
        self.eps: float = eps

    @torch.cuda.amp.autocast(enabled=False)
    def forward(
        self,
        input: torch.Tensor,
        mask: torch.Tensor,
        metas: List[Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        chunks = input.shape[0] // 2
        device = input.device
        mask = F.interpolate(mask.float(), size=input.shape[-2:], mode="nearest")

        rescales = input.shape[-2] / torch.tensor(
            [x["resized_shape"][0] for x in metas], device=device
        )
        cams = torch.cat([x["K_target"] for x in metas], dim=0).to(device)
        flips = torch.tensor([x["flip"] for x in metas], device=device)

        iters = zip(
            input.chunk(chunks),
            mask.chunk(chunks),
            cams.chunk(chunks),
            rescales.chunk(chunks),
            flips.chunk(chunks),
        )
        inputs0, inputs1, masks = [], [], []
        for i, (pair_input, pair_mask, pair_cam, pair_rescale, pair_flip) in enumerate(
            iters
        ):
            mask0, mask1 = pair_mask
            input0, input1 = pair_input
            cam0, cam1 = pair_cam
            rescale0, rescale1 = pair_rescale
            flip0, flip1 = pair_flip

            fx_0 = cam0[0, 0] * rescale0
            fx_1 = cam1[0, 0] * rescale1
            cx_0 = (cam0[0, 2] - 0.5) * rescale0 + 0.5
            cx_1 = (cam1[0, 2] - 0.5) * rescale1 + 0.5
            cy_0 = (cam0[1, 2] - 0.5) * rescale0 + 0.5
            cy_1 = (cam1[1, 2] - 0.5) * rescale1 + 0.5

            # flip image
            if flip0 ^ flip1:
                input0 = torch.flip(input0, dims=(2,))
                mask0 = torch.flip(mask0, dims=(2,))
                cx_0 = input0.shape[-1] - cx_0

            # calc zoom
            zoom_x = float(fx_1 / fx_0)

            # apply zoom
            input0 = F.interpolate(
                input0.unsqueeze(0),
                scale_factor=zoom_x,
                mode="bilinear",
                align_corners=True,
            ).squeeze(0)
            mask0 = F.interpolate(
                mask0.unsqueeze(0), scale_factor=zoom_x, mode="nearest"
            ).squeeze(0)

            # calc translation
            change_left = int(cx_1 - (cx_0 - 0.5) * zoom_x - 0.5)
            change_top = int(cy_1 - (cy_0 - 0.5) * zoom_x - 0.5)
            change_right = input1.shape[-1] - change_left - input0.shape[-1]
            change_bottom = input1.shape[-2] - change_top - input0.shape[-2]

            # apply translation
            pad_left = max(0, change_left)
            pad_right = max(0, change_right)
            pad_top = max(0, change_top)
            pad_bottom = max(0, change_bottom)

            crop_left = max(0, -change_left)
            crop_right = max(0, -change_right)
            crop_top = max(0, -change_top)
            crop_bottom = max(0, -change_bottom)

            input0 = F.pad(
                input0,
                (pad_left, pad_right, pad_top, pad_bottom),
                mode="constant",
                value=0,
            )
            mask0 = F.pad(
                mask0,
                (pad_left, pad_right, pad_top, pad_bottom),
                mode="constant",
                value=0,
            )
            input0 = input0[
                :,
                crop_top : input0.shape[-2] - crop_bottom,
                crop_left : input0.shape[-1] - crop_right,
            ]
            mask0 = mask0[
                :,
                crop_top : mask0.shape[-2] - crop_bottom,
                crop_left : mask0.shape[-1] - crop_right,
            ]

            mask = torch.logical_and(mask0, mask1)

            inputs0.append(input0)
            inputs1.append(input1)
            masks.append(mask)

        inputs0 = torch.stack(inputs0, dim=0)
        inputs1 = torch.stack(inputs1, dim=0)
        masks = torch.stack(masks, dim=0)
        loss1 = self.loss(inputs0, inputs1.detach(), masks)
        loss2 = self.loss(inputs1, inputs0.detach(), masks)
        return torch.cat([loss1, loss2], dim=0).mean()

    def loss(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        loss = masked_mean(
            (input - target).square().mean(dim=1), mask=mask, dim=(-2, -1)
        )
        return self.output_fn(loss + self.eps)

    @classmethod
    def build(cls, config: Dict[str, Any]):
        obj = cls(
            weight=config["weight"],
            output_fn=config["output_fn"],
            input_fn=config["input_fn"],
        )
        return obj
