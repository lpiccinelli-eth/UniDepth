from math import prod
from typing import Any, Dict, List, Optional, Tuple

import torch

FNS = {
    "sqrt": lambda x: torch.sqrt(x + 1e-4),
    "log": lambda x: torch.log(x + 1e-4),
    "log1": lambda x: torch.log(x + 1),
    # if x -> 0 : log(1/x)
    # if x -> inf : log(1+1/x) -> 1/x + hot
    "log1i": lambda x: torch.log(1 + 50 / (1e-4 + x)),
    "linear": lambda x: x,
    "square": torch.square,
    "disp": lambda x: 1 / (x + 1e-4),
    "disp1": lambda x: 1 / (1 + x),
}


FNS_INV = {
    "sqrt": torch.square,
    "log": torch.exp,
    "log1": lambda x: torch.exp(x) - 1,
    "linear": lambda x: x,
    "square": torch.sqrt,
    "disp": lambda x: 1 / x,
}


def masked_mean_var(
    data: torch.Tensor, mask: torch.Tensor, dim: List[int], keepdim: bool = True
):
    if mask is None:
        return data.mean(dim=dim, keepdim=keepdim), data.var(dim=dim, keepdim=keepdim)
    mask = mask.float()
    mask_sum = torch.sum(mask, dim=dim, keepdim=True)
    # data = torch.nan_to_num(data, nan=0.0)
    mask_mean = torch.sum(data * mask, dim=dim, keepdim=True) / torch.clamp(
        mask_sum, min=1.0
    )
    mask_var = torch.sum(
        mask * (data - mask_mean) ** 2, dim=dim, keepdim=True
    ) / torch.clamp(mask_sum, min=1.0)
    if not keepdim:
        mask_mean, mask_var = mask_mean.squeeze(dim), mask_var.squeeze(dim)
    return mask_mean, mask_var


def masked_mean(data: torch.Tensor, mask: torch.Tensor | None, dim: List[int]):
    if mask is None:
        return data.mean(dim=dim, keepdim=True)
    mask = mask.float()
    mask_sum = torch.sum(mask, dim=dim, keepdim=True)
    mask_mean = torch.sum(
        torch.nan_to_num(data, nan=0.0) * mask, dim=dim, keepdim=True
    ) / mask_sum.clamp(min=1.0)
    return mask_mean


def masked_quantile(
    data: torch.Tensor, mask: torch.Tensor | None, dims: List[int], q: float
):
    """
    Compute the quantile of the data only where the mask is 1 along specified dimensions.

    Args:
        data (torch.Tensor): The input data tensor.
        mask (torch.Tensor): The mask tensor with the same shape as data, containing 1s where data should be considered.
        dims (list of int): The dimensions to compute the quantile over.
        q (float): The quantile to compute, must be between 0 and 1.

    Returns:
        torch.Tensor: The quantile computed over the specified dimensions, ignoring masked values.
    """
    masked_data = data * mask if mask is not None else data

    # Get a list of all dimensions
    all_dims = list(range(masked_data.dim()))

    # Revert negative dimensions
    dims = [d % masked_data.dim() for d in dims]

    # Find the dimensions to keep (not included in the `dims` list)
    keep_dims = [d for d in all_dims if d not in dims]

    # Permute dimensions to bring `dims` to the front
    permute_order = dims + keep_dims
    permuted_data = masked_data.permute(permute_order)

    # Reshape into 2D: (-1, remaining_dims)
    collapsed_shape = (
        -1,
        prod([permuted_data.size(d) for d in range(len(dims), permuted_data.dim())]),
    )
    reshaped_data = permuted_data.reshape(collapsed_shape)
    if mask is None:
        return torch.quantile(reshaped_data, q, dim=0)

    permuted_mask = mask.permute(permute_order)
    reshaped_mask = permuted_mask.reshape(collapsed_shape)

    # Calculate quantile along the first dimension where mask is true
    quantiles = []
    for i in range(reshaped_data.shape[1]):
        valid_data = reshaped_data[:, i][reshaped_mask[:, i]]
        if valid_data.numel() == 0:
            # print("Warning: No valid data found for quantile calculation.")
            quantiles.append(reshaped_data[:, i].min() * 0.99)
        else:
            quantiles.append(torch.quantile(valid_data, q, dim=0))

    # Stack back into a tensor with reduced dimensions
    quantiles = torch.stack(quantiles)
    quantiles = quantiles.reshape(
        [permuted_data.size(d) for d in range(len(dims), permuted_data.dim())]
    )

    return quantiles


def masked_median(data: torch.Tensor, mask: torch.Tensor, dim: List[int]):
    ndim = data.ndim
    data = data.flatten(ndim - len(dim))
    mask = mask.flatten(ndim - len(dim))
    mask_median = torch.median(data[..., mask], dim=-1).values
    return mask_median


def masked_median_mad(data: torch.Tensor, mask: torch.Tensor, dim: List[int]):
    ndim = data.ndim
    data = data.flatten(ndim - len(dim))
    mask = mask.flatten(ndim - len(dim))
    mask_median = torch.median(data[mask], dim=-1, keepdim=True).values
    mask_mad = masked_mean((data - mask_median).abs(), mask, dim=[-1])
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


def ssi(
    input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, dim: list[int]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # recalculate mask with points in 95% confidence interval
    # the statistics are calculated on the stable points and
    # are similar ot median/MAD, but median/MAD gradients
    # are really weird, so this is a workaround
    input_detach = input.detach()
    input_mean, input_var = masked_mean_var(input_detach, mask=mask, dim=dim)
    target_mean, target_var = masked_mean_var(target, mask=mask, dim=dim)
    input_std = (input_var).clip(min=1e-6).sqrt()
    target_std = (target_var).clip(min=1e-6).sqrt()
    stable_points_input = torch.logical_and(
        input_detach > input_mean - 1.96 * input_std,
        input_detach < input_mean + 1.96 * input_std,
    )
    stable_points_target = torch.logical_and(
        target > target_mean - 1.96 * target_std,
        target < target_mean + 1.96 * target_std,
    )
    stable_mask = stable_points_target & stable_points_input & mask

    input_mean, input_var = masked_mean_var(input, mask=stable_mask, dim=dim)
    target_mean, target_var = masked_mean_var(target, mask=stable_mask, dim=dim)
    target_normalized = (target - target_mean) / FNS["sqrt"](target_var)
    input_normalized = (input - input_mean) / FNS["sqrt"](input_var)
    return input_normalized, target_normalized, stable_mask


def ind2sub(idx, cols):
    r = idx // cols
    c = idx % cols
    return r, c


def sub2ind(r, c, cols):
    idx = r * cols + c
    return idx


def l2(input_tensor: torch.Tensor, gamma: float = 1.0, *args, **kwargs) -> torch.Tensor:
    return gamma * (input_tensor / gamma) ** 2


def l1(input_tensor: torch.Tensor, gamma: float = 1.0, *args, **kwargs) -> torch.Tensor:
    return torch.abs(input_tensor)


def charbonnier(
    input_tensor: torch.Tensor, gamma: float = 1.0, *args, **kwargs
) -> torch.Tensor:
    return torch.sqrt(torch.square(input_tensor) + gamma**2) - gamma


def cauchy(
    input_tensor: torch.Tensor, gamma: float = 1.0, *args, **kwargs
) -> torch.Tensor:
    return gamma * torch.log(torch.square(input_tensor) / gamma + 1)


def geman_mcclure(
    input_tensor: torch.Tensor, gamma: float = 1.0, *args, **kwargs
) -> torch.Tensor:
    return gamma * torch.square(input_tensor) / (torch.square(input_tensor) + gamma)


def robust_loss(
    input_tensor: torch.Tensor, alpha: float, gamma: float = 1.0, *args, **kwargs
) -> torch.Tensor:
    coeff = abs(alpha - 2) / alpha
    power = torch.square(input_tensor) / abs(alpha - 2) / (gamma**2) + 1
    return (
        gamma * coeff * (torch.pow(power, alpha / 2) - 1)
    )  # mult gamma to keep grad magnitude invariant wrt gamma


REGRESSION_DICT = {
    "l2": l2,
    "l1": l1,
    "cauchy": cauchy,
    "charbonnier": charbonnier,
    "geman_mcclure": geman_mcclure,
    "robust_loss": robust_loss,
}
