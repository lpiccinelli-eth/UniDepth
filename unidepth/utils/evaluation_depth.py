"""
Author: Luigi Piccinelli
Licensed under the CC-BY NC 4.0 license (http://creativecommons.org/licenses/by-nc/4.0/)
"""

# We prefer not to install PyTorch3D in the package
# Code commented is how 3D metrics are computed

from collections import defaultdict
from functools import partial

import torch
import torch.nn.functional as F

# from chamfer_distance import ChamferDistance

from unidepth.utils.constants import DEPTH_BINS


# chamfer_cls = ChamferDistance()


# def chamfer_dist(tensor1, tensor2):
#     x_lengths = torch.tensor((tensor1.shape[1],), device=tensor1.device)
#     y_lengths = torch.tensor((tensor2.shape[1],), device=tensor2.device)
#     dist1, dist2, idx1, idx2 = chamfer_cls(
#         tensor1, tensor2, x_lengths=x_lengths, y_lengths=y_lengths
#     )
#     return (torch.sqrt(dist1) + torch.sqrt(dist2)) / 2


# def auc(tensor1, tensor2, thresholds):
#     x_lengths = torch.tensor((tensor1.shape[1],), device=tensor1.device)
#     y_lengths = torch.tensor((tensor2.shape[1],), device=tensor2.device)
#     dist1, dist2, idx1, idx2 = chamfer_cls(
#         tensor1, tensor2, x_lengths=x_lengths, y_lengths=y_lengths
#     )
#     # compute precision recall
#     precisions = [(dist1 < threshold).sum() / dist1.numel() for threshold in thresholds]
#     recalls = [(dist2 < threshold).sum() / dist2.numel() for threshold in thresholds]
#     auc_value = torch.trapz(
#         torch.tensor(precisions, device=tensor1.device),
#         torch.tensor(recalls, device=tensor1.device),
#     )
#     return auc_value


def delta(tensor1, tensor2, exponent):
    inlier = torch.maximum((tensor1 / tensor2), (tensor2 / tensor1))
    return (inlier < 1.25**exponent).to(torch.float32).mean()


def ssi(tensor1, tensor2, qtl=0.05):
    stability_mat = 1e-9 * torch.eye(2, device=tensor1.device)
    error = (tensor1 - tensor2).abs()
    mask = error < torch.quantile(error, 1 - qtl)
    tensor1_mask = tensor1[mask]
    tensor2_mask = tensor2[mask]
    tensor2_one = torch.stack(
        [tensor2_mask.detach(), torch.ones_like(tensor2_mask).detach()], dim=1
    )
    scale_shift = torch.inverse(tensor2_one.T @ tensor2_one + stability_mat) @ (
        tensor2_one.T @ tensor1_mask.unsqueeze(1)
    )
    scale, shift = scale_shift.squeeze().chunk(2, dim=0)
    return tensor2 * scale + shift
    # tensor2_one = torch.stack([tensor2.detach(), torch.ones_like(tensor2).detach()], dim=1)
    # scale_shift = torch.inverse(tensor2_one.T @ tensor2_one + stability_mat) @ (tensor2_one.T @ tensor1.unsqueeze(1))
    # scale, shift = scale_shift.squeeze().chunk(2, dim=0)
    # return tensor2 * scale + shift


def d1_ssi(tensor1, tensor2):
    delta_ = delta(tensor1, ssi(tensor1, tensor2), 1.0)
    return delta_


def d_auc(tensor1, tensor2):
    exponents = torch.linspace(0.01, 5.0, steps=100, device=tensor1.device)
    deltas = [delta(tensor1, tensor2, exponent) for exponent in exponents]
    return torch.trapz(torch.tensor(deltas, device=tensor1.device), exponents) / 5.0


# def f1_score(tensor1, tensor2, thresholds):
#     x_lengths = torch.tensor((tensor1.shape[1],), device=tensor1.device)
#     y_lengths = torch.tensor((tensor2.shape[1],), device=tensor2.device)
#     dist1, dist2, idx1, idx2 = chamfer_cls(
#         tensor1, tensor2, x_lengths=x_lengths, y_lengths=y_lengths
#     )
#     # compute precision recall
#     precisions = [(dist1 < threshold).sum() / dist1.numel() for threshold in thresholds]
#     recalls = [(dist2 < threshold).sum() / dist2.numel() for threshold in thresholds]
#     precisions = torch.tensor(precisions, device=tensor1.device)
#     recalls = torch.tensor(recalls, device=tensor1.device)
#     f1_thresholds = 2 * precisions * recalls / (precisions + recalls)
#     f1_thresholds = torch.where(
#         torch.isnan(f1_thresholds), torch.zeros_like(f1_thresholds), f1_thresholds
#     )
#     f1_value = torch.trapz(f1_thresholds) / len(thresholds)
#     return f1_value


DICT_METRICS = {
    "d1": partial(delta, exponent=1.0),
    "d2": partial(delta, exponent=2.0),
    "d3": partial(delta, exponent=3.0),
    "rmse": lambda gt, pred: torch.sqrt(((gt - pred) ** 2).mean()),
    "rmselog": lambda gt, pred: torch.sqrt(
        ((torch.log(gt) - torch.log(pred)) ** 2).mean()
    ),
    "arel": lambda gt, pred: (torch.abs(gt - pred) / gt).mean(),
    "sqrel": lambda gt, pred: (((gt - pred) ** 2) / gt).mean(),
    "log10": lambda gt, pred: torch.abs(torch.log10(pred) - torch.log10(gt)).mean(),
    "silog": lambda gt, pred: 100 * torch.std(torch.log(pred) - torch.log(gt)).mean(),
    "medianlog": lambda gt, pred: 100
    * (torch.log(pred) - torch.log(gt)).median().abs(),
    "d_auc": d_auc,
    "d1_ssi": d1_ssi,
}


# DICT_METRICS_3D = {
#     "chamfer": lambda gt, pred, thresholds: chamfer_dist(
#         gt.unsqueeze(0).permute(0, 2, 1), pred.unsqueeze(0).permute(0, 2, 1)
#     ),
#     "F1": lambda gt, pred, thresholds: f1_score(
#         gt.unsqueeze(0).permute(0, 2, 1),
#         pred.unsqueeze(0).permute(0, 2, 1),
#         thresholds=thresholds,
#     ),
# }


DICT_METRICS_D = {
    "a1": lambda gt, pred: (torch.maximum((gt / pred), (pred / gt)) > 1.25**1.0).to(
        torch.float32
    ),
    "abs_rel": lambda gt, pred: (torch.abs(gt - pred) / gt),
}


def eval_depth(
    gts: torch.Tensor, preds: torch.Tensor, masks: torch.Tensor, max_depth=None
):
    summary_metrics = defaultdict(list)
    preds = F.interpolate(preds, gts.shape[-2:], mode="bilinear")
    for i, (gt, pred, mask) in enumerate(zip(gts, preds, masks)):
        if max_depth is not None:
            mask = torch.logical_and(mask, gt <= max_depth)
        for name, fn in DICT_METRICS.items():
            summary_metrics[name].append(fn(gt[mask], pred[mask]).mean())
    return {name: torch.stack(vals, dim=0) for name, vals in summary_metrics.items()}


# def eval_3d(
#     gts: torch.Tensor, preds: torch.Tensor, masks: torch.Tensor, thresholds=None
# ):
#     summary_metrics = defaultdict(list)
#     w_max = min(gts.shape[-1] // 4, 400)
#     gts = F.interpolate(
#         gts, (int(w_max * gts.shape[-2] / gts.shape[-1]), w_max), mode="nearest"
#     )
#     preds = F.interpolate(preds, gts.shape[-2:], mode="nearest")
#     masks = F.interpolate(
#         masks.to(torch.float32), gts.shape[-2:], mode="nearest"
#     ).bool()
#     for i, (gt, pred, mask) in enumerate(zip(gts, preds, masks)):
#         if not torch.any(mask):
#             continue
#         for name, fn in DICT_METRICS_3D.items():
#             summary_metrics[name].append(
#                 fn(gt[:, mask.squeeze()], pred[:, mask.squeeze()], thresholds).mean()
#             )
#     return {name: torch.stack(vals, dim=0) for name, vals in summary_metrics.items()}
