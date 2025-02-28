"""
Author: Luigi Piccinelli
Licensed under the CC-BY NC 4.0 license (http://creativecommons.org/licenses/by-nc/4.0/)
"""

import torch
import torch.utils.data.distributed
import wandb
from torch.nn import functional as F

from unidepth.utils import barrier, is_main_process
from unidepth.utils.misc import remove_padding


def original_image(batch, preds=None):
    paddings = [
        torch.tensor(pads)
        for img_meta in batch["img_metas"]
        for pads in img_meta.get("paddings", [[0] * 4])
    ]
    paddings = torch.stack(paddings).to(batch["data"]["image"].device)[
        ..., [0, 2, 1, 3]
    ]  # lrtb

    T, _, H, W = batch["data"]["depth"].shape
    batch["data"]["image"] = F.interpolate(
        batch["data"]["image"],
        (H + paddings[2] + paddings[3], W + paddings[1] + paddings[2]),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    batch["data"]["image"] = remove_padding(
        batch["data"]["image"], paddings.repeat(T, 1)
    )

    if preds is not None:
        for key in ["depth"]:
            if key in preds:
                preds[key] = F.interpolate(
                    preds[key],
                    (H + paddings[2] + paddings[3], W + paddings[1] + paddings[2]),
                    mode="bilinear",
                    align_corners=False,
                    antialias=True,
                )
                preds[key] = remove_padding(preds[key], paddings.repeat(T, 1))

    return batch, preds


def log_metrics(metrics_all, step):
    for name_ds, metrics in metrics_all.items():
        for metrics_name, metrics_value in metrics.items():
            try:
                print(f"Metrics/{name_ds}/{metrics_name} {round(metrics_value, 4)}")
                wandb.log(
                    {f"Metrics/{name_ds}/{metrics_name}": metrics_value}, step=step
                )
            except:
                pass


def validate(model, test_loaders, step, context):
    metrics_all = {}
    for name_ds, test_loader in test_loaders.items():
        for i, batch in enumerate(test_loader):
            with context:
                batch["data"] = {
                    k: v.to(model.device) for k, v in batch["data"].items()
                }
                # remove temporal dimension of the dataloder, here is always 1!
                batch["data"] = {k: v.squeeze(1) for k, v in batch["data"].items()}
                batch["img_metas"] = [
                    {k: v[0] for k, v in meta.items() if isinstance(v, list)}
                    for meta in batch["img_metas"]
                ]

                preds = model(batch["data"], batch["img_metas"])

            batch, _ = original_image(batch, preds=None)
            test_loader.dataset.accumulate_metrics(
                inputs=batch["data"],
                preds=preds,
                keyframe_idx=batch["img_metas"][0].get("keyframe_idx"),
            )

        barrier()
        metrics_all[name_ds] = test_loader.dataset.get_evaluation()

    barrier()
    if is_main_process():
        log_metrics(metrics_all=metrics_all, step=step)
    return metrics_all
