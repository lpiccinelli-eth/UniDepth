import os
from abc import abstractmethod
from copy import deepcopy
from math import ceil, log
from typing import Any, Dict, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

import unidepth.datasets.pipelines as pipelines
from unidepth.utils import (eval_3d, eval_depth, identity, is_main_process,
                            recursive_index, sync_tensor_across_gpus)
from unidepth.utils.constants import (IMAGENET_DATASET_MEAN,
                                      IMAGENET_DATASET_STD,
                                      OPENAI_DATASET_MEAN, OPENAI_DATASET_STD)


class BaseDataset(Dataset):
    min_depth = 0.01
    max_depth = 1000.0

    def __init__(
        self,
        image_shape: Tuple[int, int],
        split_file: str,
        test_mode: bool,
        benchmark: bool,
        normalize: bool,
        augmentations_db: Dict[str, Any],
        resize_method: str,
        mini: float,
        num_copies: int = 1,
        **kwargs,
    ) -> None:
        super().__init__()
        assert normalize in [None, "imagenet", "openai"]

        self.split_file = split_file
        self.test_mode = test_mode
        self.data_root = os.environ["DATAROOT"]
        self.image_shape = image_shape
        self.resize_method = resize_method
        self.mini = mini
        self.num_frames = 1
        self.num_copies = num_copies
        self.metrics_store = {}
        self.metrics_count = {}

        if normalize == "imagenet":
            self.normalization_stats = {
                "mean": torch.tensor(IMAGENET_DATASET_MEAN),
                "std": torch.tensor(IMAGENET_DATASET_STD),
            }
        elif normalize == "openai":
            self.normalization_stats = {
                "mean": torch.tensor(OPENAI_DATASET_MEAN),
                "std": torch.tensor(OPENAI_DATASET_STD),
            }
        else:
            self.normalization_stats = {
                "mean": torch.tensor([0.0, 0.0, 0.0]),
                "std": torch.tensor([1.0, 1.0, 1.0]),
            }

        for k, v in augmentations_db.items():
            setattr(self, k, v)
        if not self.test_mode:
            self._augmentation_space()

        self.masker = pipelines.AnnotationMask(
            min_value=0.0,
            max_value=self.max_depth if test_mode else None,
            custom_fn=identity,
        )
        self.filler = pipelines.RandomFiller(noise_pad=True)

        shape_mult = self.shape_constraints["shape_mult"]
        self.image_shape = [
            ceil(self.image_shape[0] / shape_mult) * shape_mult,
            ceil(self.image_shape[1] / shape_mult) * shape_mult,
        ]
        self.resizer = pipelines.ContextCrop(
            image_shape=self.image_shape,
            train_ctx_range=(1.0 / self.random_scale, 1.0 * self.random_scale),
            test_min_ctx=self.test_context,
            keep_original=test_mode,
            shape_constraints=self.shape_constraints,
        )

        self.collecter = pipelines.Collect(
            keys=["image_fields", "mask_fields", "gt_fields", "camera_fields"]
        )

    def __len__(self):
        return len(self.dataset)

    def pack_batch(self, results):
        results["paddings"] = [
            results[x]["paddings"][0] for x in results["sequence_fields"]
        ]
        for fields_name in [
            "image_fields",
            "gt_fields",
            "mask_fields",
            "camera_fields",
        ]:
            fields = results.get(fields_name)
            packed = {
                field: torch.cat(
                    [results[seq][field] for seq in results["sequence_fields"]]
                )
                for field in fields
            }
            results.update(packed)
        return results

    def unpack_batch(self, results):
        for fields_name in [
            "image_fields",
            "gt_fields",
            "mask_fields",
            "camera_fields",
        ]:
            fields = results.get(fields_name)
            unpacked = {
                field: {
                    seq: results[field][idx : idx + 1]
                    for idx, seq in enumerate(results["sequence_fields"])
                }
                for field in fields
            }
            results.update(unpacked)
        return results

    def _augmentation_space(self):
        self.augmentations_dict = {
            "Flip": pipelines.RandomFlip(prob=self.flip_p),
            "Jitter": pipelines.RandomColorJitter(
                (-self.random_jitter, self.random_jitter), prob=self.jitter_p
            ),
            "Gamma": pipelines.RandomGamma(
                (-self.random_gamma, self.random_gamma), prob=self.gamma_p
            ),
            "Blur": pipelines.GaussianBlur(
                kernel_size=13, sigma=(0.1, self.random_blur), prob=self.blur_p
            ),
            "Grayscale": pipelines.RandomGrayscale(prob=self.grayscale_p),
        }

    def augment(self, results):
        for name, aug in self.augmentations_dict.items():
            results = aug(results)
        return results

    def prepare_depth_eval(self, inputs, preds):
        new_preds = {}
        keyframe_idx = getattr(self, "keyframe_idx", None)
        slice_idx = slice(
            keyframe_idx, keyframe_idx + 1 if keyframe_idx is not None else None
        )
        new_gts = inputs["depth"][slice_idx]
        new_masks = inputs["depth_mask"][slice_idx].bool()
        for key, val in preds.items():
            if "depth" in key:
                new_preds[key] = val[slice_idx]
        return new_gts, new_preds, new_masks

    def prepare_points_eval(self, inputs, preds):
        new_preds = {}
        new_gts = inputs["points"]
        new_masks = inputs["depth_mask"].bool()
        if "points_mask" in inputs:
            new_masks = inputs["points_mask"].bool()
        for key, val in preds.items():
            if "points" in key:
                new_preds[key] = val
        return new_gts, new_preds, new_masks

    def add_points(self, inputs):
        inputs["points"] = inputs.get("camera_original", inputs["camera"]).reconstruct(
            inputs["depth"]
        )
        return inputs

    @torch.autocast(device_type="cuda", enabled=False, dtype=torch.float32)
    def accumulate_metrics(
        self,
        inputs,
        preds,
        keyframe_idx=None,
        metrics=["depth", "points", "flow_fwd", "pairwise"],
    ):
        if "depth" in inputs and "points" not in inputs:
            inputs = self.add_points(inputs)

        available_metrics = []
        for metric in metrics:
            metric_in_gt = any((metric in k for k in inputs.keys()))
            metric_in_pred = any((metric in k for k in preds.keys()))
            if metric_in_gt and metric_in_pred:
                available_metrics.append(metric)

        if keyframe_idx is not None:
            inputs = recursive_index(inputs, slice(keyframe_idx, keyframe_idx + 1))
            preds = recursive_index(preds, slice(keyframe_idx, keyframe_idx + 1))

        if "depth" in available_metrics:
            depth_gt, depth_pred, depth_masks = self.prepare_depth_eval(inputs, preds)
            self.accumulate_metrics_depth(depth_gt, depth_pred, depth_masks)

        if "points" in available_metrics:
            points_gt, points_pred, points_masks = self.prepare_points_eval(
                inputs, preds
            )
            self.accumulate_metrics_3d(points_gt, points_pred, points_masks)

    @torch.autocast(device_type="cuda", enabled=False, dtype=torch.float32)
    def accumulate_metrics_depth(self, gts, preds, masks):
        for eval_type, pred in preds.items():
            log_name = eval_type.replace("depth", "").strip("-").strip("_")
            if log_name not in self.metrics_store:
                self.metrics_store[log_name] = {}
            current_count = self.metrics_count.get(
                log_name, torch.tensor([], device=gts.device)
            )
            new_count = masks.view(gts.shape[0], -1).sum(dim=-1)
            self.metrics_count[log_name] = torch.cat([current_count, new_count])
            for k, v in eval_depth(gts, pred, masks, max_depth=self.max_depth).items():
                current_metric = self.metrics_store[log_name].get(
                    k, torch.tensor([], device=gts.device)
                )
                self.metrics_store[log_name][k] = torch.cat([current_metric, v])

    @torch.autocast(device_type="cuda", enabled=False, dtype=torch.float32)
    def accumulate_metrics_3d(self, gts, preds, masks):
        thresholds = torch.linspace(
            log(self.min_depth),
            log(self.max_depth / 20),
            steps=100,
            device=gts.device,
        ).exp()
        for eval_type, pred in preds.items():
            log_name = eval_type.replace("points", "").strip("-").strip("_")
            if log_name not in self.metrics_store:
                self.metrics_store[log_name] = {}
            current_count = self.metrics_count.get(
                log_name, torch.tensor([], device=gts.device)
            )
            new_count = masks.view(gts.shape[0], -1).sum(dim=-1)
            self.metrics_count[log_name] = torch.cat([current_count, new_count])
            for k, v in eval_3d(gts, pred, masks, thresholds=thresholds).items():
                current_metric = self.metrics_store[log_name].get(
                    k, torch.tensor([], device=gts.device)
                )
                self.metrics_store[log_name][k] = torch.cat([current_metric, v])

    def get_evaluation(self, metrics=None):
        metric_vals = {}
        for eval_type in metrics if metrics is not None else self.metrics_store.keys():
            assert self.metrics_store[eval_type]
            cnts = sync_tensor_across_gpus(self.metrics_count[eval_type])
            for name, val in self.metrics_store[eval_type].items():
                # vals_r = (sync_tensor_across_gpus(val) * cnts / cnts.sum()).sum()
                vals_r = sync_tensor_across_gpus(val).mean()
                metric_vals[f"{eval_type}_{name}".strip("_")] = np.round(
                    vals_r.cpu().item(), 5
                )
            self.metrics_store[eval_type] = {}
        self.metrics_count = {}
        return metric_vals

    def replicate(self, results):
        for i in range(1, self.num_copies):
            results[(0, i)] = {k: deepcopy(v) for k, v in results[(0, 0)].items()}
            results["sequence_fields"].append((0, i))
        return results

    def log_load_dataset(self):
        if is_main_process():
            info = f"Loaded {self.__class__.__name__} with {len(self)} images."
            print(info)

    def pre_pipeline(self, results):
        results["image_fields"] = results.get("image_fields", set())
        results["gt_fields"] = results.get("gt_fields", set())
        results["mask_fields"] = results.get("mask_fields", set())
        results["sequence_fields"] = results.get("sequence_fields", set())
        results["camera_fields"] = results.get("camera_fields", set())
        results["dataset_name"] = (
            [self.__class__.__name__] * self.num_frames * self.num_copies
        )
        results["depth_scale"] = [self.depth_scale] * self.num_frames * self.num_copies
        results["si"] = [False] * self.num_frames * self.num_copies
        results["dense"] = [False] * self.num_frames * self.num_copies
        results["synthetic"] = [False] * self.num_frames * self.num_copies
        results["quality"] = [0] * self.num_frames * self.num_copies
        results["valid_camera"] = [True] * self.num_frames * self.num_copies
        results["valid_pose"] = [True] * self.num_frames * self.num_copies
        return results

    def eval_mask(self, valid_mask):
        return valid_mask

    def chunk(self, dataset, chunk_dim=1, pct=1.0):
        subsampled_datasets = [
            x
            for i in range(0, len(dataset), int(1 / pct * chunk_dim))
            for x in dataset[i : i + chunk_dim]
        ]
        return subsampled_datasets

    @abstractmethod
    def preprocess(self, results):
        raise NotImplementedError

    @abstractmethod
    def postprocess(self, results):
        raise NotImplementedError

    @abstractmethod
    def get_mapper(self):
        raise NotImplementedError

    @abstractmethod
    def get_intrinsics(self, idx, image_name):
        raise NotImplementedError

    @abstractmethod
    def get_extrinsics(self, idx, image_name):
        raise NotImplementedError

    @abstractmethod
    def load_dataset(self):
        raise NotImplementedError

    @abstractmethod
    def get_single_item(self, idx, sample=None, mapper=None):
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, idx):
        raise NotImplementedError
