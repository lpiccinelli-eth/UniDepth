from collections.abc import Sequence

import numpy as np
import torch


class Collect(object):
    def __init__(
        self,
        keys,
        meta_keys=(
            "filename",
            "keyframe_idx",
            "sequence_name",
            "image_filename",
            "depth_filename",
            "image_ori_shape",
            "camera",
            "original_camera",
            "sfm",
            "image_shape",
            "resized_shape",
            "scale_factor",
            "rotation",
            "resize_factor",
            "flip",
            "flip_direction",
            "dataset_name",
            "paddings",
            "max_value",
            "log_mean",
            "log_std",
            "image_rescale",
            "focal_rescale",
            "depth_rescale",
        ),
    ):
        self.keys = keys
        self.meta_keys = meta_keys

    def __call__(self, results):
        data_keys = [key for field in self.keys for key in results.get(field, [])]
        data = {
            key: {
                sequence_key: results[key][sequence_key]
                for sequence_key in results["sequence_fields"]
            }
            for key in data_keys
        }
        data["img_metas"] = {
            key: value for key, value in results.items() if key not in data_keys
        }
        return data

    def __repr__(self):
        return (
            self.__class__.__name__ + f"(keys={self.keys}, meta_keys={self.meta_keys})"
        )


class AnnotationMask(object):
    def __init__(self, min_value, max_value, custom_fn=lambda x: x):
        self.min_value = min_value
        self.max_value = max_value
        self.custom_fn = custom_fn

    def __call__(self, results):
        for key in results.get("gt_fields", []):
            if key + "_mask" in results["mask_fields"]:
                if "flow" in key:
                    for sequence_idx in results.get("sequence_fields", []):
                        boundaries = (results[key][sequence_idx] >= -1) & (
                            results[key][sequence_idx] <= 1
                        )
                        boundaries = boundaries[:, :1] & boundaries[:, 1:]
                        results[key + "_mask"][sequence_idx] = (
                            results[key + "_mask"][sequence_idx] & boundaries
                        )
                    continue
            for sequence_idx in results.get("sequence_fields", []):
                mask = results[key][sequence_idx] > self.min_value
                if self.max_value is not None:
                    mask = mask & (results[key][sequence_idx] < self.max_value)
                mask = self.custom_fn(mask, info=results)
                if key + "_mask" not in results:
                    results[key + "_mask"] = {}
                results[key + "_mask"][sequence_idx] = mask.bool()
            results["mask_fields"].add(key + "_mask")
        return results

    def __repr__(self):
        return (
            self.__class__.__name__
            + f"(min_value={self.min_value}, max_value={ self.max_value})"
        )
