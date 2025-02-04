import json
import os
from typing import Any

import h5py
import numpy as np
import torch

from unidepth.datasets.pipelines import AnnotationMask, Compose, KittiCrop
from unidepth.datasets.sequence_dataset import SequenceDataset
from unidepth.utils import identity


class KITTIRMVD(SequenceDataset):
    min_depth = 0.05
    max_depth = 80.0
    depth_scale = 256.0
    default_fps = 10
    test_split = "test.txt"
    train_split = "test.txt"
    sequences_file = "sequences.json"
    hdf5_paths = ["kitti_rmvd.hdf5"]

    def __init__(
        self,
        image_shape,
        split_file,
        test_mode,
        crop=None,
        augmentations_db={},
        normalize=True,
        resize_method="hard",
        mini: float = 1.0,
        num_frames: int = 1,
        benchmark: bool = False,
        decode_fields: list[str] = ["image", "depth"],
        inplace_fields: list[str] = ["K", "cam2w"],
        **kwargs,
    ):
        super().__init__(
            image_shape=image_shape,
            split_file=split_file,
            test_mode=test_mode,
            benchmark=benchmark,
            normalize=normalize,
            augmentations_db=augmentations_db,
            resize_method=resize_method,
            mini=mini,
            num_frames=num_frames,
            decode_fields=decode_fields,
            inplace_fields=inplace_fields,
            **kwargs,
        )
        self.crop = crop
        self.resizer = Compose([KittiCrop(crop_size=(352, 1216)), self.resizer])

    def eval_mask(self, valid_mask, info={}):
        """Do grag_crop or eigen_crop for testing"""
        mask_height, mask_width = valid_mask.shape[-2:]
        eval_mask = torch.zeros_like(valid_mask)
        if "garg" in self.crop:
            eval_mask[
                ...,
                int(0.40810811 * mask_height) : int(0.99189189 * mask_height),
                int(0.03594771 * mask_width) : int(0.96405229 * mask_width),
            ] = 1
        elif "eigen" in self.crop:
            eval_mask[
                ...,
                int(0.3324324 * mask_height) : int(0.91351351 * mask_height),
                int(0.03594771 * mask_width) : int(0.96405229 * mask_width),
            ] = 1
        else:
            return valid_mask
        return torch.logical_and(valid_mask, eval_mask)
