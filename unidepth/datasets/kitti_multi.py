import json
import os
from typing import Any

import h5py
import numpy as np
import torch

from unidepth.datasets.image_dataset import ImageDataset
from unidepth.datasets.pipelines import AnnotationMask, KittiCrop
from unidepth.datasets.sequence_dataset import SequenceDataset
from unidepth.datasets.utils import DatasetFromList
from unidepth.utils import identity


class KITTIMulti(SequenceDataset):
    min_depth = 0.05
    max_depth = 80.0
    depth_scale = 256.0
    default_fps = 10.0
    test_split = "val.txt"
    train_split = "train.txt"
    sequences_file = "sequences.json"
    hdf5_paths = ["KITTI_sequence.hdf5"]

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
        self.test_mode = test_mode

        self.crop = crop

        self.cropper_base = KittiCrop(crop_size=(352, 1216))

        self.masker = AnnotationMask(
            min_value=0.0,
            max_value=self.max_depth if test_mode else None,
            custom_fn=self.eval_mask if test_mode else identity,
        )
        self.eval_last = True

    def __len__(self):
        if self.test_mode:
            return 64  # FIXME: Hardcoded for now
        return len(self.dataset)

    def preprocess(self, results):
        self.resizer.ctx = None
        for i, seq in enumerate(results["sequence_fields"]):
            results[seq] = self.cropper_base(results[seq])
            results[seq] = self.resizer(results[seq])
            for key in results[seq].get("image_fields", ["image"]):
                results[seq][key] = results[seq][key].to(torch.float32) / 255
        results.update({k: v for k, v in results[(0, 0)].items() if "fields" in k})

        results = self.pack_batch(results)
        return results

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
