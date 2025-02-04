import os

import h5py
import numpy as np
import torch

from unidepth.datasets.image_dataset import ImageDataset
from unidepth.datasets.pipelines import AnnotationMask
from unidepth.datasets.utils import DatasetFromList
from unidepth.utils import identity


class NYUv2Depth(ImageDataset):
    CAM_INTRINSIC = {
        "ALL": torch.tensor(
            [
                [5.1885790117450188e02, 0, 3.2558244941119034e02],
                [0, 5.1946961112127485e02, 2.5373616633400465e02],
                [0, 0, 1],
            ]
        )
    }
    min_depth = 0.005
    max_depth = 10.0
    depth_scale = 1000.0
    log_mean = 0.9140
    log_std = 0.4825
    test_split = "nyu_test.txt"
    train_split = "nyu_train.txt"
    hdf5_paths = ["nyuv2.hdf5"]

    def __init__(
        self,
        image_shape,
        split_file,
        test_mode,
        crop=None,
        benchmark=False,
        augmentations_db={},
        normalize=True,
        resize_method="hard",
        mini=1.0,
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
            **kwargs,
        )
        self.masker = AnnotationMask(
            min_value=0.0,
            max_value=self.max_depth if test_mode else None,
            custom_fn=self.eval_mask if test_mode else lambda x, *args, **kwargs: x,
        )
        self.test_mode = test_mode
        self.load_dataset()

    def load_dataset(self):
        h5file = h5py.File(
            os.path.join(self.data_root, self.hdf5_paths[0]),
            "r",
            libver="latest",
            swmr=True,
        )
        txt_file = np.array(h5file[self.split_file])
        txt_string = txt_file.tostring().decode("ascii")[:-1]  # correct the -1
        h5file.close()
        dataset = []
        for line in txt_string.split("\n"):
            image_filename, depth_filename, _ = line.strip().split(" ")
            sample = [
                image_filename,
                depth_filename,
            ]
            dataset.append(sample)

        if not self.test_mode:
            dataset = self.chunk(dataset, chunk_dim=1, pct=self.mini)

        self.dataset = DatasetFromList(dataset)
        self.log_load_dataset()

    def pre_pipeline(self, results):
        results = super().pre_pipeline(results)
        results["dense"] = [True] * self.num_copies
        return results

    def get_intrinsics(self, idx, image_name):
        return self.CAM_INTRINSIC["ALL"].clone()

    def eval_mask(self, valid_mask, info={}):
        border_mask = torch.zeros_like(valid_mask)
        border_mask[..., 45:-9, 41:-39] = 1
        return torch.logical_and(valid_mask, border_mask)

    def get_mapper(self):
        return {
            "image_filename": 0,
            "depth_filename": 1,
        }

    def pre_pipeline(self, results):
        results = super().pre_pipeline(results)
        results["dense"] = [True] * self.num_copies
        results["quality"] = [2] * self.num_copies
        return results
