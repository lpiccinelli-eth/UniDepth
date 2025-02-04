import os

import h5py
import numpy as np
import torch

from unidepth.datasets.image_dataset import ImageDataset
from unidepth.datasets.utils import DatasetFromList


class FLSea(ImageDataset):
    CAM_INTRINSIC = {
        "canyons": torch.tensor(
            [
                [1175.3913431656817, 0.0, 466.2595428966926],
                [0.0, 1174.2805075232263, 271.2116633091501],
                [0.0, 0.0, 1.0],
            ]
        ),
        "red_sea": torch.tensor(
            [
                [1296.666758476217, 0.0, 501.50386149846],
                [0.0, 1300.831316354508, 276.161712082695],
                [0.0, 0.0, 1.0],
            ]
        ),
    }
    min_depth = 0.05
    max_depth = 20.0
    depth_scale = 1000.0
    train_split = "train.txt"
    hdf5_paths = ["FLSea.hdf5"]

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
        mini=False,
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
        self.test_mode = test_mode

        self.crop = crop
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
            image_filename, depth_filename = line.strip().split(" ")
            sample = [image_filename, depth_filename]
            dataset.append(sample)

        if not self.test_mode:
            dataset = self.chunk(dataset, chunk_dim=1, pct=self.mini)
        if self.test_mode and not self.benchmark:
            dataset = self.chunk(dataset, chunk_dim=1, pct=0.33)

        self.dataset = DatasetFromList(dataset)
        self.log_load_dataset()

    def get_intrinsics(self, idx, image_name):
        return self.CAM_INTRINSIC[image_name.split("/")[0]][:, :3].clone()

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
