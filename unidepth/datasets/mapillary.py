import json
import os

import h5py
import numpy as np
import torch

from unidepth.datasets.image_dataset import ImageDataset
from unidepth.datasets.utils import DatasetFromList


class Mapillary(ImageDataset):
    min_depth = 0.01
    max_depth = 70.0
    depth_scale = 256.0
    test_split = "mapillary_val.txt"
    train_split = "mapillary_train_clean.txt"
    intrisics_file = "intrinsics.json"
    hdf5_paths = ["Mapillary.hdf5"]

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
        txt_string = txt_file.tostring().decode("ascii")  # [:-1] # correct the -1
        intrinsics = np.array(h5file[self.intrisics_file]).tostring().decode("ascii")
        intrinsics = json.loads(intrinsics)

        dataset = []
        for line in txt_string.split("\n"):
            image_filename, depth_filename = line.strip().split(" ")
            intrinsics_val = torch.tensor(intrinsics[image_filename]).squeeze()[:, :3]
            sample = [image_filename, depth_filename, intrinsics_val]
            dataset.append(sample)
        h5file.close()

        if not self.test_mode:
            dataset = self.chunk(dataset, chunk_dim=1, pct=self.mini)
        if self.test_mode and not self.benchmark:
            dataset = self.chunk(dataset, chunk_dim=1, pct=0.05)

        self.dataset = DatasetFromList(dataset)
        self.log_load_dataset()

    def pre_pipeline(self, results):
        results = super().pre_pipeline(results)
        results["si"] = [True] * self.num_copies
        results["valid_camera"] = [False] * self.num_copies
        results["dense"] = [False] * self.num_copies
        results["quality"] = [2] * self.num_copies
        return results
