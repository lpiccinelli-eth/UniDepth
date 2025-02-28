import os

import h5py
import numpy as np
import torch

from unidepth.datasets.image_dataset import ImageDataset
from unidepth.datasets.utils import DatasetFromList


class HRWSI(ImageDataset):
    min_depth = 0.01
    max_depth = 1000.0
    depth_scale = 50.0
    test_split = "val.txt"
    train_split = "train.txt"
    hdf5_paths = ["HRWSI.hdf5"]

    def __init__(
        self,
        image_shape,
        split_file,
        test_mode,
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
        # with open(os.path.join(os.environ["TMPDIR"], self.split_file), "w") as f:
        #     f.write(txt_string)

        dataset = []

        for line in txt_string.split("\n"):
            image_filename, depth_filename = line.strip().split(" ")
            sample = [
                image_filename,
                depth_filename,
            ]
            dataset.append(sample)
        h5file.close()
        if not self.test_mode:
            dataset = self.chunk(dataset, chunk_dim=1, pct=self.mini)

        self.dataset = DatasetFromList(dataset)
        self.log_load_dataset()

    def pre_pipeline(self, results):
        results = super().pre_pipeline(results)
        results["ssi"] = [True]
        results["valid_camera"] = [False]
        return results

    def get_mapper(self):
        return {
            "image_filename": 0,
            "depth_filename": 1,
        }
