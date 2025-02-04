import json
import os

import h5py
import numpy as np
import torch

from unidepth.datasets.image_dataset import ImageDataset
from unidepth.datasets.utils import DatasetFromList


class KenBurns(ImageDataset):
    min_depth = 0.05
    max_depth = 50.0
    depth_scale = 256.0
    test_split = "val.txt"
    train_split = "train.txt"
    intrisics_file = "intrinsics.json"
    hdf5_paths = [f"3dkenburns/3DKenBurns_{i}.hdf5" for i in range(8)]

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
        txt_string = txt_file.tostring().decode("ascii").strip("\n")
        intrinsics = np.array(h5file[self.intrisics_file]).tostring().decode("ascii")
        intrinsics = json.loads(intrinsics)

        # with open(os.path.join(os.environ["TMPDIR"], self.split_file), "w") as f:
        #     f.write(txt_string)
        # with open(os.path.join(os.environ["TMPDIR"], self.intrisics_file), "w") as f:
        #     json.dump(intrinsics, f)

        dataset = []
        for line in txt_string.split("\n"):
            image_filename, depth_filename, chunk_idx = line.strip().split(" ")
            intrinsics_val = torch.tensor(intrinsics[image_filename]).squeeze()[:, :3]
            sample = [image_filename, depth_filename, intrinsics_val, chunk_idx]
            dataset.append(sample)
        h5file.close()

        if not self.test_mode:
            dataset = self.chunk(dataset, chunk_dim=1, pct=self.mini)

        if self.test_mode and not self.benchmark:  # corresponds to 500 images
            dataset = self.chunk(dataset, chunk_dim=1, pct=0.25)

        self.dataset = DatasetFromList(dataset)
        self.log_load_dataset()

    def get_mapper(self):
        return {
            "image_filename": 0,
            "depth_filename": 1,
            "K": 2,
            "chunk_idx": 3,
        }

    def pre_pipeline(self, results):
        results = super().pre_pipeline(results)
        results["dense"] = [True] * self.num_copies
        results["synthetic"] = [True] * self.num_copies
        results["quality"] = [0] * self.num_copies
        return results
