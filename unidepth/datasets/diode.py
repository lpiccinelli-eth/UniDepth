import os

import h5py
import numpy as np
import torch

from unidepth.datasets.image_dataset import ImageDataset
from unidepth.datasets.sequence_dataset import SequenceDataset
from unidepth.datasets.utils import DatasetFromList


class DiodeIndoor(ImageDataset):
    CAM_INTRINSIC = {
        "ALL": torch.tensor([[886.81, 0, 512], [0, 927.06, 384], [0, 0, 1]])
    }
    min_depth = 0.01
    max_depth = 25.0
    depth_scale = 256.0
    test_split = "val.txt"
    train_split = "train.txt"
    hdf5_paths = ["DiodeIndoor.hdf5"]

    def __init__(
        self,
        image_shape,
        split_file,
        test_mode,
        crop=None,
        benchmark=False,
        augmentations_db={},
        normalize=True,
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
            mini=mini,
            **kwargs,
        )
        self.test_mode = test_mode

        # load annotations
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
            sample = [
                image_filename,
                depth_filename,
            ]
            dataset.append(sample)

        if not self.test_mode:
            dataset = self.chunk(dataset, chunk_dim=1, pct=self.mini)

        self.dataset = DatasetFromList(dataset)
        self.log_load_dataset()

    def get_intrinsics(self, *args, **kwargs):
        return self.CAM_INTRINSIC["ALL"].clone()

    def get_mapper(self):
        return {
            "image_filename": 0,
            "depth_filename": 1,
        }

    def pre_pipeline(self, results):
        results = super().pre_pipeline(results)
        results["dense"] = [True] * self.num_copies
        results["quality"] = [1] * self.num_copies
        return results


class DiodeIndoor_F(SequenceDataset):
    min_depth = 0.01
    max_depth = 25.0
    depth_scale = 1000.0
    test_split = "train.txt"
    train_split = "train.txt"
    sequences_file = "sequences.json"
    hdf5_paths = ["DiodeIndoor-F.hdf5"]

    def __init__(
        self,
        image_shape: tuple[int, int],
        split_file: str,
        test_mode: bool,
        normalize: bool,
        augmentations_db: dict[str, float],
        resize_method: str,
        mini: float = 1.0,
        num_frames: int = 1,
        benchmark: bool = False,
        decode_fields: list[str] = ["image", "depth"],
        inplace_fields: list[str] = ["camera_params", "cam2w"],
        **kwargs,
    ) -> None:
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
            decode_fields=(
                decode_fields if not test_mode else [*decode_fields, "points"]
            ),
            inplace_fields=inplace_fields,
            **kwargs,
        )

    def pre_pipeline(self, results):
        results = super().pre_pipeline(results)
        results["dense"] = [True] * self.num_frames * self.num_copies
        results["quality"] = [1] * self.num_frames * self.num_copies
        return results


class DiodeOutdoor(ImageDataset):
    CAM_INTRINSIC = {
        "ALL": torch.tensor([[886.81, 0, 512], [0, 927.06, 384], [0, 0, 1]])
    }
    min_depth = 0.1
    max_depth = 80.0
    log_mean = 0
    log_std = 1
    test_split = "diode_outdoor_val.txt"
    train_split = "diode_outdoor_train.txt"
    hdf5_paths = ["diode.hdf5"]

    def __init__(
        self,
        image_shape,
        split_file,
        test_mode,
        depth_scale=256,
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
        self.depth_scale = depth_scale

        self.masker = AnnotationMask(
            min_value=self.min_depth,
            max_value=self.max_depth if test_mode else None,
            custom_fn=self.eval_mask if test_mode else lambda x, *args, **kwargs: x,
        )
        # load annotations
        self.load_dataset()

    def load_dataset(self):
        self.h5file = h5py.File(
            os.path.join(self.data_root, self.hdf5_path),
            "r",
            libver="latest",
            swmr=True,
        )
        txt_file = np.array(self.h5file[self.split_file])
        txt_string = txt_file.tostring().decode("ascii")[:-1]
        dataset = {"depth_filename": [], "image_filename": []}
        for line in txt_string.split("\n"):
            depth_filename = line.strip().split(" ")[1]
            img_name = line.strip().split(" ")[0]
            image_filename = img_name
            dataset["depth_filename"].append(depth_filename)
            dataset["image_filename"].append(image_filename)

        self.dataset = pl.from_dict(dataset)

        if not self.test_mode and self.mini:
            self.dataset = self.dataset[::2]


class Diode(ImageDataset):
    CAM_INTRINSIC = {
        "ALL": torch.tensor([[886.81, 0, 512], [0, 927.06, 384], [0, 0, 1]])
    }
    log_mean = 0
    log_std = 1
    min_depth = 0.6
    max_depth = 80.0
    test_split = "diode_val.txt"
    train_split = "diode_train.txt"
    hdf5_paths = ["diode.hdf5"]

    def __init__(
        self,
        image_shape,
        split_file,
        test_mode,
        depth_scale=256,
        crop=None,
        benchmark=False,
        augmentations_db={},
        normalize=True,
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
            mini=mini,
            **kwargs,
        )
        self.test_mode = test_mode
        self.depth_scale = depth_scale

        self.masker = AnnotationMask(
            min_value=self.min_depth,
            max_value=self.max_depth if test_mode else None,
            custom_fn=self.eval_mask if test_mode else lambda x, *args, **kwargs: x,
        )
        # load annotations
        self.load_dataset()

    def load_dataset(self):
        self.h5file = h5py.File(
            os.path.join(self.data_root, self.hdf5_path),
            "r",
            libver="latest",
            swmr=True,
        )
        txt_file = np.array(self.h5file[self.split_file])
        txt_string = txt_file.tostring().decode("ascii")[:-1]
        dataset = {"depth_filename": [], "image_filename": []}
        for line in txt_string.split("\n"):
            depth_filename = line.strip().split(" ")[1]
            image_filename = line.strip().split(" ")[0]
            dataset["depth_filename"].append(depth_filename)
            dataset["image_filename"].append(image_filename)

        self.dataset = pl.from_dict(dataset)

        if not self.test_mode and self.mini:
            self.dataset = self.dataset[::2]

    def get_intrinsics(self, *args, **kwargs):
        return self.CAM_INTRINSIC["ALL"].clone()
