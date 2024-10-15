import os

import h5py
import numpy as np
import torch

from unidepth.datasets.image_dataset import ImageDataset
from unidepth.datasets.pipelines import AnnotationMask, KittiCrop
from unidepth.datasets.utils import DatasetFromList
from unidepth.utils import identity


class KITTI(ImageDataset):
    CAM_INTRINSIC = {
        "2011_09_26": torch.tensor(
            [
                [7.215377e02, 0.000000e00, 6.095593e02, 4.485728e01],
                [0.000000e00, 7.215377e02, 1.728540e02, 2.163791e-01],
                [0.000000e00, 0.000000e00, 1.000000e00, 2.745884e-03],
            ]
        ),
        "2011_09_28": torch.tensor(
            [
                [7.070493e02, 0.000000e00, 6.040814e02, 4.575831e01],
                [0.000000e00, 7.070493e02, 1.805066e02, -3.454157e-01],
                [0.000000e00, 0.000000e00, 1.000000e00, 4.981016e-03],
            ]
        ),
        "2011_09_29": torch.tensor(
            [
                [7.183351e02, 0.000000e00, 6.003891e02, 4.450382e01],
                [0.000000e00, 7.183351e02, 1.815122e02, -5.951107e-01],
                [0.000000e00, 0.000000e00, 1.000000e00, 2.616315e-03],
            ]
        ),
        "2011_09_30": torch.tensor(
            [
                [7.070912e02, 0.000000e00, 6.018873e02, 4.688783e01],
                [0.000000e00, 7.070912e02, 1.831104e02, 1.178601e-01],
                [0.000000e00, 0.000000e00, 1.000000e00, 6.203223e-03],
            ]
        ),
        "2011_10_03": torch.tensor(
            [
                [7.188560e02, 0.000000e00, 6.071928e02, 4.538225e01],
                [0.000000e00, 7.188560e02, 1.852157e02, -1.130887e-01],
                [0.000000e00, 0.000000e00, 1.000000e00, 3.779761e-03],
            ]
        ),
    }
    min_depth = 0.05
    max_depth = 80.0
    depth_scale = 256.0
    test_split = "kitti_eigen_test.txt"
    train_split = "kitti_eigen_train.txt"
    test_split_benchmark = "kitti_test.txt"
    hdf5_paths = ["kitti.hdf5"]

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
        self.cropper_base = KittiCrop(crop_size=(352, 1216))
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
            image_filename = line.strip().split(" ")[0]
            depth_filename = line.strip().split(" ")[1]
            if depth_filename == "None":
                continue
            sample = [
                image_filename,
                depth_filename,
            ]
            dataset.append(sample)

        self.dataset = DatasetFromList(dataset)
        self.log_load_dataset()

    def get_intrinsics(self, idx, image_name):
        return self.CAM_INTRINSIC[image_name.split("/")[0]][:, :3].clone()

    def preprocess(self, results):
        results = self.cropper_base(results)
        results = self.resizer(results)
        for key in results.get("image_fields", ["image"]):
            results[key] = results[key].to(torch.float32) / 255
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
        return torch.logical_and(valid_mask, eval_mask)

    def get_mapper(self):
        return {
            "image_filename": 0,
            "depth_filename": 1,
        }

    def pre_pipeline(self, results):
        results = super().pre_pipeline(results)
        results["dense"] = [False]
        return results
