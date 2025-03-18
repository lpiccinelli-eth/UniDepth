import io
import os
from time import time
from typing import Any, Dict, List, Tuple

import numpy as np
import tables
import torch
import torchvision
import torchvision.transforms.v2.functional as TF
from PIL import Image

from unidepth.datasets.base_dataset import BaseDataset
from unidepth.utils import is_main_process
from unidepth.utils.camera import BatchCamera, Pinhole

"""
Awful class for legacy reasons, we assume only pinhole cameras
And we "fake" sequences by setting sequence_fields to [(0, 0)] and cam2w as eye(4)
"""


class ImageDataset(BaseDataset):
    def __init__(
        self,
        image_shape: Tuple[int, int],
        split_file: str,
        test_mode: bool,
        normalize: bool,
        augmentations_db: Dict[str, Any],
        resize_method: str,
        mini: float,
        benchmark: bool = False,
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
            **kwargs,
        )
        self.mapper = self.get_mapper()

    def get_single_item(self, idx, sample=None, mapper=None):
        sample = self.dataset[idx] if sample is None else sample
        mapper = self.mapper if mapper is None else mapper

        results = {
            (0, 0): dict(
                gt_fields=set(),
                image_fields=set(),
                mask_fields=set(),
                camera_fields=set(),
            )
        }
        results = self.pre_pipeline(results)
        results["sequence_fields"] = [(0, 0)]

        chunk_idx = (
            int(sample[self.mapper["chunk_idx"]]) if "chunk_idx" in self.mapper else 0
        )
        h5_path = os.path.join(self.data_root, self.hdf5_paths[chunk_idx])
        with tables.File(
            h5_path,
            mode="r",
            libver="latest",
            swmr=True,
        ) as h5file_chunk:
            for key_mapper, idx_mapper in mapper.items():
                if "image" not in key_mapper and "depth" not in key_mapper:
                    continue
                value = sample[idx_mapper]
                results[(0, 0)][key_mapper] = value
                name = key_mapper.replace("_filename", "")
                value_root = "/" + value

                if "image" in key_mapper:
                    results[(0, 0)]["filename"] = value
                    file = h5file_chunk.get_node(value_root).read()
                    image = (
                        torchvision.io.decode_image(torch.from_numpy(file))
                        .to(torch.uint8)
                        .squeeze()
                    )
                    results[(0, 0)]["image_fields"].add(name)
                    results[(0, 0)][f"image_ori_shape"] = image.shape[-2:]
                    results[(0, 0)][name] = image[None, ...]

                    # collect camera information for the given image
                    name = name.replace("image_", "")
                    results[(0, 0)]["camera_fields"].update({"camera", "cam2w"})
                    K = self.get_intrinsics(idx, value)
                    if K is None:
                        K = torch.eye(3)
                        K[0, 0] = K[1, 1] = 0.7 * self.image_shape[1]
                        K[0, 2] = 0.5 * self.image_shape[1]
                        K[1, 2] = 0.5 * self.image_shape[0]

                    camera = Pinhole(K=K[None, ...].clone())
                    results[(0, 0)]["camera"] = BatchCamera.from_camera(camera)
                    results[(0, 0)]["cam2w"] = self.get_extrinsics(idx, value)[
                        None, ...
                    ]

                elif "depth" in key_mapper:
                    # start = time()
                    file = h5file_chunk.get_node(value_root).read()
                    depth = Image.open(io.BytesIO(file))
                    depth = TF.pil_to_tensor(depth).squeeze().to(torch.float32)
                    if depth.ndim == 3:
                        depth = depth[2] + depth[1] * 255 + depth[0] * 255 * 255

                    results[(0, 0)]["gt_fields"].add(name)
                    results[(0, 0)][f"depth_ori_shape"] = depth.shape

                    depth = (
                        depth.view(1, 1, *depth.shape).contiguous() / self.depth_scale
                    )
                    results[(0, 0)][name] = depth

        results = self.preprocess(results)
        if not self.test_mode:
            results = self.augment(results)
        results = self.postprocess(results)
        return results

    def preprocess(self, results):
        results = self.replicate(results)
        for i, seq in enumerate(results["sequence_fields"]):
            self.resizer.ctx = None
            results[seq] = self.resizer(results[seq])
            num_pts = torch.count_nonzero(results[seq]["depth"] > 0)
            if num_pts < 50:
                raise IndexError(f"Too few points in depth map ({num_pts})")

            for key in results[seq].get("image_fields", ["image"]):
                results[seq][key] = results[seq][key].to(torch.float32) / 255

        # update fields common in sequence
        for key in ["image_fields", "gt_fields", "mask_fields", "camera_fields"]:
            if key in results[(0, 0)]:
                results[key] = results[(0, 0)][key]
        results = self.pack_batch(results)
        return results

    def postprocess(self, results):
        # normalize after because color aug requires [0,255]?
        for key in results.get("image_fields", ["image"]):
            results[key] = TF.normalize(results[key], **self.normalization_stats)
        results = self.filler(results)
        results = self.unpack_batch(results)
        results = self.masker(results)
        results = self.collecter(results)
        return results

    def __getitem__(self, idx):
        try:
            if isinstance(idx, (list, tuple)):
                results = [self.get_single_item(i) for i in idx]
            else:
                results = self.get_single_item(idx)
        except Exception as e:
            print(f"Error loading sequence {idx} for {self.__class__.__name__}: {e}")
            idx = np.random.randint(0, len(self.dataset))
            results = self[idx]
        return results

    def get_intrinsics(self, idx, image_name):
        idx_sample = self.mapper.get("K", 1000)
        sample = self.dataset[idx]
        if idx_sample >= len(sample):
            return None
        return sample[idx_sample]

    def get_extrinsics(self, idx, image_name):
        idx_sample = self.mapper.get("cam2w", 1000)
        sample = self.dataset[idx]
        if idx_sample >= len(sample):
            return torch.eye(4)
        return sample[idx_sample]

    def get_mapper(self):
        return {
            "image_filename": 0,
            "depth_filename": 1,
            "K": 2,
        }
