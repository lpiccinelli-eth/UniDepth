import json
import os
from functools import partial
from typing import Any, Dict, Tuple

import h5py
import numpy as np
import tables
import torch
import torchvision.transforms.v2.functional as TF

from unidepth.datasets.base_dataset import BaseDataset
from unidepth.datasets.utils import DatasetFromList
from unidepth.datasets.utils_decode import (decode_camera, decode_depth,
                                            decode_flow, decode_K, decode_mask,
                                            decode_numpy, decode_rgb,
                                            decode_tensor)
from unidepth.utils.distributed import is_main_process


class SequenceDataset(BaseDataset):
    DECODE_FNS = {
        "image": partial(decode_rgb, name="image"),
        "points": partial(decode_numpy, name="points"),
        "K": partial(decode_K, name="camera"),
        "camera_params": partial(decode_camera, name="camera"),
        "cam2w": partial(decode_tensor, name="cam2w"),
        "depth": partial(decode_depth, name="depth"),
        "flow_fwd": partial(decode_flow, name="flow_fwd"),
        "flow_bwd": partial(decode_flow, name="flow_bwd"),
        "flow_fwd_mask": partial(decode_mask, name="flow_fwd_mask"),
        "flow_bwd_mask": partial(decode_mask, name="flow_bwd_mask"),
    }
    default_fps = 5

    def __init__(
        self,
        image_shape: Tuple[int, int],
        split_file: str,
        test_mode: bool,
        normalize: bool,
        augmentations_db: Dict[str, Any],
        resize_method: str,
        mini: float,
        num_frames: int = 1,
        benchmark: bool = False,
        decode_fields: list[str] = ["image", "depth", "flow_fwd", "flow_fwd_mask"],
        inplace_fields: list[str] = ["K", "cam2w"],
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
        self.num_frames = num_frames
        self.original_num_frames = num_frames
        self.decode_fields = decode_fields
        self.inplace_fields = inplace_fields
        self.fps = self.default_fps
        self.fps_range = kwargs.get("fps_range", None)
        if self.fps_range is not None:
            self.fps_range[1] = min(self.default_fps, self.fps_range[1])

        self.load_dataset()

    def load_dataset(self):
        h5file = h5py.File(
            os.path.join(self.data_root, self.hdf5_paths[0]),
            "r",
            libver="latest",
            swmr=True,
        )
        txt_file = np.array(h5file[self.split_file])
        txt_string = txt_file.tostring().decode("ascii").strip()
        sequences = np.array(h5file[self.sequences_file]).tostring().decode("ascii")
        sequences = json.loads(sequences)
        h5file.close()
        dataset = []
        for line in txt_string.split("\n"):
            if len(line.strip().split(" ")) == 1:
                print(line)
                continue
            sequence_name, num_samples = line.strip().split(" ")
            dataset.append(
                {
                    "sequence_name": sequence_name,
                    "num_samples": int(num_samples),
                    "chunk_idx": 0,
                }
            )

        # filter dataset based on attr "invalid_sequences"
        invalid_sequences = getattr(self, "invalid_sequences", [])
        dataset = [
            sample
            for sample in dataset
            if sample["sequence_name"] not in invalid_sequences
        ]

        self.dataset = DatasetFromList(dataset)
        self.sequences = DatasetFromList(
            [sequences[sample["sequence_name"]] for sample in dataset]
        )
        self.log_load_dataset()

    def get_random_idxs(self, num_samples_sequence):
        if self.num_frames == 1:
            return [np.random.randint(0, num_samples_sequence)], 0

        # Check if we can satisfy the required number of frames
        if self.num_frames > num_samples_sequence:
            raise ValueError(
                "Cannot sample more frames than available in the sequence."
            )

        # Restrict FPS range to be within default FPS
        min_fps, max_fps = self.fps_range
        max_fps = min(max_fps, self.default_fps)
        if min_fps > self.default_fps:
            sampled_fps = self.default_fps
        else:
            # Compute minimal viable FPS
            min_required_fps = (
                self.num_frames / num_samples_sequence
            ) * self.default_fps
            min_fps = max(min_fps, min_required_fps)

            # Sample an FPS from the viable range
            sampled_fps = np.random.uniform(min_fps, max_fps)

        # Compute the stride based on the sampled FPS
        stride = self.default_fps / sampled_fps
        max_start_index = num_samples_sequence - int(stride * (self.num_frames - 1))

        # Ensure a valid starting position
        if max_start_index <= 0:
            raise ValueError(
                "No valid start position allows sampling num_frames with the chosen FPS."
            )

        start_index = np.random.randint(0, max_start_index + 1)

        # Compute indices based on the sampled FPS
        indices = [int(start_index + i * stride) for i in range(self.num_frames)]

        return indices, np.random.randint(0, len(indices))

    def get_test_idxs(self, num_samples_sequence, keyframe_idx):
        if self.num_frames == 1:
            return [
                keyframe_idx if keyframe_idx is not None else num_samples_sequence // 2
            ], 0

        if self.num_frames == -1:
            cap_idxs = min(32, num_samples_sequence)  # CAP 32 images
            idxs = list(
                range(max(0, num_samples_sequence - cap_idxs), num_samples_sequence, 1)
            )
            return idxs, keyframe_idx

        # pick closest keyframe_idx st they are around it or capped by the 0 and max num_samples_sequence
        keyframe_idx = (
            keyframe_idx if keyframe_idx is not None else num_samples_sequence - 1
        )
        excess_tail = 0 - min(0, keyframe_idx - self.num_frames // 2)
        excess_head = (
            max(num_samples_sequence, keyframe_idx + (self.num_frames - 1) // 2)
            - num_samples_sequence
        )
        start = keyframe_idx - self.num_frames // 2 + excess_tail - excess_head
        end = keyframe_idx + (self.num_frames - 1) // 2 + excess_head - excess_tail
        idxs = list(range(start, 1 + end))

        return idxs, idxs.index(keyframe_idx)

    def get_single_sequence(self, idx):
        self.num_frames = self.original_num_frames
        # sequence_name = self.dataset[idx]["sequence_name"]
        sample = self.sequences[idx]
        chunk_idx = int(sample.get("chunk_idx", 0))
        h5_path = os.path.join(self.data_root, self.hdf5_paths[chunk_idx])

        num_samples_sequence = len(sample["image"])
        if self.num_frames > 0 and num_samples_sequence < self.num_frames:
            raise IndexError(f"Sequence {idx} has less than {self.num_frames} frames")
        keyframe_idx = None

        if not self.test_mode:
            idxs, keyframe_idx = self.get_random_idxs(num_samples_sequence)
        else:
            idxs, keyframe_idx = self.get_test_idxs(
                num_samples_sequence, sample.get("keyframe_idx", None)
            )

        self.num_frames = len(idxs)
        results = {}
        results = self.pre_pipeline(results)
        results["sequence_fields"] = [(i, 0) for i in range(self.num_frames)]
        results["keyframe_idx"] = keyframe_idx
        with tables.File(
            h5_path,
            mode="r",
            libver="latest",
            swmr=True,
        ) as h5file_chunk:

            for i, j in enumerate(idxs):
                results[(i, 0)] = {
                    k: v.copy() for k, v in results.items() if "fields" in k
                }
                for inplace_field in self.inplace_fields:
                    inplace_field_ = inplace_field.replace("intrinsics", "K").replace(
                        "extrinsics", "cam2w"
                    )
                    results = self.DECODE_FNS[inplace_field_](
                        results, sample[inplace_field][j], idx=i, sample=sample, j=j
                    )

            for i, j in enumerate(idxs):
                for decode_field in self.decode_fields:
                    results = self.DECODE_FNS[decode_field](
                        results,
                        h5file_chunk,
                        sample[decode_field][j],
                        idx=i,
                        depth_scale=self.depth_scale,
                    )

                results["filename"] = sample["image"][j]

        results = self.preprocess(results)
        if not self.test_mode:
            results = self.augment(results)
        results = self.postprocess(results)
        return results

    def preprocess(self, results):
        results = self.replicate(results)
        for i, seq in enumerate(results["sequence_fields"]):
            results[seq] = self.resizer(results[seq])
            self.resizer.ctx = None if self.num_copies > 1 else self.resizer.ctx
            num_pts = torch.count_nonzero(results[seq]["depth"] > 0)
            if num_pts < 50:
                raise IndexError(f"Too few points in depth map ({num_pts})")

            for key in results[seq].get("image_fields", ["image"]):
                results[seq][key] = results[seq][key].to(torch.float32) / 255

        # update fields common in sequence
        for key in [
            "image_fields",
            "gt_fields",
            "mask_fields",
            "camera_fields",
        ]:
            if key in results[(0, 0)]:
                results[key] = results[(0, 0)][key]

        results = self.pack_batch(results)
        return results

    def postprocess(self, results):
        # # normalize after because color aug requires [0,255]?
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
                results = [self.get_single_sequence(i) for i in idx]
            else:
                results = self.get_single_sequence(idx)
        except Exception as e:
            print(f"Error loading sequence {idx} for {self.__class__.__name__}: {e}")
            idx = np.random.randint(0, len(self.dataset))
            results = self[idx]
        return results

    def log_load_dataset(self):
        if is_main_process():
            info = f"Loaded {self.__class__.__name__} with {sum([len(x['image']) for x in self.sequences])} images in {len(self)} sequences."
            print(info)
