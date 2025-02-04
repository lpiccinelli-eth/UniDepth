import copy
import multiprocessing as mp
import pickle
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np
import torch
import torch.utils.data

from unidepth.utils.distributed import (all_gather, get_local_rank,
                                        get_local_size, get_rank,
                                        get_world_size)


class ConcatDataset(torch.utils.data.ConcatDataset):
    def __init__(self, datasets, shape_constraints: dict[str, list[int]] = {}):
        super().__init__(datasets)

        self.sample = shape_constraints["sample"]
        self.shape_mult = shape_constraints["shape_mult"]
        self.ratio_bounds = shape_constraints["ratio_bounds"]
        self.pixels_max = shape_constraints["pixels_max"]
        self.pixels_min = shape_constraints["pixels_min"]

        self.height_min = shape_constraints["height_min"]
        self.width_min = shape_constraints["width_min"]

    def sample_shape(self):
        if not self.sample:
            return
        # 1: sample image ratio
        ratio = np.random.uniform(*self.ratio_bounds)
        pixels_min = self.pixels_min // (self.shape_mult * self.shape_mult)
        pixels_max = self.pixels_max // (self.shape_mult * self.shape_mult)
        # 2: sample image height or width, if ratio > 1 or < 1
        if ratio > 1:
            height_min = max(self.height_min, np.sqrt(pixels_min / ratio))
            height = np.random.uniform(height_min, np.sqrt(pixels_max / ratio))
            width = height * ratio
        else:
            width_min = max(self.width_min, np.sqrt(pixels_min * ratio))
            width = np.random.uniform(width_min, np.sqrt(pixels_max * ratio))
            height = width / ratio
        # 3: get final shape based on the shape_mult
        shape = [int(height) * self.shape_mult, int(width) * self.shape_mult]
        for dataset in self.datasets:
            setattr(dataset, "image_shape", shape)
            setattr(dataset.resizer, "image_shape", shape)

    def __getitem__(self, idxs):
        self.sample_shape()
        return [super(ConcatDataset, self).__getitem__(idx) for idx in idxs]


def _paddings(image_shape, network_shape):
    cur_h, cur_w = image_shape
    h, w = network_shape
    pad_top, pad_bottom = (h - cur_h) // 2, h - cur_h - (h - cur_h) // 2
    pad_left, pad_right = (w - cur_w) // 2, w - cur_w - (w - cur_w) // 2
    return pad_left, pad_right, pad_top, pad_bottom


def collate_fn(in_data: List[List[Dict[str, Any]]], is_batched: bool = True):
    out_data = defaultdict(list)
    img_metas = []
    in_data = in_data[0] if is_batched else in_data

    # get max_shape and paddings
    shapes = [tensor.shape[-2:] for x in in_data for tensor in x["depth"].values()]
    max_shape_tuple = tuple(max(elements) for elements in zip(*shapes))
    paddings = [
        [
            _paddings(tensor.shape[-2:], max_shape_tuple)
            for tensor in x["depth"].values()
        ]
        for x in in_data
    ]

    for x in in_data:  # here iter over batches
        padding = paddings.pop(0)
        for k, v in x.items():
            if "img_metas" not in k:
                values = list(v.values())
                v = torch.cat(values)
                out_data[k].append(v)
            else:
                v["depth_paddings"] = padding
                img_metas.append(v)

    output_dict = {
        "data": {k: torch.stack(v, dim=0) for k, v in out_data.items()},
        "img_metas": img_metas,
    }
    # camera are always flattened and the stack/cat so if list of B times (T, 3, 3) cameras
    # it goes to (B * T, 3, 3), to be consistent with the image shape -> reshape
    if "camera" in output_dict["data"]:
        output_dict["data"]["camera"] = output_dict["data"]["camera"].reshape(
            *output_dict["data"]["image"].shape[:2]
        )
    return output_dict


def local_scatter(array: list[Any]):
    """
    Scatter an array from local leader to all local workers.
    The i-th local worker gets array[i].

    Args:
        array: Array with same size of #local workers.
    """
    if get_world_size() == 1:
        return array[0]
    if get_local_rank() == 0:
        assert len(array) == get_local_size()
        all_gather(array)
    else:
        all_data = all_gather(None)
        array = all_data[get_rank() - get_local_rank()]
    return array[get_local_rank()]


class DatasetFromList(torch.utils.data.Dataset):  # type: ignore
    """Wrap a list to a torch Dataset.

    We serialize and wrap big python objects in a torch.Dataset due to a
    memory leak when dealing with large python objects using multiple workers.
    See: https://github.com/pytorch/pytorch/issues/13246
    """

    def __init__(self, lst: List[Any], deepcopy: bool = False, serialize: bool = True):
        """Creates an instance of the class.

        Args:
            lst: a list which contains elements to produce.
            deepcopy: whether to deepcopy the element when producing it, s.t.
            the result can be modified in place without affecting the source
            in the list.
            serialize: whether to hold memory using serialized objects. When
            enabled, data loader workers can use shared RAM from master
            process instead of making a copy.
        """
        self._copy = deepcopy
        self._serialize = serialize

        def _serialize(data: Any):
            buffer = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
            return torch.frombuffer(buffer, dtype=torch.uint8)

        if self._serialize:
            # load only on 0th rank
            if get_local_rank() == 0:
                _lst = [_serialize(x) for x in lst]
                self._addr = torch.cumsum(
                    torch.tensor([len(x) for x in _lst], dtype=torch.int64), dim=0
                )
                self._lst = torch.concatenate(_lst)
                # Move data to shared memory, obtain a handle to send to each local worker.
                handles = [None] + [
                    bytes(mp.reduction.ForkingPickler.dumps((self._addr, self._lst)))
                    for _ in range(get_local_size() - 1)
                ]
            else:
                handles = None

            # Each worker receives the handle from local leader (rank 0)
            # then materialize the tensor from shared memory
            handle = local_scatter(handles)
            if get_local_rank() > 0:
                self._addr, self._lst = mp.reduction.ForkingPickler.loads(handle)

        else:
            self._lst = lst

    def __len__(self) -> int:
        """Return len of list."""
        if self._serialize:
            return len(self._addr)
        return len(self._lst)

    def __getitem__(self, idx: int) -> Any:
        """Return item of list at idx."""
        if self._serialize:
            start_addr = 0 if idx == 0 else self._addr[idx - 1]
            end_addr = self._addr[idx]
            bytes_ = memoryview(self._lst[start_addr:end_addr].numpy())
            return pickle.loads(bytes_)
        if self._copy:
            return copy.deepcopy(self._lst[idx])

        return self._lst[idx]


def get_weights(
    train_datasets: dict[str, torch.utils.data.Dataset], sampling: dict[str, float]
) -> torch.Tensor:
    from .image_dataset import ImageDataset
    from .sequence_dataset import SequenceDataset

    weights = []
    num_samples = 0
    info_weights = {}
    for dataset_name, dataset in train_datasets.items():
        assert (
            dataset_name in sampling
        ), f"Dataset {dataset_name} not found in {sampling.keys()}"

        if isinstance(dataset, ImageDataset):
            # sum of all samples has weight as in sampling s.t. sampling dataset in general is as in sampling
            # inside is uniform
            weight = sampling[dataset_name] / len(dataset)
            weights.append(torch.full((len(dataset),), weight).double())
            num_samples += len(dataset)

        elif isinstance(dataset, SequenceDataset):
            # local weight is num_samples, but global must be as in sampling
            # hence is num_samples / (sum num_samples / sampling[dataset_name])
            # s.t. sampling anything from the dataset is
            # sum(num_samples / (sum num_samples / sampling[dataset_name]))
            # -> sampling[dataset_name]
            numerator = [int(data["num_samples"]) for data in dataset.dataset]
            weights.append(
                sampling[dataset_name]
                * torch.tensor(numerator).double()
                / sum(numerator)
            )
            num_samples += sum(numerator)

        else:
            weight = sampling[dataset_name] / len(dataset)
            weights.append(torch.full((len(dataset),), weight).double())

        info_weights[dataset_name] = weights[-1][-1]

    return torch.cat(weights), num_samples
