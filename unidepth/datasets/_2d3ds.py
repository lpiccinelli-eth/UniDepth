from typing import Any

import torch

from unidepth.datasets.pipelines import Compose, PanoCrop, PanoRoll
from unidepth.datasets.sequence_dataset import SequenceDataset


class _2D3DS(SequenceDataset):
    min_depth = 0.01
    max_depth = 10.0
    depth_scale = 512.0
    test_split = "train.txt"
    train_split = "train.txt"
    sequences_file = "sequences.json"
    hdf5_paths = [f"2D3DS.hdf5"]

    def __init__(
        self,
        image_shape: tuple[int, int],
        split_file: str,
        test_mode: bool,
        normalize: bool,
        augmentations_db: dict[str, Any],
        resize_method: str,
        mini: float = 1.0,
        num_frames: int = 1,
        benchmark: bool = False,
        decode_fields: list[str] = ["image", "depth"],
        inplace_fields: list[str] = ["cam2w", "camera_params"],
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
            decode_fields=decode_fields,
            inplace_fields=inplace_fields,
            **kwargs,
        )
        self.resizer = Compose(
            [PanoCrop(), PanoRoll(test_mode=test_mode), self.resizer]
        )

    def preprocess(self, results):
        self.resizer.ctx = None
        if self.test_mode:
            for i, seq in enumerate(results["sequence_fields"]):
                results[seq]["points"] = results[seq]["camera"].reconstruct(
                    results[seq]["depth"]
                )
                results[seq]["depth"] = results[seq]["points"][:, -1:]
                results[seq]["gt_fields"].add("points")
        return super().preprocess(results)

    def pre_pipeline(self, results):
        results = super().pre_pipeline(results)
        results["dense"] = [True] * self.num_frames * self.num_copies
        results["synthetic"] = [False] * self.num_frames * self.num_copies
        results["quality"] = [1] * self.num_frames * self.num_copies
        return results
