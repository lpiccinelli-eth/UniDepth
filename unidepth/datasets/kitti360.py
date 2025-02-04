from typing import Any

import torch

from unidepth.datasets.sequence_dataset import SequenceDataset


class KITTI360(SequenceDataset):
    min_depth = 0.01
    max_depth = 80.0
    depth_scale = 256.0
    train_split = "train.txt"
    test_split = "val_split.txt"
    sequences_file = "sequences_split.json"
    hdf5_paths = [f"KITTI360.hdf5"]

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

    def preprocess(self, results):
        self.resizer.ctx = None
        for i, seq in enumerate(results["sequence_fields"]):
            # Create a mask where the distance from the center is less than H/2
            H, W = results[seq]["image"].shape[-2:]
            x = torch.linspace(-W / 2, W / 2, W)
            y = torch.linspace(-H / 2, H / 2, H)
            xv, yv = torch.meshgrid(x, y, indexing="xy")
            distance_from_center = torch.sqrt(xv**2 + yv**2).reshape(1, 1, H, W)
            results[seq]["validity_mask"] = distance_from_center < (H / 2)
        return super().preprocess(results)

    def pre_pipeline(self, results):
        results = super().pre_pipeline(results)
        results["dense"] = [False] * self.num_frames * self.num_copies
        results["quality"] = [1] * self.num_frames * self.num_copies
        return results
