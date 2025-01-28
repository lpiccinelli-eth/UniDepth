from typing import Any

import torch

from unidepth.datasets.sequence_dataset import SequenceDataset


class ADT(SequenceDataset):
    min_depth = 0.01
    max_depth = 20.0
    depth_scale = 1000.0
    test_split = "val.txt"
    train_split = "train.txt"
    sequences_file = "sequences.json"
    hdf5_paths = [f"ADT.hdf5"]

    def __init__(
        self,
        image_shape: tuple[int, int],
        split_file: str,
        test_mode: bool,
        normalize: bool,
        augmentations_db: dict[str, Any],
        resize_method: str,
        mini: float,
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
            decode_fields=decode_fields,  # if not test_mode else [*decode_fields, "points"],
            inplace_fields=inplace_fields,
            **kwargs,
        )

    def preprocess(self, results):
        self.resizer.ctx = None
        for i, seq in enumerate(results["sequence_fields"]):
            # Create a mask where the distance from the center is less than H/2
            H, W = results[seq]["image"].shape[-2:]
            x = torch.linspace(-W / 2 - 0.5, W / 2 + 0.5, W)
            y = torch.linspace(-H / 2 - 0.5, H / 2 + 0.5, H)
            xv, yv = torch.meshgrid(x, y, indexing="xy")
            distance_from_center = torch.sqrt(xv**2 + yv**2).reshape(1, 1, H, W)
            results[seq]["validity_mask"] = distance_from_center < (H / 2) + 20
            results[seq]["depth_mask"] = results[seq]["validity_mask"].clone()
            results[seq]["mask_fields"].add("depth_mask")
            results[seq]["mask_fields"].add("validity_mask")
        return super().preprocess(results)

    def pre_pipeline(self, results):
        results = super().pre_pipeline(results)
        results["dense"] = [True] * self.num_frames * self.num_copies
        results["synthetic"] = [True] * self.num_frames * self.num_copies
        results["quality"] = [0] * self.num_frames * self.num_copies
        return results
