from typing import Any

from unidepth.datasets.sequence_dataset import SequenceDataset


class VKITTI(SequenceDataset):
    min_depth = 0.01
    max_depth = 255.0
    depth_scale = 256.0
    test_split = "training.txt"
    train_split = "training.txt"
    sequences_file = "sequences.json"
    hdf5_paths = ["VKITTI2.hdf5"]

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
            num_frames=num_frames,
            decode_fields=decode_fields,
            inplace_fields=inplace_fields,
            **kwargs,
        )

    def pre_pipeline(self, results):
        results = super().pre_pipeline(results)
        results["dense"] = [True] * self.num_frames * self.num_copies
        results["synthetic"] = [True] * self.num_frames * self.num_copies
        results["quality"] = [0] * self.num_frames * self.num_copies
        return results
