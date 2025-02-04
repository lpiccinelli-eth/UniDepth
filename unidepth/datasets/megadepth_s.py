from typing import Any

from unidepth.datasets.sequence_dataset import SequenceDataset


class MegaDepthS(SequenceDataset):
    min_depth = 0.001
    max_depth = 10000.0
    depth_scale = 512.0
    test_split = "train.txt"
    train_split = "train.txt"
    sequences_file = "sequences_filter_clean.json"
    hdf5_paths = ["MegaDepthS.hdf5"]

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
        inplace_fields: list[str] = ["intrinsics", "cam2w"],
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
        results["si"] = [True] * self.num_frames * self.num_copies
        results["dense"] = [False] * self.num_frames * self.num_copies
        results["quality"] = [2] * self.num_frames * self.num_copies
        return results
