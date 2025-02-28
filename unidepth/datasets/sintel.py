from typing import Any

from unidepth.datasets.sequence_dataset import SequenceDataset


class Sintel(SequenceDataset):
    min_depth = 0.001
    max_depth = 1000.0
    depth_scale = 1000.0
    test_split = "training.txt"
    train_split = "training.txt"
    sequences_file = "sequences.json"
    hdf5_paths = ["Sintel.hdf5"]

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
        results["dense"] = [True] * self.num_frames
        results["synthetic"] = [True] * self.num_frames
        return results
