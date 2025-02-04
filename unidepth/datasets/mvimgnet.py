from typing import Any

from unidepth.datasets.sequence_dataset import SequenceDataset

INVALID_SEQUENCES = [
    "1/000121f2-0",
    "15/1600ae56-0",
    "26/000000f3-0",
    "33/1d00e677-0",
    "43/22008925-0",
    "49/000147db-0",
    "51/23002a43-0",
    "51/23000916-0",
    "108/000133ae-0",
    "129/000037f2-0",
    "141/17012545-0",
    "141/1700f3de-0",
    "152/1b00e061-0",
    "154/1d00decb-0",
    "154/1d017c1c-0",
    "154/1d0019a5-0",
    "154/1d00334d-0",
    "154/1d012ed6-0",
    "154/1d016b8a-0",
    "154/1d016cc1-0",
    "154/1d008d5f-0",
    "159/000157f9-0",
    "159/00000b96-0",
    "159/000075c0-0",
    "159/0000445c-0",
    "159/000056a0-0",
    "159/00010c68-0",
    "159/0000573b-0",
    "159/00002698-0",
    "159/00008fca-0",
    "159/00009ef8-0",
    "159/00015f05-0",
    "159/0000c6df-0",
    "159/0000ee59-0",
    "163/290159d2-0",
    "163/29016c7c-0",
    "163/2900239c-0",
    "163/29002f7b-0",
    "163/29014b05-0",
    "163/29000196-0",
    "163/2901750f-0",
    "164/1b0145cf-0",
    "164/1b00eb1d-0",
    "164/1b00c28b-0",
    "164/1b0110d0-0",
    "164/1b00dd20-0",
    "165/2600e15a-0",
    "165/26008444-0",
    "165/260145c5-0",
    "165/26003a0c-0",
    "165/260106ba-0",
    "165/26001548-0",
    "167/2a0092b0-0",
    "167/2a014dbe-0",
    "167/2a003ce6-0",
    "169/1800c645-0",
    "171/2500014d-0",
    "176/1d0021c2-0",
    "176/1d014abf-0",
    "176/1d00e714-0",
    "176/1d0159cb-0",
    "176/1e016629-0",
    "178/000102b8-0",
    "191/23008fdb-0",
    "191/2300187f-0",
    "191/2300ae68-0",
    "191/230076dd-0",
    "191/24007d7e-0",
    "192/000107b5-0",
    "195/1f012359-0",
    "195/1f00f751-0",
    "195/1f011331-0",
    "195/1e00d999-0",
    "196/1c01304e-0",
    "198/1a00e02f-0",
    "198/050084ac-0",
    "198/1a0075fa-0",
    "199/1e001742-0",
    "199/1e00116a-0",
    "199/1e011d00-0",
    "199/1e018040-0",
    "199/1e001107-0",
]


class MVImgNet(SequenceDataset):
    min_depth = 0.005
    max_depth = 10.0
    # weird scale issue, should be 1000, but avg depth is ~10meters...
    depth_scale = 1000.0
    test_split = "train.txt"
    train_split = "train.txt"
    sequences_file = "sequences.json"
    hdf5_paths = ["MVImgNet.hdf5"]
    invalid_sequences = INVALID_SEQUENCES

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
