import numpy as np
import torch
from torch.utils.data import Dataset


class Dummy(Dataset):
    train_split = None
    test_split = None

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.dataset = np.arange(1_000_000)

    def get_single_item(self, idx):
        # results = {}
        # results["cam2w"] = torch.eye(4).unsqueeze(0)
        # results["K"] = torch.eye(3).unsqueeze(0)
        # results["image"] = torch.zeros(1, 3, 1024, 1024).to(torch.uint8)
        # results["depth"] = torch.zeros(1, 1, 1024, 1024).to(torch.float32)
        return {
            "x": {(0, 0): torch.rand(1, 3, 1024, 1024, dtype=torch.float32)},
            "img_metas": {"val": torch.rand(1, 1024, dtype=torch.float32)},
        }

    def __getitem__(self, idx):
        if isinstance(idx, (list, tuple)):
            results = [self.get_single_item(i) for i in idx]
        else:
            results = self.get_single_item(idx)
        return results

    def __len__(self):
        return len(self.dataset)
