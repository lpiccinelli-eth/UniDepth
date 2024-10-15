import torch
import torch.nn as nn


class Dummy(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.name: str = self.__class__.__name__
        self.weight = 1.0

    def forward(self, dummy: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return torch.tensor([0.0] * dummy.shape[0], device=dummy.device)

    @classmethod
    def build(cls, config):
        obj = cls()
        return obj
