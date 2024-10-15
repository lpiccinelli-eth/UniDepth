import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .utils import FNS, masked_mean


class SelfDistill(nn.Module):
    def __init__(self, weight: float, output_fn: str = "sqrt", eps: float = 1e-5):
        super().__init__()
        self.name: str = self.__class__.__name__
        self.weight: float = weight
        self.dims = (-2, -1)
        self.output_fn = FNS[output_fn]
        self.eps: float = eps

    @torch.autocast(device_type="cuda", enabled=False, dtype=torch.float32)
    def forward(
        self,
        input: torch.Tensor,
        intrinsics: torch.Tensor,
        mask: torch.Tensor,
        flips: torch.Tensor,
        downsample_ratio=14,
    ) -> torch.Tensor:
        chunks = input.shape[0] // 2
        mask = F.interpolate(mask.float(), size=input.shape[-2:], mode="nearest")

        iters = zip(
            input.chunk(chunks),
            mask.chunk(chunks),
            intrinsics.chunk(chunks),
            flips.chunk(chunks),
        )
        inputs0, inputs1, masks = [], [], []
        for i, (pair_input, pair_mask, pair_cam, pair_flip) in enumerate(iters):

            mask0, mask1 = pair_mask
            input0, input1 = pair_input
            cam0, cam1 = pair_cam
            flip0, flip1 = pair_flip

            fx_0 = cam0[0, 0] / downsample_ratio
            fx_1 = cam1[0, 0] / downsample_ratio
            cx_0 = cam0[0, 2] / downsample_ratio
            cx_1 = cam1[0, 2] / downsample_ratio
            cy_0 = cam0[1, 2] / downsample_ratio
            cy_1 = cam1[1, 2] / downsample_ratio

            # flip image
            if flip0 ^ flip1:
                input0 = torch.flip(input0, dims=(2,))
                mask0 = torch.flip(mask0, dims=(2,))
                cx_0 = input0.shape[-1] - cx_0

            # calc zoom
            zoom_x = float(fx_1 / fx_0)

            # apply zoom
            input0 = F.interpolate(
                input0.unsqueeze(0), scale_factor=zoom_x, mode="bilinear"
            ).squeeze(0)
            mask0 = F.interpolate(
                mask0.unsqueeze(0), scale_factor=zoom_x, mode="nearest"
            ).squeeze(0)

            # calc translation
            change_left = int(cx_1 - (cx_0 - 0.5) * zoom_x - 0.5)
            change_top = int(cy_1 - (cy_0 - 0.5) * zoom_x - 0.5)
            change_right = input1.shape[-1] - change_left - input0.shape[-1]
            change_bottom = input1.shape[-2] - change_top - input0.shape[-2]

            # apply translation
            pad_left = max(0, change_left)
            pad_right = max(0, change_right)
            pad_top = max(0, change_top)
            pad_bottom = max(0, change_bottom)

            crop_left = max(0, -change_left)
            crop_right = max(0, -change_right)
            crop_top = max(0, -change_top)
            crop_bottom = max(0, -change_bottom)

            input0 = F.pad(
                input0,
                (pad_left, pad_right, pad_top, pad_bottom),
                mode="constant",
                value=0,
            )
            mask0 = F.pad(
                mask0,
                (pad_left, pad_right, pad_top, pad_bottom),
                mode="constant",
                value=0,
            )
            input0 = input0[
                :,
                crop_top : input0.shape[-2] - crop_bottom,
                crop_left : input0.shape[-1] - crop_right,
            ]
            mask0 = mask0[
                :,
                crop_top : mask0.shape[-2] - crop_bottom,
                crop_left : mask0.shape[-1] - crop_right,
            ]

            mask = torch.logical_and(mask0, mask1)

            inputs0.append(input0)
            inputs1.append(input1)
            masks.append(mask)

        inputs0 = torch.stack(inputs0, dim=0)
        inputs1 = torch.stack(inputs1, dim=0)
        masks = torch.stack(masks, dim=0)
        loss1 = self.loss(inputs0, inputs1.detach(), masks)
        loss2 = self.loss(inputs1, inputs0.detach(), masks)
        return torch.cat([loss1, loss2], dim=0)

    def loss(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        loss = masked_mean(
            (input - target).square().mean(dim=1), mask=mask, dim=[-2, -1]
        )
        return self.output_fn(loss + self.eps)

    @classmethod
    def build(cls, config):
        obj = cls(
            weight=config["weight"],
            output_fn=config["output_fn"],
        )
        return obj


class TeacherDistill(nn.Module):
    def __init__(
        self,
        weight: float,
        output_fn: str = "sqrt",
        cross: bool = False,
        eps: float = 1e-5,
    ):
        super().__init__()
        assert output_fn in FNS
        self.name: str = self.__class__.__name__
        self.weight: float = weight
        self.dims = (-2, -1)
        self.output_fn = FNS[output_fn]
        self.eps: float = eps
        self.cross = cross
        self.threshold = 0.05
        self.head_dim = 64  # hardcoded for vit

    @torch.autocast(device_type="cuda", enabled=False, dtype=torch.float32)
    def forward(
        self,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor,
        student_tokens: torch.Tensor,
        teacher_tokens: torch.Tensor,
        mask: torch.Tensor,
        # metas: List[Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        B = student_features.shape[0]
        device = student_features.device
        chunks = student_features.shape[0] // 2

        mask = (
            F.interpolate(
                mask.float() + 1e-3, size=student_features.shape[-2:], mode="nearest"
            )
            > 0.5
        )

        # chunk features as self.head_dim
        student_features = rearrange(
            student_features, "b (n c) h w -> b c h w n", c=self.head_dim
        )
        teacher_features = rearrange(
            teacher_features, "b (n c) h w -> b c h w n", c=self.head_dim
        )
        student_tokens = rearrange(
            student_tokens, "b t (n c) -> b t c n", c=self.head_dim
        )
        teacher_tokens = rearrange(
            teacher_tokens, "b t (n c) -> b t c n", c=self.head_dim
        )

        distance = (
            (student_features - teacher_features)
            .square()
            .sum(dim=1, keepdim=True)
            .sqrt()
            .mean(dim=-1)
        )
        loss_features = masked_mean(distance, mask=mask, dim=[-2, -1])
        loss_features = self.output_fn(loss_features.clamp(min=self.eps)).squeeze(
            1, 2, 3
        )

        distance = (
            (student_tokens - teacher_tokens).square().sum(dim=-2).sqrt().mean(dim=-1)
        )
        loss_tokens = self.output_fn(distance.clamp(min=self.eps)).squeeze(1)

        return loss_features + 0.01 * loss_tokens

    @classmethod
    def build(cls, config):
        obj = cls(
            weight=config["weight"],
            output_fn=config["output_fn"],
            cross=config["cross"],
        )
        return obj
