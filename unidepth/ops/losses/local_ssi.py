import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import FNS, masked_mean, ssi


class LocalSSI(nn.Module):
    def __init__(
        self,
        weight: float,
        output_fn: str = "sqrt",
        patch_size: tuple[int, int] = (32, 32),
        min_samples: int = 4,
        num_levels: int = 4,
        input_fn: str = "linear",
        eps: float = 1e-5,
    ):
        super(LocalSSI, self).__init__()
        self.name: str = self.__class__.__name__
        self.weight = weight
        self.output_fn = FNS[output_fn]
        self.input_fn = FNS[input_fn]
        self.min_samples = min_samples
        self.eps = eps
        patch_logrange = np.linspace(
            start=np.log2(min(patch_size)),
            stop=np.log2(max(patch_size)),
            endpoint=True,
            num=num_levels + 1,
        )
        self.patch_logrange = [
            (x, y) for x, y in zip(patch_logrange[:-1], patch_logrange[1:])
        ]
        self.rescale_fn = ssi

    @torch.autocast(device_type="cuda", enabled=False, dtype=torch.float32)
    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        mask = mask.bool()
        input = self.input_fn(input.float())
        target = self.input_fn(target.float())
        B, C, H, W = input.shape
        total_errors = []

        for ii, patch_logrange in enumerate(self.patch_logrange):

            log_kernel = (
                np.random.uniform(*patch_logrange)
                if self.training
                else np.mean(patch_logrange)
            )
            kernel_size = int(
                (2**log_kernel) * min(input.shape[-2:])
            )  # always smaller than min_shape
            kernel_size = (kernel_size, kernel_size)
            stride = (int(kernel_size[0] * 0.9), int(kernel_size[1] * 0.9))

            # unfold is always exceeding right/bottom, roll image only negative
            # to have them back in the unfolding window
            max_roll = (
                (W - kernel_size[1]) % stride[1],
                (H - kernel_size[0]) % stride[0],
            )
            roll_x, roll_y = np.random.randint(-max_roll[0], 1), np.random.randint(
                -max_roll[1], 1
            )
            input_fold = torch.roll(input, shifts=(roll_y, roll_x), dims=(2, 3))
            target_fold = torch.roll(target, shifts=(roll_y, roll_x), dims=(2, 3))
            mask_fold = torch.roll(mask.float(), shifts=(roll_y, roll_x), dims=(2, 3))

            # unfold in patches
            input_fold = F.unfold(
                input_fold, kernel_size=kernel_size, stride=stride
            ).permute(
                0, 2, 1
            )  # B N C*H_p*W_p
            target_fold = F.unfold(
                target_fold, kernel_size=kernel_size, stride=stride
            ).permute(0, 2, 1)
            mask_fold = (
                F.unfold(mask_fold, kernel_size=kernel_size, stride=stride)
                .bool()
                .permute(0, 2, 1)
            )

            # calculate error patchwise, then mean over patch, then over image based if sample size is significant
            input_fold, target_fold, _ = self.rescale_fn(
                input_fold, target_fold, mask_fold, dim=[-1]
            )
            error = (input_fold - target_fold).abs()

            # calculate elements more then 95 percentile and lower than 5percentile of error
            valid_patches = mask_fold.sum(dim=-1) >= self.min_samples
            error_mean_patch = masked_mean(error, mask_fold, dim=[-1]).squeeze(-1)
            error_mean_image = self.output_fn(error_mean_patch.clamp(min=self.eps))
            error_mean_image = masked_mean(
                error_mean_image, mask=valid_patches, dim=[-1]
            )
            total_errors.append(error_mean_image.squeeze(-1))

        # global
        input_rescale = input.reshape(B, C, -1)
        target_rescale = target.reshape(B, C, -1)
        mask = mask.reshape(B, 1, -1).clone()
        input, target, mask = self.rescale_fn(
            input_rescale, target_rescale, mask, dim=[-1]
        )
        error = (input - target).abs().squeeze(1)

        mask = mask.squeeze(1)
        error_mean_image = masked_mean(error, mask, dim=[-1]).squeeze(-1)
        error_mean_image = self.output_fn(error_mean_image.clamp(min=self.eps))

        total_errors.append(error_mean_image)

        errors = torch.stack(total_errors).mean(dim=0)
        return errors

    @classmethod
    def build(cls, config):
        obj = cls(
            weight=config["weight"],
            patch_size=config["patch_size"],
            output_fn=config["output_fn"],
            min_samples=config["min_samples"],
            num_levels=config["num_levels"],
            input_fn=config["input_fn"],
        )
        return obj
