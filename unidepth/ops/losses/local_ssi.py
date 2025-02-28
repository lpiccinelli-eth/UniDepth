import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from unidepth.utils.geometric import erode

from .utils import FNS, ind2sub, masked_mean, masked_quantile, ssi


def sample_strong_edges(edges_img, quantile=0.95, reshape=8):
    # flat
    edges_img = F.interpolate(
        edges_img, scale_factor=1 / reshape, mode="bilinear", align_corners=False
    )
    edges_img_flat = edges_img.flatten(1)

    # Find strong edges
    edges_mask = edges_img_flat > torch.quantile(
        edges_img_flat, quantile, dim=-1, keepdim=True
    )
    num_samples = edges_mask.sum(dim=-1)
    if (num_samples < 10).any():
        # sample random edges where num_samples < 2
        random = torch.rand_like(edges_img_flat[num_samples < 10, :]) > quantile
        edges_mask[num_samples < 10, :] = torch.logical_or(
            edges_mask[num_samples < 10, :], random
        )
        num_samples = edges_mask.sum(dim=-1)

    min_samples = num_samples.min()

    # Compute the coordinates of the strong edges as B, N, 2
    edges_coords = torch.stack(
        [torch.nonzero(x, as_tuple=False)[:min_samples].squeeze() for x in edges_mask]
    )
    edges_coords = (
        torch.stack(ind2sub(edges_coords, edges_img.shape[-1]), dim=-1) * reshape
    )
    return edges_coords


@torch.jit.script
def extract_patches(tensor, sample_coords, patch_size: tuple[int, int] = (32, 32)):
    N, _, H, W = tensor.shape
    device = tensor.device
    dtype = tensor.dtype
    patch_width, patch_height = patch_size
    pad_width = patch_width // 2
    pad_height = patch_height // 2

    # Pad the RGB images for both sheep
    tensor_padded = F.pad(
        tensor,
        (pad_width, pad_width, pad_height, pad_height),
        mode="constant",
        value=0.0,
    )

    # Adjust edge coordinates to account for padding
    sample_coords_padded = sample_coords + torch.tensor(
        [pad_height, pad_width], dtype=dtype, device=device
    ).reshape(1, 1, 2)

    # Calculate the indices for gather operation
    x_centers = sample_coords_padded[:, :, 1].int()
    y_centers = sample_coords_padded[:, :, 0].int()

    all_patches = []
    for tensor_i, x_centers_i, y_centers_i in zip(tensor_padded, x_centers, y_centers):
        patches = []
        for x_center, y_center in zip(x_centers_i, y_centers_i):
            y_start, y_end = y_center - pad_height, y_center + pad_height + 1
            x_start, x_end = x_center - pad_width, x_center + pad_width + 1
            patches.append(tensor_i[..., y_start:y_end, x_start:x_end])
        all_patches.append(torch.stack(patches, dim=0))

    return torch.stack(all_patches, dim=0).reshape(N, -1, patch_height * patch_width)


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


class EdgeGuidedLocalSSI(nn.Module):
    def __init__(
        self,
        weight: float,
        output_fn: str = "sqrt",
        min_samples: int = 4,
        input_fn: str = "linear",
        use_global: bool = True,
        eps: float = 1e-5,
    ):
        super(EdgeGuidedLocalSSI, self).__init__()
        self.name: str = self.__class__.__name__
        self.weight = weight
        self.output_fn = FNS[output_fn]
        self.input_fn = FNS[input_fn]
        self.min_samples = min_samples
        self.eps = eps
        self.use_global = use_global
        self.rescale_fn = ssi

        delta_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], requires_grad=False
        )
        delta_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], requires_grad=False
        )
        self.delta_x = delta_x.reshape(1, 1, 3, 3)
        self.delta_y = delta_y.reshape(1, 1, 3, 3)

        try:
            from unidepth.ops.extract_patches import RandomPatchExtractor

            self.random_patch_extractor = RandomPatchExtractor()
        except Exception as e:
            self.random_patch_extractor = extract_patches
            print(
                "EdgeGuidedLocalSSI reverts to a non cuda-optimized operation, "
                "you will experince large slowdown, "
                "please install it: ",
                "`cd ./unidepth/ops/extract_patches && bash compile.sh`",
            )

    def get_edge(self, image, mask):
        channels = image.shape[1]
        device = image.device
        delta_x = self.delta_x.to(device).repeat(channels, 1, 1, 1)
        delta_y = self.delta_y.to(device).repeat(channels, 1, 1, 1)
        image_Gx = F.conv2d(image, delta_x, groups=channels, padding="same") / 8
        image_Gy = F.conv2d(image, delta_y, groups=channels, padding="same") / 8
        image_Gx = (
            image_Gx.square().mean(dim=1, keepdim=True).sqrt()
        )  # RMSE over color dim
        image_Gy = image_Gy.square().mean(dim=1, keepdim=True).sqrt()
        edges = torch.sqrt(image_Gx**2 + image_Gy**2)
        edges[:, :, :3, :] = 0
        edges[:, :, -3:, :] = 0
        edges[:, :, :, :3] = 0
        edges[:, :, :, -3:] = 0
        edges[~mask.bool()] = 0
        return edges

    def compute_sample_patch_error(
        self, input, target, mask, sampling_coords, kernel_size, image_size
    ):
        B, C, H, W = input.shape
        patch_size = kernel_size[0] * kernel_size[1]
        input = self.random_patch_extractor(
            input, sampling_coords, kernel_size
        ).reshape(B, -1, patch_size)
        target = self.random_patch_extractor(
            target, sampling_coords, kernel_size
        ).reshape(B, -1, patch_size)
        mask = (
            self.random_patch_extractor(mask.float(), sampling_coords, kernel_size)
            .bool()
            .reshape(B, -1, patch_size)
        )
        input, target, mask = self.rescale_fn(input, target, mask, dim=[-1])
        error = (input - target).abs().clamp(min=self.eps)

        valid_patches = mask.sum(dim=-1) >= self.min_samples
        error_mean_patch = masked_mean(error, mask, dim=[-1]).squeeze(-1)
        error_mean_image = self.output_fn(error_mean_patch.clamp(min=self.eps))
        error_mean_image = masked_mean(error_mean_image, mask=valid_patches, dim=[-1])
        return error_mean_image

    def compute_image_error(self, input, target, mask, image_size):
        H, W = image_size
        input = input.reshape(-1, 1, H * W)
        target = target.reshape(-1, 1, H * W)
        mask = mask.reshape(-1, 1, H * W)
        input, target, mask = self.rescale_fn(input, target, mask, dim=[-1])
        error = (input - target).abs().clamp(min=self.eps)

        error_mean_image = masked_mean(error, mask, dim=[-1]).squeeze(-1)
        error_mean_image = self.output_fn(error_mean_image.clamp(min=self.eps))
        return error_mean_image

    @torch.autocast(device_type="cuda", enabled=False, dtype=torch.float32)
    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        image: torch.Tensor | None = None,
        validity_mask: torch.Tensor | None = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:

        mask = mask.bool()
        input = self.input_fn(input.float())
        target = self.input_fn(target.float())
        B, _, H, W = input.shape
        total_errors = []

        # remove border and black border
        if validity_mask is not None:
            validity_mask = erode(validity_mask.float(), kernel_size=3)

        edges = self.get_edge(image, validity_mask)
        # quantile was 0.95?
        edges_coords = sample_strong_edges(edges, quantile=0.9, reshape=14)
        log_kernel = np.random.uniform(0.04, 0.08) if self.training else 0.05
        kernel_size = int(
            log_kernel * min(input.shape[-2:])
        )  # always smaller than min_shape
        kernel_size = kernel_size + int(kernel_size % 2 == 0)  # odd num
        kernel_size = (kernel_size, kernel_size)
        error_mean_image = self.compute_sample_patch_error(
            input, target, mask, edges_coords, kernel_size, (H, W)
        )
        total_errors.append(error_mean_image.squeeze(-1))

        if self.use_global:
            error_mean_image = self.compute_image_error(input, target, mask, (H, W))
            total_errors.append(error_mean_image.squeeze(-1))

        errors = torch.stack(total_errors).mean(dim=0)
        return errors

    @classmethod
    def build(cls, config):
        obj = cls(
            weight=config["weight"],
            output_fn=config["output_fn"],
            input_fn=config["input_fn"],
            use_global=config["use_global"],
            min_samples=config.get("min_samples", 6),
        )
        return obj
