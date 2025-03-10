import importlib
import warnings
from copy import deepcopy
from math import ceil

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from einops import rearrange
from huggingface_hub import PyTorchModelHubMixin

from unidepth.models.unidepthv2.decoder_old import Decoder
from unidepth.utils.constants import (IMAGENET_DATASET_MEAN,
                                      IMAGENET_DATASET_STD)
from unidepth.utils.distributed import is_main_process
from unidepth.utils.geometric import (generate_rays,
                                      spherical_zbuffer_to_euclidean)
from unidepth.utils.misc import (first_stack, last_stack, max_stack,
                                 mean_stack, softmax_stack)

STACKING_FNS = {
    "max": max_stack,
    "mean": mean_stack,
    "first": first_stack,
    "last": last_stack,
    "softmax": softmax_stack,
}
RESOLUTION_LEVELS = 10


# inference helpers
def _check_ratio(image_ratio, ratio_bounds):
    ratio_bounds = sorted(ratio_bounds)
    if ratio_bounds is not None and (
        image_ratio < ratio_bounds[0] or image_ratio > ratio_bounds[1]
    ):
        warnings.warn(
            f"Input image ratio ({image_ratio:.3f}) is out of training "
            f"distribution: {ratio_bounds}. This may lead to unexpected results. "
            f"Consider resizing/padding the image to match the training distribution."
        )


def _check_resolution(shape_constraints, resolution_level):
    if resolution_level is None:
        warnings.warn(
            "Resolution level is not set. Using max resolution. "
            "You can tradeoff resolution for speed by setting a number in [0,10]. "
            "This can be achieved by setting model's `resolution_level` attribute."
        )
        resolution_level = RESOLUTION_LEVELS
    pixel_bounds = sorted(shape_constraints["pixels_bounds_ori"])
    pixel_range = pixel_bounds[-1] - pixel_bounds[0]
    clipped_resolution_level = min(max(resolution_level, 0), RESOLUTION_LEVELS)
    if clipped_resolution_level != resolution_level:
        warnings.warn(
            f"Resolution level {resolution_level} is out of bounds ([0,{RESOLUTION_LEVELS}]). "
            f"Clipping to {clipped_resolution_level}."
        )
    shape_constraints["pixels_bounds"] = [
        pixel_bounds[0]
        + ceil(pixel_range * clipped_resolution_level / RESOLUTION_LEVELS),
        pixel_bounds[0]
        + ceil(pixel_range * clipped_resolution_level / RESOLUTION_LEVELS),
    ]
    return shape_constraints


def _get_closes_num_pixels(image_shape, pixels_bounds):
    h, w = image_shape
    num_pixels = h * w
    pixels_bounds = sorted(pixels_bounds)
    num_pixels = max(min(num_pixels, pixels_bounds[1]), pixels_bounds[0])
    return num_pixels


def _shapes(image_shape, shape_constraints):
    h, w = image_shape
    image_ratio = w / h
    _check_ratio(image_ratio, shape_constraints["ratio_bounds"])
    num_pixels = _get_closes_num_pixels(
        (h / shape_constraints["patch_size"], w / shape_constraints["patch_size"]),
        shape_constraints["pixels_bounds"],
    )
    h = ceil((num_pixels / image_ratio) ** 0.5 - 0.5)
    w = ceil(h * image_ratio - 0.5)
    ratio = h / image_shape[0] * shape_constraints["patch_size"]
    return (
        h * shape_constraints["patch_size"],
        w * shape_constraints["patch_size"],
    ), ratio


def _preprocess(rgbs, intrinsics, shapes, ratio):
    rgbs = F.interpolate(rgbs, size=shapes, mode="bilinear", antialias=True)
    if intrinsics is not None:
        intrinsics = intrinsics.clone()
        intrinsics[:, 0, 0] = intrinsics[:, 0, 0] * ratio
        intrinsics[:, 1, 1] = intrinsics[:, 1, 1] * ratio
        intrinsics[:, 0, 2] = intrinsics[:, 0, 2] * ratio
        intrinsics[:, 1, 2] = intrinsics[:, 1, 2] * ratio
        return rgbs, intrinsics
    return rgbs, None


def _postprocess(outs, ratio, original_shapes, mode="nearest-exact"):
    outs["depth"] = F.interpolate(outs["depth"], size=original_shapes, mode=mode)
    outs["confidence"] = F.interpolate(
        outs["confidence"], size=original_shapes, mode="bilinear", antialias=True
    )
    outs["K"][:, 0, 0] = outs["K"][:, 0, 0] / ratio
    outs["K"][:, 1, 1] = outs["K"][:, 1, 1] / ratio
    outs["K"][:, 0, 2] = outs["K"][:, 0, 2] / ratio
    outs["K"][:, 1, 2] = outs["K"][:, 1, 2] / ratio
    return outs


class UniDepthV2old(
    nn.Module,
    PyTorchModelHubMixin,
    library_name="UniDepth",
    repo_url="https://github.com/lpiccinelli-eth/UniDepth",
    tags=["monocular-metric-depth-estimation"],
):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super().__init__()
        self.build(config)

    def forward(self, inputs, image_metas):
        H, W = inputs["depth"].shape[-2:]

        if "K" in inputs:
            rays, angles = generate_rays(inputs["K"], (H, W))
            inputs["rays"] = rays
            inputs["angles"] = angles

        features, tokens = self.pixel_encoder(inputs[f"image"])

        cls_tokens = [x.contiguous() for x in tokens]
        features = [
            self.stacking_fn(features[i:j]).contiguous()
            for i, j in self.slices_encoder_range
        ]
        tokens = [
            self.stacking_fn(tokens[i:j]).contiguous()
            for i, j in self.slices_encoder_range
        ]
        global_tokens = [cls_tokens[i] for i in [-2, -1]]
        camera_tokens = [cls_tokens[i] for i in [-3, -2, -1]] + [tokens[-2]]

        inputs["features"] = features
        inputs["tokens"] = tokens
        inputs["global_tokens"] = global_tokens
        inputs["camera_tokens"] = camera_tokens

        outs = self.pixel_decoder(inputs, image_metas)

        angles = rearrange(
            generate_rays(outs["K"], (H, W), noisy=False)[-1],
            "b (h w) c -> b c h w",
            h=H,
            w=W,
        )
        predictions = F.interpolate(
            outs["depth"],
            size=(H, W),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        confidence = F.interpolate(
            outs["confidence"],
            size=(H, W),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        predictions_3d = torch.cat((angles, predictions), dim=1)
        predictions_3d = spherical_zbuffer_to_euclidean(
            predictions_3d.permute(0, 2, 3, 1)
        ).permute(0, 3, 1, 2)

        outputs = {
            "K": outs["K"],
            "depth": predictions,
            "confidence": confidence,
            "points": predictions_3d,
            "depth_features": outs["depth_features"],
        }
        return outputs

    @torch.no_grad()
    def infer(self, rgbs: torch.Tensor, intrinsics=None):
        shape_constraints = self.shape_constraints
        if rgbs.ndim == 3:
            rgbs = rgbs.unsqueeze(0)
        if intrinsics is not None and intrinsics.ndim == 2:
            intrinsics = intrinsics.unsqueeze(0)
        B, _, H, W = rgbs.shape

        rgbs = rgbs.to(self.device)
        if intrinsics is not None:
            intrinsics = intrinsics.to(self.device)

        # process image and intrinsiscs (if any) to match network input (slow?)
        if rgbs.max() > 5 or rgbs.dtype == torch.uint8:
            rgbs = rgbs.to(torch.float32).div(255)
        if rgbs.min() >= 0.0 and rgbs.max() <= 1.0:
            rgbs = TF.normalize(
                rgbs,
                mean=IMAGENET_DATASET_MEAN,
                std=IMAGENET_DATASET_STD,
            )

        # check resolution constraints: tradeoff resolution and speed
        shape_constraints = _check_resolution(shape_constraints, self.resolution_level)

        # get image shape
        (h, w), ratio = _shapes((H, W), shape_constraints)
        rgbs, gt_intrinsics = _preprocess(
            rgbs,
            intrinsics,
            (h, w),
            ratio,
        )

        # run encoder
        features, tokens = self.pixel_encoder(rgbs)

        cls_tokens = [x.contiguous() for x in tokens]
        features = [
            self.stacking_fn(features[i:j]).contiguous()
            for i, j in self.slices_encoder_range
        ]
        tokens = [
            self.stacking_fn(tokens[i:j]).contiguous()
            for i, j in self.slices_encoder_range
        ]
        global_tokens = [cls_tokens[i] for i in [-2, -1]]
        camera_tokens = [cls_tokens[i] for i in [-3, -2, -1]] + [tokens[-2]]

        # get data fro decoder and adapt to given camera
        inputs = {}
        inputs["features"] = features
        inputs["tokens"] = tokens
        inputs["global_tokens"] = global_tokens
        inputs["camera_tokens"] = camera_tokens
        inputs["image"] = rgbs
        if gt_intrinsics is not None:
            rays, angles = generate_rays(gt_intrinsics, (h, w))
            inputs["rays"] = rays
            inputs["angles"] = angles
            inputs["K"] = gt_intrinsics

        outs = self.pixel_decoder(inputs, {})
        # undo the reshaping and get original image size (slow)
        outs = _postprocess(outs, ratio, (H, W), mode=self.interpolation_mode)
        pred_intrinsics = outs["K"]
        depth = outs["depth"]
        confidence = outs["confidence"]

        # final 3D points backprojection
        intrinsics = intrinsics if intrinsics is not None else pred_intrinsics
        angles = generate_rays(intrinsics, (H, W))[-1]
        angles = rearrange(angles, "b (h w) c -> b c h w", h=H, w=W)
        points_3d = torch.cat((angles, depth), dim=1)
        points_3d = spherical_zbuffer_to_euclidean(
            points_3d.permute(0, 2, 3, 1)
        ).permute(0, 3, 1, 2)

        outputs = {
            "intrinsics": pred_intrinsics,
            "points": points_3d,
            "depth": depth,
            "confidence": confidence,
        }
        return outputs

    def load_pretrained(self, model_file):
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        dict_model = torch.load(model_file, map_location=device)
        if "model" in dict_model:
            dict_model = dict_model["model"]
        dict_model = deepcopy(
            {k.replace("module.", ""): v for k, v in dict_model.items()}
        )

        info = self.load_state_dict(dict_model, strict=False)
        if is_main_process():
            print(
                f"Loaded from {model_file} for {self.__class__.__name__} results in:",
                info,
            )

    @property
    def device(self):
        return next(self.parameters()).device

    def build(self, config):
        mod = importlib.import_module("unidepth.models.encoder")
        pixel_encoder_factory = getattr(mod, config["model"]["pixel_encoder"]["name"])
        pixel_encoder_config = {
            **config["training"],
            **config["model"]["pixel_encoder"],
            **config["data"],
        }
        pixel_encoder = pixel_encoder_factory(pixel_encoder_config)

        config["model"]["pixel_encoder"]["patch_size"] = (
            14 if "dino" in config["model"]["pixel_encoder"]["name"] else 16
        )
        pixel_encoder_embed_dims = (
            pixel_encoder.embed_dims
            if hasattr(pixel_encoder, "embed_dims")
            else [getattr(pixel_encoder, "embed_dim") * 2**i for i in range(4)]
        )
        config["model"]["pixel_encoder"]["embed_dim"] = getattr(
            pixel_encoder, "embed_dim"
        )
        config["model"]["pixel_encoder"]["embed_dims"] = pixel_encoder_embed_dims
        config["model"]["pixel_encoder"]["depths"] = pixel_encoder.depths

        pixel_decoder = Decoder(config)

        self.pixel_encoder = pixel_encoder
        self.pixel_decoder = pixel_decoder
        stacking_fn = config["model"]["pixel_encoder"]["stacking_fn"]
        assert (
            stacking_fn in STACKING_FNS
        ), f"Stacking function {stacking_fn} not found in {STACKING_FNS.keys()}"
        self.stacking_fn = STACKING_FNS[stacking_fn]

        self.slices_encoder_range = list(
            zip([0, *pixel_encoder.depths[:-1]], pixel_encoder.depths)
        )
        self.shape_constraints = config["data"]["shape_constraints"]
        self.shape_constraints["pixels_bounds_ori"] = self.shape_constraints.get(
            "pixels_bounds", [1400, 2400]
        )
        self.interpolation_mode = "bilinear"
        self.eps = 1e-6
        self.resolution_level = None

    def build_losses(self, config):
        self.losses = {}
        for loss_name, loss_config in config["training"]["losses"].items():
            mod = importlib.import_module("unidepth.ops.losses")
            loss_factory = getattr(mod, loss_config["name"])
            self.losses[loss_name] = loss_factory.build(loss_config)
