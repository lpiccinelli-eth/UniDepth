"""
Author: Luigi Piccinelli
Licensed under the CC-BY NC 4.0 license (http://creativecommons.org/licenses/by-nc/4.0/)
"""

import importlib
from copy import deepcopy
from math import ceil

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from einops import rearrange
from huggingface_hub import PyTorchModelHubMixin

from unidepth.models.unidepthv1.decoder import Decoder
from unidepth.utils.constants import (IMAGENET_DATASET_MEAN,
                                      IMAGENET_DATASET_STD)
from unidepth.utils.distributed import is_main_process
from unidepth.utils.geometric import (generate_rays,
                                      spherical_zbuffer_to_euclidean)
from unidepth.utils.misc import (get_params, match_gt, match_intrinsics,
                                 profile_method)

VERBOSE = False


# inference helpers
def _paddings(image_shape, network_shape):
    cur_h, cur_w = image_shape
    h, w = network_shape
    pad_top, pad_bottom = (h - cur_h) // 2, h - cur_h - (h - cur_h) // 2
    pad_left, pad_right = (w - cur_w) // 2, w - cur_w - (w - cur_w) // 2
    return pad_left, pad_right, pad_top, pad_bottom


def _shapes(image_shape, network_shape):
    h, w = image_shape
    input_ratio = w / h
    output_ratio = network_shape[1] / network_shape[0]
    if output_ratio > input_ratio:
        ratio = network_shape[0] / h
    elif output_ratio <= input_ratio:
        ratio = network_shape[1] / w
    return (ceil(h * ratio - 0.5), ceil(w * ratio - 0.5)), ratio


def _preprocess(rgbs, intrinsics, shapes, pads, ratio, output_shapes):
    (pad_left, pad_right, pad_top, pad_bottom) = pads
    rgbs = F.interpolate(
        rgbs, size=shapes, mode="bilinear", align_corners=False, antialias=True
    )
    rgbs = F.pad(rgbs, (pad_left, pad_right, pad_top, pad_bottom), mode="constant")
    if intrinsics is not None:
        intrinsics = intrinsics.clone()
        intrinsics[:, 0, 0] = intrinsics[:, 0, 0] * ratio
        intrinsics[:, 1, 1] = intrinsics[:, 1, 1] * ratio
        intrinsics[:, 0, 2] = intrinsics[:, 0, 2] * ratio + pad_left
        intrinsics[:, 1, 2] = intrinsics[:, 1, 2] * ratio + pad_top
        return rgbs, intrinsics
    return rgbs, None


def _postprocess(predictions, intrinsics, shapes, pads, ratio, original_shapes):
    (pad_left, pad_right, pad_top, pad_bottom) = pads
    # pred mean, trim paddings, and upsample to input dim
    predictions = sum(
        [
            F.interpolate(
                x.clone(),
                size=shapes,
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
            for x in predictions
        ]
    ) / len(predictions)
    predictions = predictions[
        ..., pad_top : shapes[0] - pad_bottom, pad_left : shapes[1] - pad_right
    ]
    predictions = F.interpolate(
        predictions,
        size=original_shapes,
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    intrinsics[:, 0, 0] = intrinsics[:, 0, 0] / ratio
    intrinsics[:, 1, 1] = intrinsics[:, 1, 1] / ratio
    intrinsics[:, 0, 2] = (intrinsics[:, 0, 2] - pad_left) / ratio
    intrinsics[:, 1, 2] = (intrinsics[:, 1, 2] - pad_top) / ratio
    return predictions, intrinsics


class UniDepthV1(
    nn.Module,
    PyTorchModelHubMixin,
    library_name="UniDepth",
    repo_url="https://github.com/lpiccinelli-eth/UniDepth",
    tags=["monocular-metric-depth-estimation"],
):
    def __init__(
        self,
        config,
        eps: float = 1e-6,
        **kwargs,
    ):
        super().__init__()
        self.build(config)
        self.build_losses(config)
        self.eps = eps

    @profile_method(verbose=VERBOSE)
    def forward_train(self, inputs, image_metas):
        inputs, outputs = self.encode_decode(inputs, image_metas)
        losses = self.compute_losses(outputs, inputs, image_metas)
        return outputs, losses

    @profile_method(verbose=VERBOSE)
    def forward_test(self, inputs, image_metas):
        inputs, outputs = self.encode_decode(inputs, image_metas)
        depth_gt = inputs["depth"]
        test_outputs = {}
        test_outputs["depth"] = match_gt(
            outputs["depth"], depth_gt, padding1=inputs["paddings"], padding2=None
        )
        test_outputs["points"] = match_gt(
            outputs["points"], depth_gt, padding1=inputs["paddings"], padding2=None
        )
        test_outputs["confidence"] = match_gt(
            outputs["confidence"], depth_gt, padding1=inputs["paddings"], padding2=None
        )
        test_outputs["rays"] = match_gt(
            outputs["rays"], depth_gt, padding1=inputs["paddings"], padding2=None
        )
        test_outputs["rays"] = outputs["rays"] / torch.norm(
            outputs["rays"], dim=1, keepdim=True
        ).clip(min=1e-5)
        test_outputs["intrinsics"] = match_intrinsics(
            outputs["intrinsics"],
            inputs["image"],
            depth_gt,
            padding1=inputs["paddings"],
            padding2=None,
        )
        return test_outputs

    def forward(self, inputs, image_metas):
        if self.training:
            return self.forward_train(inputs, image_metas)
        else:
            return self.forward_test(inputs, image_metas)

    def encode_decode(self, inputs, image_metas):
        rgbs = inputs["image"]
        B, _, H, W = rgbs.shape
        cameras = inputs["camera"]

        # shortcut eval should avoid errors
        if len(image_metas) and "paddings" in image_metas[0]:
            inputs["paddings"] = torch.tensor(
                [image_meta["paddings"] for image_meta in image_metas],
                device=self.device,
            )[
                ..., [0, 2, 1, 3]
            ]  # lrtb
            inputs["depth_paddings"] = torch.tensor(
                [image_meta["depth_paddings"] for image_meta in image_metas],
                device=self.device,
            )
            if (
                self.training
            ):  # at inference we do not have image paddings on top of depth ones (we have not "crop" on gt in ContextCrop)
                inputs["depth_paddings"] = inputs["depth_paddings"] + inputs["paddings"]

        # Get camera rays for supervision, all in unit sphere
        if inputs.get("camera", None) is not None:
            inputs["rays"] = rearrange(
                inputs["camera"].get_rays(shapes=(B, H, W)), "b c h w -> b (h w) c"
            )

        # Encode
        encoder_outputs, cls_tokens = self.pixel_encoder(rgbs)
        if "dino" in self.pixel_encoder.__class__.__name__.lower():
            encoder_outputs = [
                (x + y.unsqueeze(1)).contiguous()
                for x, y in zip(encoder_outputs, cls_tokens)
            ]
        inputs["encoder_outputs"] = encoder_outputs
        inputs["cls_tokens"] = cls_tokens

        # Decode
        pred_intrinsics, predictions, depth_features = self.pixel_decoder(inputs, {})
        predictions = sum(
            [
                F.interpolate(
                    x.clone(),
                    size=(H, W),
                    mode="bilinear",
                    align_corners=False,
                    antialias=True,
                )
                for x in predictions
            ]
        ) / len(predictions)

        # Final 3D points backprojection
        pred_rays, pred_angles = generate_rays(pred_intrinsics, (H, W), noisy=False)

        # You may want to use inputs["angles"] if available?
        pred_angles = rearrange(pred_angles, "b (h w) c -> b c h w", h=H, w=W)

        points_3d = torch.cat((pred_angles, predictions), dim=1)
        points_3d = spherical_zbuffer_to_euclidean(
            points_3d.permute(0, 2, 3, 1)
        ).permute(0, 3, 1, 2)

        # Output data, use for loss computation
        outputs = {
            "angles": pred_angles,
            "rays": pred_rays,
            "intrinsics": pred_intrinsics,
            "points": points_3d,
            "depth": predictions[:, -1:],
            "cond_features": depth_features,
        }
        self.pixel_decoder.test_fixed_camera = False
        outputs["rays"] = rearrange(outputs["rays"], "b (h w) c -> b c h w", h=H, w=W)
        if "rays" in inputs:
            inputs["rays"] = rearrange(inputs["rays"], "b (h w) c -> b c h w", h=H, w=W)
        return inputs, outputs

    def compute_losses(self, outputs, inputs, image_metas):
        B, _, H, W = inputs["image"].shape
        losses = {"opt": {}, "stat": {}}
        if (
            not self.training
        ):  # only compute losses during training, avoid issues for mismatch size of pred and GT
            return losses
        losses_to_be_computed = list(self.losses.keys())

        # depth loss
        si = torch.tensor(
            [x.get("si", False) for x in image_metas], device=self.device
        ).reshape(B)
        loss = self.losses["depth"]
        depth_losses = loss(
            outputs["depth"],
            target=inputs["depth"],
            mask=inputs["depth_mask"].clone(),
            si=si,
        )
        losses["opt"][loss.name] = loss.weight * depth_losses.mean()
        losses_to_be_computed.remove("depth")

        # camera loss, here we apply to rays for simplicity
        # in the original training was on angles
        # however, we saw no difference (see supplementary)
        loss = self.losses["camera"]
        camera_losses = loss(outputs["rays"], target=inputs["rays"])
        losses["opt"][loss.name] = loss.weight * camera_losses.mean()
        losses_to_be_computed.remove("camera")

        # invariance loss
        flips = torch.tensor(
            [x.get("flip", False) for x in image_metas], device=self.device
        ).reshape(B)
        loss = self.losses["invariance"]
        invariance_losses = loss(
            outputs["cond_features"],
            intrinsics=inputs["camera"].K,
            mask=inputs["depth_mask"],
            flips=flips,
        )
        losses["opt"][loss.name] = loss.weight * invariance_losses.mean()
        losses_to_be_computed.remove("invariance")

        # remaining losses, we expect no more losses to be computed
        assert (
            not losses_to_be_computed
        ), f"Losses {losses_to_be_computed} not computed, revise `compute_loss` method"

        return losses

    @torch.no_grad()
    def infer(self, rgbs: torch.Tensor, intrinsics=None, skip_camera=False):
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

        (h, w), ratio = _shapes((H, W), self.image_shape)
        pad_left, pad_right, pad_top, pad_bottom = _paddings((h, w), self.image_shape)
        rgbs, gt_intrinsics = _preprocess(
            rgbs,
            intrinsics,
            (h, w),
            (pad_left, pad_right, pad_top, pad_bottom),
            ratio,
            self.image_shape,
        )

        # run encoder
        encoder_outputs, cls_tokens = self.pixel_encoder(rgbs)
        if "dino" in self.pixel_encoder.__class__.__name__.lower():
            encoder_outputs = [
                (x + y.unsqueeze(1)).contiguous()
                for x, y in zip(encoder_outputs, cls_tokens)
            ]

        # get data for decoder and adapt to given camera
        inputs = {}
        inputs["encoder_outputs"] = encoder_outputs
        inputs["cls_tokens"] = cls_tokens
        inputs["image"] = rgbs
        if gt_intrinsics is not None:
            rays, angles = generate_rays(
                gt_intrinsics, self.image_shape, noisy=self.training
            )
            inputs["rays"] = rays
            inputs["angles"] = angles
            inputs["K"] = gt_intrinsics
            self.pixel_decoder.test_fixed_camera = True
            self.pixel_decoder.skip_camera = skip_camera

        # decode all
        pred_intrinsics, predictions, _ = self.pixel_decoder(inputs, {})

        # undo the reshaping and get original image size (slow)
        predictions, pred_intrinsics = _postprocess(
            predictions,
            pred_intrinsics,
            self.image_shape,
            (pad_left, pad_right, pad_top, pad_bottom),
            ratio,
            (H, W),
        )

        # final 3D points backprojection
        intrinsics = gt_intrinsics if gt_intrinsics is not None else pred_intrinsics
        angles = generate_rays(intrinsics, (H, W), noisy=False)[-1]
        angles = rearrange(angles, "b (h w) c -> b c h w", h=H, w=W)
        points_3d = torch.cat((angles, predictions), dim=1)
        points_3d = spherical_zbuffer_to_euclidean(
            points_3d.permute(0, 2, 3, 1)
        ).permute(0, 3, 1, 2)

        # output data
        outputs = {
            "intrinsics": pred_intrinsics,
            "points": points_3d,
            "depth": predictions[:, -1:],
        }
        self.pixel_decoder.test_fixed_camera = False
        self.pixel_decoder.skip_camera = False
        return outputs

    def load_pretrained(self, model_file):
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        dict_model = torch.load(model_file, map_location=device)

        if "model" in dict_model:
            dict_model = dict_model["model"]
        new_state_dict = deepcopy(
            {k.replace("module.", ""): v for k, v in dict_model.items()}
        )

        info = self.load_state_dict(new_state_dict, strict=False)
        if is_main_process():
            print(
                f"Loaded from {model_file} for {self.__class__.__name__} results in:",
                info,
            )

    def get_params(self, config):
        if hasattr(self.pixel_encoder, "get_params"):
            encoder_p, encoder_lr = self.pixel_encoder.get_params(
                config["model"]["pixel_encoder"]["lr"],
                config["training"]["wd"],
                config["training"]["ld"],
            )
        else:
            encoder_p, encoder_lr = get_params(
                self.pixel_encoder,
                config["model"]["pixel_encoder"]["lr"],
                config["training"]["wd"],
            )
        decoder_p, decoder_lr = get_params(
            self.pixel_decoder, config["training"]["lr"], config["training"]["wd"]
        )
        return [*encoder_p, *decoder_p]

    @property
    def device(self):
        return next(self.parameters()).device

    def build(self, config):
        mod = importlib.import_module("unidepth.models.encoder")
        pixel_encoder_factory = getattr(mod, config["model"]["pixel_encoder"]["name"])
        pixel_encoder_config = {
            **config["training"],
            **config["data"],
            **config["model"]["pixel_encoder"],
            "interpolate_offset": 0.1,
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

        self.pixel_encoder = pixel_encoder
        self.pixel_decoder = Decoder(config)
        self.image_shape = config["data"]["image_shape"]

    def build_losses(self, config):
        self.losses = {}
        for loss_name, loss_config in config["training"].get("losses", {}).items():
            mod = importlib.import_module("unidepth.ops.losses")
            loss_factory = getattr(mod, loss_config["name"])
            self.losses[loss_name] = loss_factory.build(loss_config)
