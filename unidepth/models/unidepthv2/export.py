"""
Author: Luigi Piccinelli
Licensed under the CC-BY NC 4.0 license (http://creativecommons.org/licenses/by-nc/4.0/)
"""

import argparse
import json
import os
from math import ceil

import huggingface_hub
import torch.nn.functional as F
import torch.onnx

from unidepth.models.unidepthv2 import UniDepthV2


class UniDepthV2ONNX(UniDepthV2):
    def __init__(
        self,
        config,
        eps: float = 1e-6,
        **kwargs,
    ):
        super().__init__(config, eps)

    def forward(self, rgbs):
        B, _, H, W = rgbs.shape
        features, tokens = self.pixel_encoder(rgbs)

        inputs = {}
        inputs["image"] = rgbs
        inputs["features"] = [
            self.stacking_fn(features[i:j]).contiguous()
            for i, j in self.slices_encoder_range
        ]
        inputs["tokens"] = [
            self.stacking_fn(tokens[i:j]).contiguous()
            for i, j in self.slices_encoder_range
        ]
        outputs = self.pixel_decoder(inputs, [])
        outputs["rays"] = outputs["rays"].permute(0, 2, 1).reshape(B, 3, H, W)
        pts_3d = outputs["rays"] * outputs["radius"]

        return pts_3d, outputs["confidence"], outputs["intrinsics"]


class UniDepthV2ONNXcam(UniDepthV2):
    def __init__(
        self,
        config,
        eps: float = 1e-6,
        **kwargs,
    ):
        super().__init__(config, eps)

    def forward(self, rgbs, rays):
        B, _, H, W = rgbs.shape
        features, tokens = self.pixel_encoder(rgbs)

        inputs = {}
        inputs["image"] = rgbs
        inputs["rays"] = rays
        inputs["features"] = [
            self.stacking_fn(features[i:j]).contiguous()
            for i, j in self.slices_encoder_range
        ]
        inputs["tokens"] = [
            self.stacking_fn(tokens[i:j]).contiguous()
            for i, j in self.slices_encoder_range
        ]
        outputs = self.pixel_decoder(inputs, [])
        outputs["rays"] = outputs["rays"].permute(0, 2, 1).reshape(B, 3, H, W)
        pts_3d = outputs["rays"] * outputs["radius"]

        return pts_3d, outputs["confidence"], outputs["intrinsics"]



def export(model, path, shape=(462, 630), with_camera=False):
    model.eval()
    image = torch.rand(1, 3, *shape)
    dynamic_axes_in = {"rgbs": {0: "batch"}}
    inputs = [image]
    if with_camera:
        rays = torch.rand(1, 3, *shape)
        inputs.append(rays)
        dynamic_axes_in["rays"] = {0: "batch"}

    dynamic_axes_out = {
        "pts_3d": {0: "batch"},
        "confidence": {0: "batch"},
        "intrinsics": {0: "batch"},
    }
    torch.onnx.export(
        model,
        tuple(inputs),
        path,
        input_names=list(dynamic_axes_in.keys()),
        output_names=list(dynamic_axes_out.keys()),
        opset_version=14,
        dynamic_axes={**dynamic_axes_in, **dynamic_axes_out},
    )
    print(f"Model exported to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export UniDepthV2 model to ONNX")
    parser.add_argument(
        "--version", type=str, default="v2", choices=["v2"], help="UniDepth version"
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="vitl",
        choices=["vits", "vitb", "vitl"],
        help="Backbone model",
    )
    parser.add_argument(
        "--shape",
        type=int,
        nargs=2,
        default=(462, 630),
        help="Input shape. No dyamic shape supported!",
    )
    parser.add_argument(
        "--output-path", type=str, default="unidepthv2.onnx", help="Output ONNX file"
    )
    parser.add_argument(
        "--with-camera",
        action="store_true",
        help="Export model that expects GT camera as unprojected rays at inference",
    )
    args = parser.parse_args()

    version = args.version
    backbone = args.backbone
    shape = args.shape
    output_path = args.output_path
    with_camera = args.with_camera

    # force shape to be multiple of 14
    shape_rounded = [14 * ceil(x // 14 - 0.5) for x in shape]
    if list(shape) != list(shape_rounded):
        print(f"Shape {shape} is not multiple of 14. Rounding to {shape_rounded}")
        shape = shape_rounded

    # assumes command is from root of repo
    with open(os.path.join("configs", f"config_{version}_{backbone}.json")) as f:
        config = json.load(f)

    # tell DINO not to use efficient attention: not exportable
    config["training"]["export"] = True

    model = UniDepthV2ONNX(config) if not with_camera else UniDepthV2ONNXcam(config)
    path = huggingface_hub.hf_hub_download(
        repo_id=f"lpiccinelli/unidepth-{version}-{backbone}14",
        filename=f"pytorch_model.bin",
        repo_type="model",
    )
    info = model.load_state_dict(torch.load(path), strict=False)
    print(f"UniDepth_{version}_{backbone} is loaded with:")
    print(f"\t missing keys: {info.missing_keys}")
    print(f"\t additional keys: {info.unexpected_keys}")

    export(
        model=model,
        path=os.path.join(os.environ.get("TMPDIR", "."), output_path),
        shape=shape,
        with_camera=with_camera,
    )

