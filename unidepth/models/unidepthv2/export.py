"""
Author: Luigi Piccinelli
Licensed under the CC-BY NC 4.0 license (http://creativecommons.org/licenses/by-nc/4.0/)
"""

import os
import json
import argparse
from math import ceil

import torch.nn.functional as F
import torch.onnx
import huggingface_hub


from unidepth.utils.geometric import generate_rays
from unidepth.models.unidepthv2 import UniDepthV2


class UniDepthV2ONNX(UniDepthV2):
    def __init__(
        self,
        config,
        eps: float = 1e-6,
        **kwargs,
    ):
        super(UniDepthV2ONNX, self).__init__(config, eps)

    def forward(self, rgbs):
        H, W = rgbs.shape[-2:]

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

        inputs = {}
        inputs["image"] = rgbs
        inputs["features"] = features
        inputs["tokens"] = tokens
        inputs["global_tokens"] = global_tokens
        inputs["camera_tokens"] = camera_tokens

        outs = self.pixel_decoder(inputs, {})

        predictions = F.interpolate(
            outs["depth"],
            size=(H, W),
            mode="bilinear",
        )
        predictions_normalized = F.interpolate(
            outs["depth_ssi"],
            size=(H, W),
            mode="bilinear",
        )
        confidence = F.interpolate(
            outs["confidence"],
            size=(H, W),
            mode="bilinear",
        )

        return outs["K"], predictions, predictions_normalized, confidence


class UniDepthV2wCamONNX(UniDepthV2):
    def __init__(
        self,
        config,
        eps: float = 1e-6,
        **kwargs,
    ):
        super(UniDepthV2wCamONNX, self).__init__(config, eps)

    def forward(self, rgbs, K):
        H, W = rgbs.shape[-2:]

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

        inputs = {}
        inputs["image"] = rgbs
        inputs["features"] = features
        inputs["tokens"] = tokens
        inputs["global_tokens"] = global_tokens
        inputs["camera_tokens"] = camera_tokens
        rays, angles = generate_rays(K, (H, W))
        inputs["rays"] = rays
        inputs["angles"] = angles
        inputs["K"] = K

        outs = self.pixel_decoder(inputs, {})

        predictions = F.interpolate(
            outs["depth"],
            size=(H, W),
            mode="bilinear",
        )
        predictions_normalized = F.interpolate(
            outs["depth_ssi"],
            size=(H, W),
            mode="bilinear",
        )
        confidence = F.interpolate(
            outs["confidence"],
            size=(H, W),
            mode="bilinear",
        )

        return outs["K"], predictions, predictions_normalized, confidence


def export(model, path, shape=(462, 616), with_camera=False):
    model.eval()
    image = torch.rand(1, 3, *shape)
    dynamic_axes_in = {"image": {0: "batch"}}
    inputs = [image]
    if with_camera:
        K = torch.rand(1, 3, 3)
        inputs.append(K)
        dynamic_axes_in["K"] = {0: "batch"}

    dynamic_axes_out = {
        "out_K": {0: "batch"},
        "depth": {0: "batch"},
        "depth_ssi": {0: "batch"},
        "confidence": {0: "batch"},
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
        default="vitl14",
        choices=["vitl14"],
        help="Backbone model",
    )
    parser.add_argument(
        "--shape",
        type=int,
        nargs=2,
        default=(462, 616),
        help="Input shape. No dyamic shape supported!",
    )
    parser.add_argument(
        "--output-path", type=str, default="unidepthv2.onnx", help="Output ONNX file"
    )
    parser.add_argument(
        "--with-camera",
        action="store_true",
        help="Export model that expects GT camera matrix at inference",
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

    model_factory = UniDepthV2ONNX if not with_camera else UniDepthV2wCamONNX
    model = model_factory(config)
    path = huggingface_hub.hf_hub_download(
        repo_id=f"lpiccinelli/unidepth-{version}-{backbone}",
        filename=f"pytorch_model.bin",
        repo_type="model",
    )
    info = model.load_state_dict(torch.load(path), strict=False)
    print(f"UniDepth_{version}_{backbone} is loaded with:")
    print(f"\t missing keys: {info.missing_keys}")
    print(f"\t additional keys: {info.unexpected_keys}")

    export(
        model=model,
        path=os.path.join(os.environ["TMPDIR"], output_path),
        shape=shape,
        with_camera=with_camera,
    )
