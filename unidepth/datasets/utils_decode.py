import io

import cv2
import numpy as np
import torch
import torchvision
import torchvision.transforms.v2.functional as TF
from PIL import Image

from unidepth.utils.camera import (EUCM, MEI, BatchCamera, Fisheye624, Pinhole,
                                   Spherical)


def decode_depth(results, h5file, value, idx, depth_scale, name="depth", **kwargs):
    file = h5file.get_node("/" + value).read()
    decoded_data = Image.open(io.BytesIO(file))
    decoded_data = TF.pil_to_tensor(decoded_data).squeeze()

    if decoded_data.ndim == 3:  # 24 channel loading
        decoded_channels = [
            (decoded_data[0] & 0xFF).to(torch.int32),
            (decoded_data[1] & 0xFF).to(torch.int32),
            (decoded_data[2] & 0xFF).to(torch.int32),
        ]
        # Reshape and extract the original depth map
        decoded_data = (
            decoded_channels[0]
            | (decoded_channels[1] << 8)
            | (decoded_channels[2] << 16)
        )

    decoded_data = decoded_data.to(torch.float32)
    results.get("gt_fields", set()).add(name)
    results[(idx, 0)].get("gt_fields", set()).add(name)
    results[f"{name}_ori_shape"] = decoded_data.shape
    results[(idx, 0)][name] = (
        decoded_data.view(1, 1, *decoded_data.shape).contiguous() / depth_scale
    )
    return results


def decode_numpy(results, h5file, value, idx, name="points", **kwargs):
    file = h5file.get_node("/" + value).read()
    decoded_data = np.load(io.BytesIO(file), allow_pickle=False)
    decoded_data = torch.from_numpy(decoded_data).to(torch.float32)
    if decoded_data.ndim > 2:
        decoded_data = decoded_data.permute(2, 0, 1)
    results.get("gt_fields", set()).add(name)
    results[(idx, 0)].get("gt_fields", set()).add(name)
    results[(idx, 0)][name] = decoded_data.unsqueeze(0)
    return results


def decode_tensor(results, value, idx, name, **kwargs):
    results.get("camera_fields", set()).add(name)
    results[(idx, 0)].get("camera_fields", set()).add(name)
    results[(idx, 0)][name] = torch.tensor(value).unsqueeze(0)
    return results


def decode_camera(results, value, idx, name, sample, j, **kwargs):
    results.get("camera_fields", set()).add(name)
    results[(idx, 0)].get("camera_fields", set()).add(name)
    camera = eval(sample["camera_model"][j])(params=torch.tensor(value).unsqueeze(0))
    results[(idx, 0)][name] = BatchCamera.from_camera(camera)
    return results


def decode_K(results, value, idx, name, **kwargs):
    results.get("camera_fields", set()).add(name)
    results[(idx, 0)].get("camera_fields", set()).add(name)
    camera = Pinhole(K=torch.tensor(value).unsqueeze(0))
    results[(idx, 0)][name] = BatchCamera.from_camera(camera)
    return results


def decode_mask(results, h5file, value, idx, name, **kwargs):
    file = h5file.get_node("/" + value).read()
    mask = torchvision.io.decode_image(torch.from_numpy(file)).bool().squeeze()
    results.get("mask_fields", set()).add(name)
    results[(idx, 0)].get("mask_fields", set()).add(name)
    results[f"{name}_ori_shape"] = mask.shape[-2:]
    results[(idx, 0)][name] = mask.view(1, 1, *mask.shape).contiguous()
    return results


def decode_rgb(results, h5file, value, idx, name="image", **kwargs):
    file = h5file.get_node("/" + value).read()
    image = (
        torchvision.io.decode_image(torch.from_numpy(file)).to(torch.uint8).squeeze()
    )
    results.get("image_fields", set()).add(name)
    results[(idx, 0)].get("image_fields", set()).add(name)
    results[f"{name}_ori_shape"] = image.shape[-2:]
    if image.ndim == 2:
        image = image.unsqueeze(0).repeat(3, 1, 1)
    results[(idx, 0)][name] = image.unsqueeze(0)
    return results


def decode_flow(results, h5file, value, idx, name, **kwargs):
    file = h5file.get_node("/" + value).read()
    image = (
        torchvision.io.decode_image(torch.from_numpy(file)).to(torch.uint8).squeeze()
    )
    decoded_channels = [
        (image[0] & 0xFF).to(torch.int16),
        (image[1] & 0xFF).to(torch.int16),
        (image[2] & 0xFF).to(torch.int16),
    ]

    # Reshape and extract the original 2-channel flow map
    flow = torch.zeros((2, image.shape[1], image.shape[2]), dtype=torch.int16)
    flow[0] = (decoded_channels[0] | decoded_channels[1] << 8) & 0xFFF
    flow[1] = (decoded_channels[1] >> 4 | decoded_channels[2] << 4) & 0xFFF

    results.get("gt_fields", set()).add(name)
    results[(idx, 0)].get("gt_fields", set()).add(name)
    results[f"{name}_ori_shape"] = flow.shape[-2:]
    flow = flow.unsqueeze(0).contiguous().float()
    results[(idx, 0)][name] = (0.5 + flow) / 4095.0 * 2 - 1
    return results
