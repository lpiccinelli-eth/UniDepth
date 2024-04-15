dependencies = ["torch"]

import os
import json

import torch
import huggingface_hub

from unidepth.models import UniDepthV1


MAP_VERSIONS = {
    "v1": UniDepthV1
}

MAP_BACKBONES = {
    "v1": {
        "ViTL14": "vitl14", 
        "ConvNextL": "cnvnxtl"
    }
}


def UniDepth(version="v1", backbone="ViTL14", pretrained=True):
    assert version in MAP_VERSIONS.keys(), f"version must be one of {list(MAP_VERSIONS.keys())}"
    assert backbone in MAP_BACKBONES[version].keys(), f"backbone must be one of {list(MAP_BACKBONES[version].keys())}"
    backbones = MAP_BACKBONES[version]
    repo_dir = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(repo_dir, "configs", f"config_{version}_{backbones[backbone]}.json")) as f:
        config = json.load(f)
    
    model = MAP_VERSIONS[version](config)
    if pretrained:
        path = huggingface_hub.hf_hub_download(repo_id=f"lpiccinelli/unidepth-{version}-{backbones[backbone]}", filename=f"pytorch_model.bin", repo_type="model")
        info = model.load_state_dict(torch.load(path), strict=False)
        print(f"UniDepth_{version}_{backbone} is loaded with:")
        print(f"\tmissing keys: {info.missing_keys}\n\tadditional keys: {info.unexpected_keys}")

    return model



def UniDepthV1_ViTL14(pretrained):
    repo_dir = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(repo_dir, "configs", "config_v1_vitl14.json")) as f:
        config = json.load(f)
    
    model = UniDepthV1(config)
    if pretrained:
        path = huggingface_hub.hf_hub_download(repo_id="lpiccinelli/unidepth-v1-vitl14", filename="pytorch_model.bin", repo_type="model")
        info = model.load_state_dict(torch.load(path), strict=False)
        print("UniDepthV1_ViTL14 is loaded with:")
        print(f"\tmissing keys: {info.missing_keys}\n\tadditional keys: {info.unexpected_keys}")

    return model


def UniDepthV1_ConvNextL(pretrained):
    repo_dir = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(repo_dir, "configs", "config_v1_cnvnxtl.json")) as f:
        config = json.load(f)
    
    model = UniDepthV1(config)
    if pretrained:
        path = huggingface_hub.hf_hub_download(repo_id="lpiccinelli/unidepth-v1-cnvnxtl", filename="pytorch_model.bin", repo_type="model")
        info = model.load_state_dict(torch.load(path), strict=False)
        print("UniDepthV1_ConvNextL is loaded with:")
        print(f"\tmissing keys: {info.missing_keys}\n\tadditional keys: {info.unexpected_keys}")

    return model