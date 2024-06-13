dependencies = ["torch", "huggingface_hub"]

import os
import json

import torch
import huggingface_hub

from unidepth.models import UniDepthV1, UniDepthV2


MAP_VERSIONS = {
    "v1": UniDepthV1,
    "v2": UniDepthV2
}

BACKBONES = {
    "v1": ["vitl14", "cnvnxtl"],
    "v2": ["vitl14", "vits14"]
}


def UniDepth(version="v2", backbone="vitl14", pretrained=True):
    assert version in MAP_VERSIONS.keys(), f"version must be one of {list(MAP_VERSIONS.keys())}"
    assert backbone in BACKBONES[version], f"backbone for current version ({version}) must be one of {list(BACKBONES[version])}"
    repo_dir = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(repo_dir, "configs", f"config_{version}_{backbone}.json")) as f:
        config = json.load(f)
    
    model = MAP_VERSIONS[version](config)
    if pretrained:
        path = huggingface_hub.hf_hub_download(repo_id=f"lpiccinelli/unidepth-{version}-{backbone}", filename=f"pytorch_model.bin", repo_type="model")
        info = model.load_state_dict(torch.load(path), strict=False)
        print(f"UniDepth_{version}_{backbone} is loaded with:")
        print(f"\t missing keys: {info.missing_keys}")
        print(f"\t additional keys: {info.unexpected_keys}")

    return model

