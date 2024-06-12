import torch
import torch.nn as nn

from unidepth.models.backbones import ConvNeXt, ConvNeXtV2, _make_dinov2_model


class ModelWrap(nn.Module):
    def __init__(self, model) -> None:
        super().__init__()
        self.backbone = model

    def forward(self, x, *args, **kwargs):
        features = []
        for layer in self.backbone.features:
            x = layer(x)
            features.append(x)
        return features


def convnextv2_base(config, **kwargs):
    model = ConvNeXtV2(
        depths=[3, 3, 27, 3],
        dims=[128, 256, 512, 1024],
        output_idx=config.get("output_idx", [3, 6, 33, 36]),
        use_checkpoint=config.get("use_checkpoint", False),
        **kwargs,
    )
    url = "https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_base_22k_384_ema.pt"
    state_dict = torch.hub.load_state_dict_from_url(
        url, map_location="cpu", progress=False
    )["model"]
    info = model.load_state_dict(state_dict, strict=False)
    print(info)
    return model


def convnextv2_large(config, **kwargs):
    model = ConvNeXtV2(
        depths=[3, 3, 27, 3],
        dims=[192, 384, 768, 1536],
        output_idx=config.get("output_idx", [3, 6, 33, 36]),
        use_checkpoint=config.get("use_checkpoint", False),
        **kwargs,
    )
    url = "https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_large_22k_384_ema.pt"
    state_dict = torch.hub.load_state_dict_from_url(
        url, map_location="cpu", progress=False
    )["model"]
    info = model.load_state_dict(state_dict, strict=False)
    print(info)
    return model


def convnextv2_large_mae(config, **kwargs):
    model = ConvNeXtV2(
        depths=[3, 3, 27, 3],
        dims=[192, 384, 768, 1536],
        output_idx=config.get("output_idx", [3, 6, 33, 36]),
        use_checkpoint=config.get("use_checkpoint", False),
        **kwargs,
    )
    url = "https://dl.fbaipublicfiles.com/convnext/convnextv2/pt_only/convnextv2_large_1k_224_fcmae.pt"
    state_dict = torch.hub.load_state_dict_from_url(
        url, map_location="cpu", progress=False
    )["model"]
    info = model.load_state_dict(state_dict, strict=False)
    print(info)
    return model


def convnextv2_huge(config, **kwargs):
    model = ConvNeXtV2(
        depths=[3, 3, 27, 3],
        dims=[352, 704, 1408, 2816],
        output_idx=config.get("output_idx", [3, 6, 33, 36]),
        use_checkpoint=config.get("use_checkpoint", False),
        **kwargs,
    )
    url = "https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_huge_22k_512_ema.pt"
    state_dict = torch.hub.load_state_dict_from_url(
        url, map_location="cpu", progress=False
    )["model"]
    info = model.load_state_dict(state_dict, strict=False)
    print(info)
    return model


def convnextv2_huge_mae(config, **kwargs):
    model = ConvNeXtV2(
        depths=[3, 3, 27, 3],
        dims=[352, 704, 1408, 2816],
        output_idx=config.get("output_idx", [3, 6, 33, 36]),
        use_checkpoint=config.get("use_checkpoint", False),
        **kwargs,
    )
    url = "https://dl.fbaipublicfiles.com/convnext/convnextv2/pt_only/convnextv2_huge_1k_224_fcmae.pt"
    state_dict = torch.hub.load_state_dict_from_url(
        url, map_location="cpu", progress=False
    )["model"]
    info = model.load_state_dict(state_dict, strict=False)
    print(info)
    return model


def convnext_large_pt(config, **kwargs):
    model = ConvNeXt(
        depths=[3, 3, 27, 3],
        dims=[192, 384, 768, 1536],
        output_idx=config.get("output_idx", [3, 6, 33, 36]),
        use_checkpoint=config.get("use_checkpoint", False),
        **kwargs,
    )
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import disable_progress_bars

    from unidepth.models.backbones.convnext import HF_URL, checkpoint_filter_fn

    disable_progress_bars()
    repo_id, filename = HF_URL["convnext_large_pt"]
    state_dict = torch.load(hf_hub_download(repo_id=repo_id, filename=filename))
    state_dict = checkpoint_filter_fn(state_dict, model)
    info = model.load_state_dict(state_dict, strict=False)
    print(info)
    return model


def convnext_large(config, **kwargs):
    model = ConvNeXt(
        depths=[3, 3, 27, 3],
        dims=[192, 384, 768, 1536],
        output_idx=config.get("output_idx", [3, 6, 33, 36]),
        use_checkpoint=config.get("use_checkpoint", False),
        drop_path_rate=config.get("drop_path", 0.0),
        **kwargs,
    )
    return model


def dinov2_vits14(config, pretrained: bool = True, **kwargs):
    """
    DINOv2 ViT-S/14 model (optionally) pretrained on the LVD-142M dataset.
    """
    vit = _make_dinov2_model(
        arch_name="vit_small",
        pretrained=config["pretrained"],
        output_idx=config.get("output_idx", [3, 6, 9, 12]),
        checkpoint=config.get("use_checkpoint", False),
        drop_path_rate=config.get("drop_path", 0.0),
        num_register_tokens=config.get("num_register_tokens", 0),
        use_norm=config.get("use_norm", False),
        export=config.get("export", False),
        interpolate_offset=config.get("interpolate_offset", 0.0),
        **kwargs,
    )
    return vit


def dinov2_vitb14(config, pretrained: bool = True, **kwargs):
    """
    DINOv2 ViT-B/14 model (optionally) pretrained on the LVD-142M dataset.
    """
    vit = _make_dinov2_model(
        arch_name="vit_base",
        pretrained=config["pretrained"],
        output_idx=config.get("output_idx", [3, 6, 9, 12]),
        checkpoint=config.get("use_checkpoint", False),
        drop_path_rate=config.get("drop_path", 0.0),
        num_register_tokens=config.get("num_register_tokens", 0),
        use_norm=config.get("use_norm", False),
        export=config.get("export", False),
        interpolate_offset=config.get("interpolate_offset", 0.0),
        **kwargs,
    )
    return vit


def dinov2_vitl14(config, pretrained: str = "", **kwargs):
    """
    DINOv2 ViT-L/14 model (optionally) pretrained on the LVD-142M dataset.
    """
    vit = _make_dinov2_model(
        arch_name="vit_large",
        pretrained=config["pretrained"],
        output_idx=config.get("output_idx", [5, 12, 18, 24]),
        checkpoint=config.get("use_checkpoint", False),
        drop_path_rate=config.get("drop_path", 0.0),
        num_register_tokens=config.get("num_register_tokens", 0),
        use_norm=config.get("use_norm", False),
        export=config.get("export", False),
        interpolate_offset=config.get("interpolate_offset", 0.0),
        **kwargs,
    )
    return vit
