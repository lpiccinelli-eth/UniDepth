from collections import OrderedDict
from functools import partial
from typing import Callable, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from timm.layers import (AvgPool2dSame, DropPath, GlobalResponseNormMlp,
                         LayerNorm, LayerNorm2d, Mlp, create_conv2d,
                         get_act_layer, make_divisible, to_ntuple,
                         trunc_normal_)
from torch.utils.checkpoint import checkpoint


def get_num_layer_for_convnext(var_name):
    """
    Divide [3, 3, 27, 3] layers into 12 groups; each group is three
    consecutive blocks, including possible neighboring downsample layers;
    adapted from https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py
    """
    if var_name.startswith("downsample_layers"):
        stage_id = int(var_name.split(".")[1])
        if stage_id == 0:
            layer_id = 0
        elif stage_id == 1 or stage_id == 2:
            layer_id = stage_id + 1
        elif stage_id == 3:
            layer_id = 12

    elif var_name.startswith("stages"):
        stage_id = int(var_name.split(".")[1])
        block_id = int(var_name.split(".")[3])
        if stage_id == 0 or stage_id == 1:
            layer_id = stage_id + 1
        elif stage_id == 2:
            layer_id = 3 + block_id // 3
        elif stage_id == 3:
            layer_id = 12

    elif var_name.startswith("stem"):
        return 0
    else:
        layer_id = 12
    return layer_id + 1


def get_parameter_groups(model, lr, wd=1e-5, ld=0.9, skip_list=None):
    parameter_group_names = {}
    parameter_group_vars = {}
    skip = set()
    if skip_list is not None:
        skip = skip_list
    if hasattr(model, "no_weight_decay"):
        skip.update(model.no_weight_decay())
    num_layers = 12
    layer_scale = list(ld ** (num_layers + 1 - i) for i in range(num_layers + 2))
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip:
            group_name = "no_decay"
            this_wd = 0.0
        else:
            group_name = "decay"
            this_wd = wd

        layer_id = get_num_layer_for_convnext(name)
        group_name = "layer_%d_%s" % (layer_id, group_name)

        if group_name not in parameter_group_names:
            scale = layer_scale[layer_id]
            cur_lr = lr * scale

            parameter_group_names[group_name] = {
                "weight_decay": this_wd,
                "weight_decay_init": this_wd,
                "weight_decay_base": this_wd,
                "params": [],
                "lr_init": cur_lr,
                "lr_base": lr,
                "lr": cur_lr,
            }
            parameter_group_vars[group_name] = {
                "weight_decay": this_wd,
                "weight_decay_init": this_wd,
                "weight_decay_base": this_wd,
                "params": [],
                "lr_init": cur_lr,
                "lr_base": lr,
                "lr": cur_lr,
            }
            if this_wd == 0.0:
                parameter_group_names[group_name]["weight_decay_final"] = 0.0
                parameter_group_vars[group_name]["weight_decay_final"] = 0.0
        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)
    # from unidepth.utils import is_main_process
    # import json
    # if is_main_process():
    #     print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    return list(parameter_group_vars.values()), [
        v["lr"] for k, v in parameter_group_vars.items()
    ]


class Downsample(nn.Module):
    def __init__(self, in_chs, out_chs, stride=1, dilation=1):
        super().__init__()
        avg_stride = stride if dilation == 1 else 1
        if stride > 1 or dilation > 1:
            avg_pool_fn = (
                AvgPool2dSame if avg_stride == 1 and dilation > 1 else nn.AvgPool2d
            )
            self.pool = avg_pool_fn(
                2, avg_stride, ceil_mode=True, count_include_pad=False
            )
        else:
            self.pool = nn.Identity()

        if in_chs != out_chs:
            self.conv = create_conv2d(in_chs, out_chs, 1, stride=1)
        else:
            self.conv = nn.Identity()

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x


class ConvNeXtBlock(nn.Module):
    """ConvNeXt Block
    There are two equivalent implementations:
      (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
      (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back

    Unlike the official impl, this one allows choice of 1 or 2, 1x1 conv can be faster with appropriate
    choice of LayerNorm impl, however as model size increases the tradeoffs appear to change and nn.Linear
    is a better choice. This was observed with PyTorch 1.10 on 3090 GPU, it could change over time & w/ different HW.
    """

    def __init__(
        self,
        in_chs: int,
        out_chs: Optional[int] = None,
        kernel_size: int = 7,
        stride: int = 1,
        dilation: Union[int, Tuple[int, int]] = (1, 1),
        mlp_ratio: float = 4,
        conv_mlp: bool = False,
        conv_bias: bool = True,
        use_grn: bool = False,
        ls_init_value: Optional[float] = 1e-6,
        act_layer: Union[str, Callable] = "gelu",
        norm_layer: Optional[Callable] = None,
        drop_path: float = 0.0,
    ):
        """

        Args:
            in_chs: Block input channels.
            out_chs: Block output channels (same as in_chs if None).
            kernel_size: Depthwise convolution kernel size.
            stride: Stride of depthwise convolution.
            dilation: Tuple specifying input and output dilation of block.
            mlp_ratio: MLP expansion ratio.
            conv_mlp: Use 1x1 convolutions for MLP and a NCHW compatible norm layer if True.
            conv_bias: Apply bias for all convolution (linear) layers.
            use_grn: Use GlobalResponseNorm in MLP (from ConvNeXt-V2)
            ls_init_value: Layer-scale init values, layer-scale applied if not None.
            act_layer: Activation layer.
            norm_layer: Normalization layer (defaults to LN if not specified).
            drop_path: Stochastic depth probability.
        """
        super().__init__()
        out_chs = out_chs or in_chs
        dilation = to_ntuple(2)(dilation)
        act_layer = get_act_layer(act_layer)
        if not norm_layer:
            norm_layer = LayerNorm2d if conv_mlp else LayerNorm
        mlp_layer = partial(
            GlobalResponseNormMlp if use_grn else Mlp, use_conv=conv_mlp
        )
        self.use_conv_mlp = conv_mlp
        self.conv_dw = create_conv2d(
            in_chs,
            out_chs,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation[0],
            depthwise=True,
            bias=conv_bias,
        )
        self.norm = norm_layer(out_chs)
        self.mlp = mlp_layer(out_chs, int(mlp_ratio * out_chs), act_layer=act_layer)
        self.gamma = (
            nn.Parameter(ls_init_value * torch.ones(out_chs))
            if ls_init_value is not None
            else None
        )
        if in_chs != out_chs or stride != 1 or dilation[0] != dilation[1]:
            self.shortcut = Downsample(
                in_chs, out_chs, stride=stride, dilation=dilation[0]
            )
        else:
            self.shortcut = nn.Identity()
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.conv_dw(x.contiguous())
        if self.use_conv_mlp:
            x = self.norm(x)
            x = self.mlp(x)
        else:
            x = x.permute(0, 2, 3, 1).contiguous()
            x = self.norm(x)
            x = self.mlp(x)
            x = x.permute(0, 3, 1, 2).contiguous()
        if self.gamma is not None:
            x = x.mul(self.gamma.reshape(1, -1, 1, 1))

        x = self.drop_path(x) + self.shortcut(shortcut)
        return x.contiguous()


class ConvNeXtStage(nn.Module):
    def __init__(
        self,
        in_chs,
        out_chs,
        kernel_size=7,
        stride=2,
        depth=2,
        dilation=(1, 1),
        drop_path_rates=None,
        ls_init_value=1.0,
        conv_mlp=False,
        conv_bias=True,
        use_grn=False,
        act_layer="gelu",
        norm_layer=None,
        norm_layer_cl=None,
    ):
        super().__init__()
        self.grad_checkpointing = False

        if in_chs != out_chs or stride > 1 or dilation[0] != dilation[1]:
            ds_ks = 2 if stride > 1 or dilation[0] != dilation[1] else 1
            pad = (
                "same" if dilation[1] > 1 else 0
            )  # same padding needed if dilation used
            self.downsample = nn.Sequential(
                norm_layer(in_chs),
                create_conv2d(
                    in_chs,
                    out_chs,
                    kernel_size=ds_ks,
                    stride=stride,
                    dilation=dilation[0],
                    padding=pad,
                    bias=conv_bias,
                ),
            )
            in_chs = out_chs
        else:
            self.downsample = nn.Identity()

        drop_path_rates = drop_path_rates or [0.0] * depth
        stage_blocks = []
        for i in range(depth):
            stage_blocks.append(
                ConvNeXtBlock(
                    in_chs=in_chs,
                    out_chs=out_chs,
                    kernel_size=kernel_size,
                    dilation=dilation[1],
                    drop_path=drop_path_rates[i],
                    ls_init_value=ls_init_value,
                    conv_mlp=conv_mlp,
                    conv_bias=conv_bias,
                    use_grn=use_grn,
                    act_layer=act_layer,
                    norm_layer=norm_layer if conv_mlp else norm_layer_cl,
                )
            )
            in_chs = out_chs
        self.blocks = nn.ModuleList(stage_blocks)

    def forward(self, x):
        xs = []
        x = self.downsample(x)
        for block in self.blocks:
            if self.grad_checkpointing:
                x = checkpoint(block, x)
            else:
                x = block(x)
            xs.append(x)
        return xs


class ConvNeXt(nn.Module):
    def __init__(
        self,
        in_chans: int = 3,
        output_stride: int = 32,
        depths: Tuple[int, ...] = (3, 3, 9, 3),
        dims: Tuple[int, ...] = (96, 192, 384, 768),
        kernel_sizes: Union[int, Tuple[int, ...]] = 7,
        ls_init_value: Optional[float] = 1e-6,
        stem_type: str = "patch",
        patch_size: int = 4,
        conv_mlp: bool = False,
        conv_bias: bool = True,
        use_grn: bool = False,
        act_layer: Union[str, Callable] = "gelu",
        norm_layer: Optional[Union[str, Callable]] = None,
        norm_eps: Optional[float] = None,
        drop_path_rate: float = 0.0,
        output_idx=[],
        use_checkpoint=False,
    ):
        """
        Args:
            in_chans: Number of input image channels.
            num_classes: Number of classes for classification head.
            global_pool: Global pooling type.
            output_stride: Output stride of network, one of (8, 16, 32).
            depths: Number of blocks at each stage.
            dims: Feature dimension at each stage.
            kernel_sizes: Depthwise convolution kernel-sizes for each stage.
            ls_init_value: Init value for Layer Scale, disabled if None.
            stem_type: Type of stem.
            patch_size: Stem patch size for patch stem.
            head_init_scale: Init scaling value for classifier weights and biases.
            head_norm_first: Apply normalization before global pool + head.
            head_hidden_size: Size of MLP hidden layer in head if not None and head_norm_first == False.
            conv_mlp: Use 1x1 conv in MLP, improves speed for small networks w/ chan last.
            conv_bias: Use bias layers w/ all convolutions.
            use_grn: Use Global Response Norm (ConvNeXt-V2) in MLP.
            act_layer: Activation layer type.
            norm_layer: Normalization layer type.
            drop_rate: Head pre-classifier dropout rate.
            drop_path_rate: Stochastic depth drop rate.
        """
        super().__init__()
        self.num_layers = len(depths)
        self.depths = output_idx
        self.embed_dims = [
            int(dim) for i, dim in enumerate(dims) for _ in range(depths[i])
        ]
        self.embed_dim = dims[0]

        assert output_stride in (8, 16, 32)
        kernel_sizes = to_ntuple(4)(kernel_sizes)
        if norm_layer is None:
            norm_layer = LayerNorm2d
            norm_layer_cl = norm_layer if conv_mlp else LayerNorm
            if norm_eps is not None:
                norm_layer = partial(norm_layer, eps=norm_eps)
                norm_layer_cl = partial(norm_layer_cl, eps=norm_eps)
        else:
            assert (
                conv_mlp
            ), "If a norm_layer is specified, conv MLP must be used so all norm expect rank-4, channels-first input"
            norm_layer_cl = norm_layer
            if norm_eps is not None:
                norm_layer_cl = partial(norm_layer_cl, eps=norm_eps)

        self.feature_info = []

        assert stem_type in ("patch", "overlap", "overlap_tiered")
        if stem_type == "patch":
            # NOTE: this stem is a minimal form of ViT PatchEmbed, as used in SwinTransformer w/ patch_size = 4
            self.stem = nn.Sequential(
                nn.Conv2d(
                    in_chans,
                    dims[0],
                    kernel_size=patch_size,
                    stride=patch_size,
                    bias=conv_bias,
                ),
                norm_layer(dims[0]),
            )
            stem_stride = patch_size
        else:
            mid_chs = make_divisible(dims[0] // 2) if "tiered" in stem_type else dims[0]
            self.stem = nn.Sequential(
                nn.Conv2d(
                    in_chans,
                    mid_chs,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=conv_bias,
                ),
                nn.Conv2d(
                    mid_chs, dims[0], kernel_size=3, stride=2, padding=1, bias=conv_bias
                ),
                norm_layer(dims[0]),
            )
            stem_stride = 4

        self.stages = nn.Sequential()
        dp_rates = [
            x.tolist()
            for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)
        ]
        stages = []
        prev_chs = dims[0]
        curr_stride = stem_stride
        dilation = 1
        # 4 feature resolution stages, each consisting of multiple residual blocks
        for i in range(4):
            stride = 2 if curr_stride == 2 or i > 0 else 1
            if curr_stride >= output_stride and stride > 1:
                dilation *= stride
                stride = 1
            curr_stride *= stride
            first_dilation = 1 if dilation in (1, 2) else 2
            out_chs = dims[i]
            stages.append(
                ConvNeXtStage(
                    prev_chs,
                    out_chs,
                    kernel_size=kernel_sizes[i],
                    stride=stride,
                    dilation=(first_dilation, dilation),
                    depth=depths[i],
                    drop_path_rates=dp_rates[i],
                    ls_init_value=ls_init_value,
                    conv_mlp=conv_mlp,
                    conv_bias=conv_bias,
                    use_grn=use_grn,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    norm_layer_cl=norm_layer_cl,
                )
            )
            prev_chs = out_chs
            # NOTE feature_info use currently assumes stage 0 == stride 1, rest are stride 2
            self.feature_info += [
                dict(num_chs=prev_chs, reduction=curr_stride, module=f"stages.{i}")
            ]
        self.stages = nn.ModuleList(stages)
        self.mask_token = nn.Parameter(torch.zeros(1, self.embed_dim, 1, 1))
        self.num_features = prev_chs
        self.apply(self._init_weights)
        self.set_grad_checkpointing(use_checkpoint)

    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            nn.init.zeros_(module.bias)

    def forward(self, x, masks=None):
        outs = []
        x = self.stem(x)
        if masks is not None:
            masks = torch.nn.functional.interpolate(
                masks.float(), size=x.shape[-2:], mode="nearest"
            )
            x = torch.where(masks.bool(), self.mask_token.to(x.dtype), x).contiguous()
        for stage in self.stages:
            xs = stage(x)
            outs.extend([x.permute(0, 2, 3, 1).contiguous() for x in xs])
            x = xs[-1]
        return outs, [x.mean(dim=(1, 2)).unsqueeze(1).contiguous() for x in outs]

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(
            stem=r"^stem",
            blocks=(
                r"^stages\.(\d+)"
                if coarse
                else [
                    (r"^stages\.(\d+)\.downsample", (0,)),  # blocks
                    (r"^stages\.(\d+)\.blocks\.(\d+)", None),
                    (r"^norm_pre", (99999,)),
                ]
            ),
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        for s in self.stages:
            s.grad_checkpointing = enable

    def freeze(self) -> None:
        for module in self.modules():
            module.eval()
        for parameters in self.parameters():
            parameters.requires_grad = False

    def get_params(self, lr, wd, ld, *args, **kwargs):
        encoder_p, encoder_lr = get_parameter_groups(self, lr, wd, ld)
        return encoder_p, encoder_lr

    def no_weight_decay(self):
        return {"mask_token"}

    @classmethod
    def build(cls, config):
        obj = globals()[config["model"]["encoder"]["name"]](config)
        return obj


def checkpoint_filter_fn(state_dict, model):
    """Remap FB checkpoints -> timm"""
    if "head.norm.weight" in state_dict or "norm_pre.weight" in state_dict:
        return state_dict  # non-FB checkpoint
    if "model" in state_dict:
        state_dict = state_dict["model"]

    out_dict = {}
    if "visual.trunk.stem.0.weight" in state_dict:
        out_dict = {
            k.replace("visual.trunk.", ""): v
            for k, v in state_dict.items()
            if k.startswith("visual.trunk.")
        }
        if "visual.head.proj.weight" in state_dict:
            out_dict["head.fc.weight"] = state_dict["visual.head.proj.weight"]
            out_dict["head.fc.bias"] = torch.zeros(
                state_dict["visual.head.proj.weight"].shape[0]
            )
        elif "visual.head.mlp.fc1.weight" in state_dict:
            out_dict["head.pre_logits.fc.weight"] = state_dict[
                "visual.head.mlp.fc1.weight"
            ]
            out_dict["head.pre_logits.fc.bias"] = state_dict["visual.head.mlp.fc1.bias"]
            out_dict["head.fc.weight"] = state_dict["visual.head.mlp.fc2.weight"]
            out_dict["head.fc.bias"] = torch.zeros(
                state_dict["visual.head.mlp.fc2.weight"].shape[0]
            )
        return out_dict

    import re

    for k, v in state_dict.items():
        k = k.replace("downsample_layers.0.", "stem.")
        k = re.sub(r"stages.([0-9]+).([0-9]+)", r"stages.\1.blocks.\2", k)
        k = re.sub(
            r"downsample_layers.([0-9]+).([0-9]+)", r"stages.\1.downsample.\2", k
        )
        k = k.replace("dwconv", "conv_dw")
        k = k.replace("pwconv", "mlp.fc")
        if "grn" in k:
            k = k.replace("grn.beta", "mlp.grn.bias")
            k = k.replace("grn.gamma", "mlp.grn.weight")
            v = v.reshape(v.shape[-1])
        k = k.replace("head.", "head.fc.")
        if k.startswith("norm."):
            k = k.replace("norm", "head.norm")
        if v.ndim == 2 and "head" not in k:
            model_shape = model.state_dict()[k].shape
            v = v.reshape(model_shape)
        out_dict[k] = v

    return out_dict


HF_URL = {
    "convnext_xxlarge_pt": (
        "laion/CLIP-convnext_xxlarge-laion2B-s34B-b82K-augreg-soup",
        "open_clip_pytorch_model.bin",
    ),
    "convnext_large_pt": (
        "laion/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup",
        "open_clip_pytorch_model.bin",
    ),
    "convnext_large": (
        "timm/convnext_large_mlp.clip_laion2b_soup_ft_in12k_in1k_384",
        "pytorch_model.bin",
    ),
}
