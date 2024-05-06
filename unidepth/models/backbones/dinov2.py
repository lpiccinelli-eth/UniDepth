from functools import partial
import math
import logging
from typing import Sequence, Callable

import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_

from .metadinov2 import (
    Mlp,
    PatchEmbed,
    SwiGLUFFNFused,
    MemEffAttention,
    Attention,
    NestedTensorBlock as Block,
)


_DINOV2_BASE_URL = "https://dl.fbaipublicfiles.com/dinov2"
logger = logging.getLogger("dinov2")


def named_apply(
    fn: Callable, module: nn.Module, name="", depth_first=True, include_root=False
) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        named_apply(
            fn=fn,
            module=child_module,
            name=child_name,
            depth_first=depth_first,
            include_root=True,
        )
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


def get_parameter_groups(model, lr, wd=1e-5, ld=0.9, skip_list=()):
    parameter_group_names = {}
    parameter_group_vars = {}
    skip = {}
    if skip_list is not None:
        skip = skip_list
    elif hasattr(model, "no_weight_decay"):
        skip = model.no_weight_decay()

    num_layers = model.n_blocks
    layer_scale = list(ld ** (num_layers - i) for i in range(num_layers))

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if len(param.shape) == 1:  # norm
            group_name = "no_decay"
            this_wd = 0.0
        # layer scale, bias beta?
        elif (
            name in skip
            or name.endswith(".gamma")
            or name.endswith(".beta")
            or name.endswith(".bias")
        ):
            group_name = "no_decay"
            this_wd = 0.0
        elif "cls_token" in name or "pos_embed" in name or "mask_token" in name:
            group_name = "no_decay"
            this_wd = 0.0
        else:
            group_name = "decay"
            this_wd = wd

        if name.startswith("blocks"):
            layer_id = int(name.split(".")[1])
        elif name.startswith("patch_embed"):
            layer_id = 0
        else:
            layer_id = 0

        group_name = f"layer_{layer_id}_{group_name}"

        if group_name not in parameter_group_names:
            scale = layer_scale[layer_id]
            cur_lr = lr * scale

            parameter_group_names[group_name] = {
                "weight_decay": this_wd,
                "params": [],
                "lr_init": cur_lr,
                "lr_base": lr,
                "lr": cur_lr,
            }
            parameter_group_vars[group_name] = {
                "weight_decay": this_wd,
                "params": [],
                "lr_init": cur_lr,
                "lr_base": lr,
                "lr": cur_lr,
            }
        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)
    return list(parameter_group_vars.values()), [
        v["lr"] for k, v in parameter_group_vars.items()
    ]


class BlockChunk(nn.ModuleList):
    def forward(self, x):
        for b in self:
            x = b(x)
        return x


class DinoVisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        ffn_bias=True,
        proj_bias=True,
        drop_path_rate=0.0,
        drop_path_uniform=False,
        init_values=None,  # for layerscale: None or 0 => no layerscale
        embed_layer=PatchEmbed,
        act_layer=nn.GELU,
        block_fn=Block,
        ffn_layer="mlp",
        block_chunks=1,
        output_idx=[5, 12, 18, 24],
        checkpoint: bool = False,
        num_register_tokens=0,
        interpolate_antialias=False,
        interpolate_offset=0.1,
        use_norm=False,
    ):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            proj_bias (bool): enable bias for proj in attn if True
            ffn_bias (bool): enable bias for ffn if True
            drop_path_rate (float): stochastic depth rate
            drop_path_uniform (bool): apply uniform drop rate across blocks
            weight_init (str): weight init scheme
            init_values (float): layer-scale init values
            embed_layer (nn.Module): patch embedding layer
            act_layer (nn.Module): MLP activation layer
            block_fn (nn.Module): transformer block class
            ffn_layer (str): "mlp", "swiglu", "swiglufused" or "identity"
            block_chunks: (int) split block sequence into block_chunks units for FSDP wrap
        """
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.num_features = self.embed_dim = (
            embed_dim  # num_features for consistency with other models
        )
        self.embed_dims = [embed_dim] * output_idx[-1]
        self.num_tokens = 1
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.depths = output_idx
        self.checkpoint = checkpoint
        self.num_register_tokens = num_register_tokens
        self.interpolate_antialias = interpolate_antialias
        self.interpolate_offset = interpolate_offset

        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + self.num_tokens, embed_dim)
        )
        assert num_register_tokens >= 0
        self.register_tokens = nn.Parameter(
            torch.zeros(1, max(1, num_register_tokens), embed_dim)
        )

        if drop_path_uniform is True:
            dpr = [drop_path_rate] * depth
        else:
            dpr = [
                x.item() for x in torch.linspace(0, drop_path_rate, depth)
            ]  # stochastic depth decay rule

        if ffn_layer == "mlp":
            logger.info("using MLP layer as FFN")
            ffn_layer = Mlp
        elif ffn_layer == "swiglufused" or ffn_layer == "swiglu":
            logger.info("using SwiGLU layer as FFN")
            ffn_layer = SwiGLUFFNFused
        elif ffn_layer == "identity":
            logger.info("using Identity layer as FFN")

            def f(*args, **kwargs):
                return nn.Identity()

            ffn_layer = f
        else:
            raise NotImplementedError

        blocks_list = [
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                ffn_layer=ffn_layer,
                init_values=init_values,
            )
            for i in range(depth)
        ]
        if block_chunks > 0:
            self.chunked_blocks = True
            chunked_blocks = []
            chunksize = depth // block_chunks
            for i in range(0, depth, chunksize):
                # this is to keep the block index consistent if we chunk the block list
                chunked_blocks.append(
                    [nn.Identity()] * i + blocks_list[i : i + chunksize]
                )
            self.blocks = nn.ModuleList([BlockChunk(p) for p in chunked_blocks])
        else:
            self.chunked_blocks = False
            self.blocks = nn.ModuleList(blocks_list)

        self.norm = norm_layer(embed_dim)
        self.use_norm = use_norm
        self.head = nn.Identity()
        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))
        self.init_weights()

    def init_weights(self):
        trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=1e-6)
        if self.num_register_tokens:
            nn.init.normal_(self.register_tokens, std=1e-6)
        named_apply(init_weights_vit_timm, self)

    def interpolate_pos_encoding(self, x, w, h):
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + self.interpolate_offset, h0 + self.interpolate_offset

        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(
                1, int(math.sqrt(N)), int(math.sqrt(N)), dim
            ).permute(0, 3, 1, 2),
            scale_factor=(int(w0) / math.sqrt(N), int(h0) / math.sqrt(N)),
            mode="bicubic",
            antialias=self.interpolate_antialias,
        )

        assert (
            int(w0) == patch_pos_embed.shape[-2]
            and int(h0) == patch_pos_embed.shape[-1]
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(
            previous_dtype
        )

    def prepare_tokens_with_masks(self, x, masks=None):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)
        if masks is not None:
            masks = masks.bool().view(B, -1, 1)
            x = torch.where(masks, self.mask_token.to(x.dtype).unsqueeze(0), x)

        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)

        if self.num_register_tokens:
            x = torch.cat(
                (x[:, :1], self.register_tokens.expand(x.shape[0], -1, -1), x[:, 1:]),
                dim=1,
            )
        return x

    def forward(self, x, masks=None):
        shapes = [val // self.patch_size for val in x.shape[-2:]]
        batch_size = x.shape[0]
        x = self.prepare_tokens_with_masks(x, masks)
        outputs = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            outputs.append(x)

        if self.use_norm:
            outputs = [self.norm(out) for out in outputs]
        class_tokens = [out[:, :1] for out in outputs]
        outputs = [out[:, self.num_register_tokens + 1 :] for out in outputs]
        outputs = [out.reshape(batch_size, *shapes, -1) for out in outputs]

        return (outputs, class_tokens)

    def get_params(self, lr, wd, ld, *args, **kwargs):
        encoder_p, encoder_lr = get_parameter_groups(self, lr, wd, ld)
        return encoder_p, encoder_lr

    def freeze(self) -> None:
        for module in self.modules():
            module.eval()
        for parameters in self.parameters():
            parameters.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        self.mask_token.requires_grad = False
        self.register_tokens.requires_grad = False


def init_weights_vit_timm(module: nn.Module, name: str = ""):
    """ViT weight initialization, original timm impl (for reproducibility)"""
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def vit_small(patch_size=16, num_register_tokens=0, export=False, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        num_register_tokens=num_register_tokens,
        block_fn=partial(Block, attn_class=Attention if export else MemEffAttention),
        **kwargs,
    )
    return model


def vit_base(patch_size=16, num_register_tokens=0, export=False, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        num_register_tokens=num_register_tokens,
        block_fn=partial(Block, attn_class=Attention if export else MemEffAttention),
        **kwargs,
    )
    return model


def vit_large(patch_size=16, num_register_tokens=0, export=False, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        num_register_tokens=num_register_tokens,
        block_fn=partial(Block, attn_class=Attention if export else MemEffAttention),
        **kwargs,
    )
    return model


def _make_dinov2_model_name(arch_name: str, patch_size: int) -> str:
    compact_arch_name = arch_name.replace("_", "")[:4]
    return f"dinov2_{compact_arch_name}{patch_size}"


def _make_dinov2_model(
    *,
    arch_name: str = "vit_large",
    img_size: int = 518,
    patch_size: int = 14,
    init_values: float = 1.0,
    ffn_layer: str = "mlp",
    block_chunks: int = 0,
    pretrained: str = "",
    output_idx: Sequence[int] = [],
    num_register_tokens: int = 0,
    drop_path_rate: float = 0.0,
    use_norm: bool = False,
    export: bool = False,
    **kwargs,
):
    model_name = _make_dinov2_model_name(arch_name, patch_size)

    vit_kwargs = dict(
        img_size=img_size,
        patch_size=patch_size,
        init_values=init_values,
        ffn_layer=ffn_layer,
        block_chunks=block_chunks,
        output_idx=output_idx,
        drop_path_rate=drop_path_rate,
        num_register_tokens=num_register_tokens,
        use_norm=use_norm,
        export=export,
    )
    vit_kwargs.update(**kwargs)
    model = eval(arch_name)(**vit_kwargs)
    if pretrained == "":
        url = _DINOV2_BASE_URL + f"/{model_name}/{model_name}"
        if num_register_tokens > 0:
            url += "_reg4"
        url += "_pretrain.pth"
        state_dict = torch.hub.load_state_dict_from_url(
            url, map_location="cpu", progress=False
        )
        info = model.load_state_dict(state_dict, strict=False)
        print(info)
    elif pretrained is not None:
        state_dict = torch.load(pretrained, map_location="cpu")
        info = model.load_state_dict(state_dict, strict=False)
        print(f"loading from {pretrained} with:", info)
    return model
