"""
Author: Luigi Piccinelli
Licensed under the CC-BY NC 4.0 license (http://creativecommons.org/licenses/by-nc/4.0/)
"""

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import trunc_normal_

from unidepth.layers import (MLP, AttentionBlock, ConvUpsample, NystromBlock,
                             PositionEmbeddingSine)
from unidepth.utils.geometric import flat_interpolate, generate_rays
from unidepth.utils.misc import max_stack
from unidepth.utils.sht import rsh_cart_8


class ListAdapter(nn.Module):
    def __init__(self, input_dims: List[int], hidden_dim: int):
        super().__init__()
        self.input_adapters = nn.ModuleList([])
        self.num_chunks = len(input_dims)
        for input_dim in input_dims:
            self.input_adapters.append(
                nn.Sequential(
                    nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim), nn.GELU()
                )
            )

    def forward(self, x: torch.Tensor, splits: torch.Tensor) -> torch.Tensor:
        xs = torch.split(x, splits.int().tolist(), dim=-1)
        xs = [adapter(x) for x, adapter in zip(xs, self.input_adapters)]
        return torch.cat(xs, dim=-1)


class CameraHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_heads: int = 8,
        expansion: int = 4,
        depth: int = 4,
        dropout: float = 0.0,
        layer_scale: float = 1.0,
        **kwargs,
    ):
        super().__init__()

        self.aggregate = AttentionBlock(
            hidden_dim,
            num_heads=1,
            expansion=expansion,
            dropout=dropout,
            layer_scale=layer_scale,
        )
        self.latents_pos = nn.Parameter(
            torch.randn(1, 4, hidden_dim), requires_grad=True
        )
        self.layers = nn.ModuleList([])
        self.in_features = MLP(hidden_dim, expansion=2, dropout=dropout)
        for _ in range(depth):
            blk = AttentionBlock(
                hidden_dim,
                num_heads=num_heads,
                expansion=expansion,
                dropout=dropout,
                layer_scale=layer_scale,
            )
            self.layers.append(blk)
        self.out = MLP(hidden_dim, expansion=2, dropout=0.0, output_dim=1)
        self.cls_project = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
        )

    def forward(self, features, cls_tokens, pos_embed) -> torch.Tensor:
        features = features.unbind(dim=-1)
        cls_tokens = self.cls_project(cls_tokens)
        features_stack = torch.cat(features, dim=1)
        features_stack = features_stack + pos_embed
        latents_pos = self.latents_pos.expand(cls_tokens.shape[0], -1, -1)
        features_stack = self.in_features(features_stack)
        features = torch.cat((features_stack, cls_tokens), dim=1)
        cls_tokens = self.aggregate(cls_tokens, context=features, pos_embed=latents_pos)
        for i, layer in enumerate(self.layers):
            cls_tokens = layer(cls_tokens, pos_embed=latents_pos)

        # project
        x = self.out(cls_tokens).squeeze(-1)
        camera_intrinsics = torch.zeros(
            x.shape[0], 3, 3, device=x.device, requires_grad=False
        )
        camera_intrinsics[:, 0, 0] = x[:, 0].exp()
        camera_intrinsics[:, 1, 1] = x[:, 1].exp()
        camera_intrinsics[:, 0, 2] = x[:, 2].sigmoid()
        camera_intrinsics[:, 1, 2] = x[:, 3].sigmoid()
        camera_intrinsics[:, 2, 2] = 1.0
        return camera_intrinsics

    def set_shapes(self, shapes: Tuple[int, int]):
        self.shapes = shapes


class DepthHead(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        expansion: int = 4,
        depths: int | list[int] = 4,
        camera_dim: int = 256,
        num_resolutions: int = 4,
        dropout: float = 0.0,
        layer_scale: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__()
        if isinstance(depths, int):
            depths = [depths] * 3
        assert len(depths) == 3

        self.project_rays16 = MLP(
            camera_dim, expansion=expansion, dropout=dropout, output_dim=hidden_dim
        )
        self.project_rays8 = MLP(
            camera_dim, expansion=expansion, dropout=dropout, output_dim=hidden_dim // 2
        )
        self.project_rays4 = MLP(
            camera_dim, expansion=expansion, dropout=dropout, output_dim=hidden_dim // 4
        )
        self.to_latents = MLP(hidden_dim, expansion=2, dropout=dropout)

        self.features_channel_cat = nn.Linear(hidden_dim * num_resolutions, hidden_dim)

        self.up8 = ConvUpsample(
            hidden_dim, expansion=expansion, layer_scale=layer_scale
        )
        self.up4 = ConvUpsample(
            hidden_dim // 2, expansion=expansion, layer_scale=layer_scale
        )
        self.up2 = ConvUpsample(
            hidden_dim // 4, expansion=expansion, layer_scale=layer_scale
        )

        self.layers_16 = nn.ModuleList([])
        self.layers_8 = nn.ModuleList([])
        self.layers_4 = nn.ModuleList([])
        self.aggregate_16 = AttentionBlock(
            hidden_dim,
            num_heads=1,
            expansion=expansion,
            dropout=dropout,
            layer_scale=layer_scale,
            context_dim=hidden_dim,
        )
        self.prompt_camera = AttentionBlock(
            hidden_dim,
            num_heads=1,
            expansion=expansion,
            dropout=dropout,
            layer_scale=layer_scale,
            context_dim=hidden_dim,
        )
        for i, (blk_lst, depth) in enumerate(
            zip([self.layers_16, self.layers_8, self.layers_4], depths)
        ):
            attn_cls = AttentionBlock if i == 0 else NystromBlock
            for _ in range(depth):
                blk_lst.append(
                    attn_cls(
                        hidden_dim // (2**i),
                        num_heads=num_heads // (2**i),
                        expansion=expansion,
                        dropout=dropout,
                        layer_scale=layer_scale,
                    )
                )

        self.out2 = nn.Conv2d(hidden_dim // 8, 1, 3, padding=1)
        self.out4 = nn.Conv2d(hidden_dim // 4, 1, 3, padding=1)
        self.out8 = nn.Conv2d(hidden_dim // 2, 1, 3, padding=1)

    def set_original_shapes(self, shapes: Tuple[int, int]):
        self.original_shapes = shapes

    def set_shapes(self, shapes: Tuple[int, int]):
        self.shapes = shapes

    def forward(
        self, features: torch.Tensor, rays_hr: torch.Tensor, pos_embed, level_embed
    ) -> torch.Tensor:
        features = features.unbind(dim=-1)
        shapes = self.shapes
        rays_hr = rays_hr.detach()

        # camera_embedding
        rays_embedding_16 = F.normalize(
            flat_interpolate(rays_hr, old=self.original_shapes, new=shapes), dim=-1
        )
        rays_embedding_8 = F.normalize(
            flat_interpolate(
                rays_hr, old=self.original_shapes, new=[x * 2 for x in shapes]
            ),
            dim=-1,
        )
        rays_embedding_4 = F.normalize(
            flat_interpolate(
                rays_hr, old=self.original_shapes, new=[x * 4 for x in shapes]
            ),
            dim=-1,
        )
        rays_embedding_16 = self.project_rays16(rsh_cart_8(rays_embedding_16))
        rays_embedding_8 = self.project_rays8(rsh_cart_8(rays_embedding_8))
        rays_embedding_4 = self.project_rays4(rsh_cart_8(rays_embedding_4))
        features_tokens = torch.cat(features, dim=1)
        features_tokens_pos = pos_embed + level_embed

        # Generate latents with init as pooled features
        features_channels = torch.cat(features, dim=-1)
        features_16 = self.features_channel_cat(features_channels)
        latents_16 = self.to_latents(
            flat_interpolate(features_16, old=self.shapes, new=shapes, antialias=False)
        )

        # Aggregate features: F -> D
        latents_16 = self.aggregate_16(
            latents_16, context=features_tokens, pos_embed_context=features_tokens_pos
        )

        # Aggregate camera: D- > D|E
        latents_16 = self.prompt_camera(latents_16, context=rays_embedding_16)

        # Block 16 - Out 8
        for layer in self.layers_16:
            latents_16 = layer(latents_16, pos_embed=rays_embedding_16)
        latents_8 = self.up8(
            rearrange(
                latents_16 + rays_embedding_16,
                "b (h w) c -> b c h w",
                h=shapes[0],
                w=shapes[1],
            ).contiguous()
        )
        out8 = self.out8(
            rearrange(
                latents_8, "b (h w) c -> b c h w", h=shapes[0] * 2, w=shapes[1] * 2
            )
        )

        # Block 8 - Out 4
        for layer in self.layers_8:
            latents_8 = layer(latents_8, pos_embed=rays_embedding_8)
        latents_4 = self.up4(
            rearrange(
                latents_8 + rays_embedding_8,
                "b (h w) c -> b c h w",
                h=shapes[0] * 2,
                w=shapes[1] * 2,
            ).contiguous()
        )
        out4 = self.out4(
            rearrange(
                latents_4, "b (h w) c -> b c h w", h=shapes[0] * 4, w=shapes[1] * 4
            )
        )

        # Block 4 - Out 2
        for layer in self.layers_4:
            latents_4 = layer(latents_4, pos_embed=rays_embedding_4)
        latents_2 = self.up2(
            rearrange(
                latents_4 + rays_embedding_4,
                "b (h w) c -> b c h w",
                h=shapes[0] * 4,
                w=shapes[1] * 4,
            ).contiguous()
        )
        out2 = self.out2(
            rearrange(
                latents_2, "b (h w) c -> b c h w", h=shapes[0] * 8, w=shapes[1] * 8
            )
        )

        # Depth features
        proj_latents_16 = rearrange(
            latents_16, "b (h w) c -> b c h w", h=shapes[0], w=shapes[1]
        ).contiguous()

        # MS Outputs
        out2 = out2.clamp(-10.0, 10.0).exp()
        out4 = out4.clamp(-10.0, 10.0).exp()
        out8 = out8.clamp(-10.0, 10.0).exp()

        return out8, out4, out2, proj_latents_16


class Decoder(nn.Module):
    def __init__(
        self,
        config,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.build(config)
        self.apply(self._init_weights)
        self.test_fixed_camera = False
        self.skip_camera = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_adapted_features(self, features_flat, splits):
        features_flat_cat = torch.cat(features_flat, dim=-1)
        features_projected = self.input_adapter(
            features_flat_cat, splits
        )  # list [b hw c] shapes
        features = torch.chunk(features_projected, len(splits), dim=-1)
        return features

    def run_camera(self, cls_tokens, features, pos_embed, original_shapes, rays):
        # get cls tokens projections
        cls_tokens_splits = torch.tensor(
            [x.shape[-1] for x in cls_tokens],
            device=features.device,
            requires_grad=False,
            dtype=features.dtype,
        )
        cls_tokens = torch.cat(cls_tokens, dim=-1)
        cls_tokens = self.token_adapter(cls_tokens, cls_tokens_splits)
        cls_tokens = torch.cat(
            torch.chunk(cls_tokens, len(cls_tokens_splits), dim=-1), dim=1
        )

        # camera layer
        intrinsics = self.camera_layer(
            features=features, cls_tokens=cls_tokens, pos_embed=pos_embed
        )
        intrinsics[:, 0, 0] = max(original_shapes) / 2 * intrinsics[:, 0, 0]
        intrinsics[:, 1, 1] = max(original_shapes) / 2 * intrinsics[:, 1, 1]
        intrinsics[:, 0, 2] = intrinsics[:, 0, 2] * original_shapes[1]
        intrinsics[:, 1, 2] = intrinsics[:, 1, 2] * original_shapes[0]
        if not self.test_fixed_camera:
            rays, _ = generate_rays(intrinsics, original_shapes, noisy=False)

        return intrinsics, rays

    def forward(self, inputs, image_metas) -> torch.Tensor:
        B, _, H, W = inputs["image"].shape
        device = inputs["image"].device

        # make stride happy?
        original_encoder_outputs = [x.contiguous() for x in inputs["encoder_outputs"]]
        cls_tokens = [x.contiguous() for x in inputs["cls_tokens"]]

        # collect features and tokens
        original_encoder_outputs = [
            max_stack(original_encoder_outputs[i:j])
            for i, j in self.slices_encoder_range
        ]
        # detach tokens for camera
        cls_tokens = [
            cls_tokens[-i - 1].detach() for i in range(len(self.slices_encoder_range))
        ]

        # get features in b n d format
        # level shapes, the shape per level, for swin like [[128, 128], [64, 64],...], for vit [[32,32]] -> mult times resolutions
        resolutions = [
            tuple(sorted([x.shape[1], x.shape[2]])) for x in original_encoder_outputs
        ]
        level_shapes = sorted(list(set(resolutions)))[::-1]

        if len(level_shapes) == 1:
            level_shapes = level_shapes * self.num_resolutions
        input_shapes = [
            level_shapes[i]
            for i, (start, end) in enumerate(self.slices_encoder)
            for _ in range(end - start)
        ]
        common_shape = level_shapes[-2]

        # input shapes repeat shapes for each level, times the amount of the layers:
        features_flat = [
            flat_interpolate(
                rearrange(x, "b h w c -> b (h w) c"), old=input_shape, new=common_shape
            )
            for x, input_shape in zip(original_encoder_outputs, input_shapes)
        ]
        features_splits = torch.tensor(
            [x.shape[-1] for x in features_flat],
            device=device,
            requires_grad=False,
            dtype=torch.float32,
        )

        # input adapter, then do mean of features in same blocks
        features = self.get_adapted_features(features_flat, features_splits)
        features = torch.stack(features, dim=-1)

        # positional embeddings, spatial and level
        level_embed = torch.cat(
            [
                self.level_embed_layer(self.level_embeds)[i : i + 1]
                .unsqueeze(0)
                .repeat(B, common_shape[0] * common_shape[1], 1)
                for i in range(self.num_resolutions)
            ],
            dim=1,
        )
        pos_embed = self.pos_embed(
            torch.zeros(
                B,
                1,
                common_shape[0],
                common_shape[1],
                device=device,
                requires_grad=False,
            )
        )
        pos_embed = rearrange(pos_embed, "b c h w -> b (h w) c").repeat(
            1, self.num_resolutions, 1
        )

        self.camera_layer.set_shapes(common_shape)
        intrinsics, rays = (
            self.run_camera(
                cls_tokens,
                features=features,
                pos_embed=pos_embed + level_embed,
                original_shapes=(H, W),
                rays=inputs.get("rays", None),
            )
            if not self.skip_camera
            else (inputs["K"], inputs["rays"])
        )

        # run bulk of the model
        self.depth_layer.set_shapes(common_shape)
        self.depth_layer.set_original_shapes((H, W))
        out8, out4, out2, depth_features = self.depth_layer(
            features=features,
            rays_hr=rays,
            pos_embed=pos_embed,
            level_embed=level_embed,
        )

        return intrinsics, [out8, out4, out2], depth_features

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"latents_pos", "level_embeds"}

    def build(self, config):
        depth = config["model"]["pixel_decoder"]["depths"]
        input_dims = config["model"]["pixel_encoder"]["embed_dims"]
        hidden_dim = config["model"]["pixel_decoder"]["hidden_dim"]
        num_heads = config["model"]["num_heads"]
        expansion = config["model"]["expansion"]
        dropout = config["model"]["pixel_decoder"]["dropout"]
        depths_encoder = config["model"]["pixel_encoder"]["depths"]
        layer_scale = 1.0

        self.depth = depth
        self.dim = hidden_dim
        self.downsample = 4
        self.num_heads = num_heads
        self.num_resolutions = len(depths_encoder)
        self.depths_encoder = depths_encoder

        self.slices_encoder_single = list(
            zip([d - 1 for d in self.depths_encoder], self.depths_encoder)
        )
        self.slices_encoder_range = list(
            zip([0, *self.depths_encoder[:-1]], self.depths_encoder)
        )
        cls_token_input_dims = [input_dims[-i - 1] for i in range(len(depths_encoder))]

        input_dims = [input_dims[d - 1] for d in depths_encoder]
        self.slices_encoder = self.slices_encoder_single

        # adapt from encoder features, just project
        self.input_adapter = ListAdapter(input_dims, hidden_dim)
        self.token_adapter = ListAdapter(cls_token_input_dims, hidden_dim)

        # camera layer
        self.camera_layer = CameraHead(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            expansion=expansion,
            depth=2,
            dropout=dropout,
            layer_scale=layer_scale,
        )

        self.depth_layer = DepthHead(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            expansion=expansion,
            depths=depth,
            dropout=dropout,
            camera_dim=81,
            num_resolutions=self.num_resolutions,
            layer_scale=layer_scale,
        )

        # transformer part
        self.pos_embed = PositionEmbeddingSine(hidden_dim // 2, normalize=True)
        self.level_embeds = nn.Parameter(
            torch.randn(len(input_dims), hidden_dim), requires_grad=True
        )
        self.level_embed_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
