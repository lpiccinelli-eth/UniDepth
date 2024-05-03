"""
Author: Luigi Piccinelli
Licensed under the CC-BY NC 4.0 license (http://creativecommons.org/licenses/by-nc/4.0/)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from timm.models.layers import trunc_normal_

from unidepth.layers import (
    MLP,
    AttentionBlock,
    NystromBlock,
    PositionEmbeddingSine,
    ConvUpsampleShuffle,
)
from unidepth.utils.positional_embedding import generate_fourier_features
from unidepth.utils.geometric import generate_rays, flat_interpolate


class ListAdapter(nn.Module):
    def __init__(self, input_dims: list[int], hidden_dim: int):
        super().__init__()
        self.input_adapters = nn.ModuleList([])
        self.num_chunks = len(input_dims)
        self.checkpoint = True
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
        hidden_dim: int,
        expansion: int = 4,
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.aggregate1 = AttentionBlock(
            hidden_dim, num_heads=1, expansion=expansion, dropout=dropout
        )
        self.aggregate2 = AttentionBlock(
            hidden_dim, num_heads=1, expansion=expansion, dropout=dropout
        )
        self.latents_pos = nn.Parameter(
            torch.randn(1, 4, hidden_dim), requires_grad=True
        )
        self.in_features = MLP(hidden_dim, expansion=2, dropout=dropout)
        self.project_cls = MLP(hidden_dim, dropout=dropout)
        self.out = MLP(hidden_dim, expansion=2, dropout=0.0, output_dim=1)

    def fill_intrinsics(self, x):
        camera_intrinsics = torch.zeros(
            x.shape[0], 3, 3, device=x.device, requires_grad=False
        )
        camera_intrinsics[:, 0, 0] = x[:, 0].exp()
        camera_intrinsics[:, 1, 1] = x[:, 1].exp()
        camera_intrinsics[:, 0, 2] = x[:, 2].sigmoid()
        camera_intrinsics[:, 1, 2] = x[:, 3].sigmoid()
        camera_intrinsics[:, 2, 2] = 1.0
        return camera_intrinsics

    def forward(self, features, cls_tokens, pos_embed) -> torch.Tensor:
        features = features.unbind(dim=-1)
        cls_tokens = self.project_cls(cls_tokens)
        latents_pos = self.latents_pos.expand(cls_tokens.shape[0], -1, -1)
        features = self.in_features(torch.cat(features, dim=1) + pos_embed)
        features = torch.cat((features, cls_tokens), dim=1)
        cls_tokens = self.aggregate1(
            cls_tokens, context=features, pos_embed=latents_pos
        )
        cls_tokens = self.aggregate2(
            cls_tokens, context=features, pos_embed=latents_pos
        )

        # project to intrinsics
        x = self.out(cls_tokens).squeeze(-1)
        camera_intrinsics = self.fill_intrinsics(x)

        return camera_intrinsics

    def set_shapes(self, shapes: tuple[int, int]):
        self.shapes = shapes


class GlobalHead(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        camera_dim: int,
        expansion: int = 4,
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.camera_dim = camera_dim
        self.in_features = nn.Linear(hidden_dim, hidden_dim)
        self.project_rays = nn.Linear(camera_dim + 3, hidden_dim)
        self.aggregate1 = AttentionBlock(
            hidden_dim, num_heads=1, expansion=expansion, dropout=dropout
        )
        self.aggregate2 = AttentionBlock(
            hidden_dim, num_heads=1, expansion=expansion, dropout=dropout
        )
        self.project_cls = MLP(hidden_dim, dropout=dropout)
        self.out = MLP(hidden_dim, expansion=2, dropout=0.0, output_dim=1)

    def embed_rays(self, rays, shapes):
        rays_embedding = flat_interpolate(rays, old=self.original_shapes, new=shapes)
        rays_embedding = F.normalize(rays_embedding, dim=-1)
        rays_embedding = generate_fourier_features(
            rays_embedding,
            dim=self.camera_dim,
            max_freq=max(shapes) // 2,
            use_log=True,
            cat_orig=True,
        )
        return rays_embedding

    def set_original_shapes(self, shapes: tuple[int, int]):
        self.original_shapes = shapes

    def set_shapes(self, shapes: tuple[int, int]):
        self.shapes = shapes

    def get_scaleshift(self, x):
        scale, shift = torch.chunk(x, 2, dim=1)
        scale = scale.exp().reshape(-1, 1, 1, 1)
        shift = shift.reshape(-1, 1, 1, 1)
        return scale, shift

    def forward(self, features, cls_tokens, rays) -> torch.Tensor:
        features = features.unbind(dim=-1)
        cls_tokens = self.project_cls(cls_tokens)
        rays_embedding = self.project_rays(self.embed_rays(rays, self.shapes))
        rays_embedding = rays_embedding.repeat(1, len(features), 1)
        features = self.in_features(torch.cat(features, dim=1) + rays_embedding)
        features = torch.cat((features, cls_tokens), dim=1)
        cls_tokens = self.aggregate1(cls_tokens, context=features)
        cls_tokens = self.aggregate2(cls_tokens, context=features)
        x = self.out(cls_tokens).squeeze(-1)
        scale, shift = self.get_scaleshift(x)
        return scale, shift


class DepthHead(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        expansion: int = 4,
        depths: int | list[int] = 4,
        checkpoint: bool = True,
        camera_dim: int = 256,
        num_resolutions: int = 4,
        dropout: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__()
        self.checkpoint = checkpoint
        self.camera_dim = camera_dim
        self.skip_depth = False

        self.to_latents = MLP(hidden_dim, expansion=2, dropout=dropout)
        self.features_channel_cat = nn.Linear(hidden_dim * num_resolutions, hidden_dim)
        self.aggregate_16 = AttentionBlock(
            hidden_dim,
            num_heads=1,
            expansion=expansion,
            dropout=dropout,
            context_dim=hidden_dim,
        )
        self.prompt_camera = AttentionBlock(
            hidden_dim,
            num_heads=1,
            expansion=expansion,
            dropout=dropout,
            context_dim=hidden_dim,
        )

        self.rays_layers = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        self.process_layers = nn.ModuleList([])
        self.norms, self.out_layers = nn.ModuleList([]), nn.ModuleList([])
        self.confidence_norms, self.confidence_out_layers = nn.ModuleList(
            []
        ), nn.ModuleList([])
        for i, depth in enumerate(depths):
            blk_lst = nn.ModuleList([])
            for _ in range(depth):
                blk_lst.append(
                    NystromBlock(
                        hidden_dim // int(2**i),
                        num_heads=num_heads // int(2**i),
                        expansion=expansion,
                        dropout=dropout,
                    )
                )
            self.process_layers.append(blk_lst)
            self.rays_layers.append(nn.Linear(camera_dim + 3, hidden_dim // int(2**i)))
            self.ups.append(
                ConvUpsampleShuffle(
                    hidden_dim // int(2**i),
                    expansion=expansion,
                    kernel_size=7,
                    num_layers=2,
                )
            )
            self.norms.append(nn.LayerNorm(hidden_dim // int(2 ** (i + 1))))
            self.out_layers.append(
                nn.Conv2d(hidden_dim // int(2 ** (i + 1)), 1, 3, padding=1)
            )
            self.confidence_norms.append(nn.LayerNorm(hidden_dim // int(2 ** (i + 1))))
            self.confidence_out_layers.append(
                nn.Conv2d(hidden_dim // int(2 ** (i + 1)), 1, 3, padding=1)
            )

    def set_original_shapes(self, shapes: tuple[int, int]):
        self.original_shapes = shapes

    def set_shapes(self, shapes: tuple[int, int]):
        self.shapes = shapes

    def embed_rays(self, rays, shapes):
        rays_embedding = flat_interpolate(rays, old=self.original_shapes, new=shapes)
        rays_embedding = F.normalize(rays_embedding, dim=-1)
        rays_embedding = generate_fourier_features(
            rays_embedding,
            dim=self.camera_dim,
            max_freq=max(shapes) // 2,
            use_log=True,
            cat_orig=True,
        )
        return rays_embedding

    def project_rays(self, rays, shapes):
        embedded_rays = []
        for i, layer in enumerate(self.rays_layers):
            embedded_rays.append(
                layer(self.embed_rays(rays, [(2**i) * x for x in shapes]))
            )
        return embedded_rays

    def decode_depth(self, latents_16, rays, shapes):
        dtype = latents_16.dtype
        latents = latents_16
        out_features, confidences, outs = [], [], []
        for i, (up, layers, rays_embedding) in enumerate(
            zip(self.ups, self.process_layers, rays)
        ):
            for layer in layers:
                latents = layer(latents, pos_embed=rays_embedding)
            latents = up(
                rearrange(
                    latents + rays_embedding,
                    "b (h w) c -> b c h w",
                    h=shapes[0] * int(2**i),
                    w=shapes[1] * int(2**i),
                ).contiguous()
            )
            out = rearrange(
                latents,
                "b (h w) c -> b h w c",
                h=shapes[0] * int(2 ** (1 + i)),
                w=shapes[1] * int(2 ** (1 + i)),
            )
            out_features.append(out)

        for i, (norm, out_layer, features) in enumerate(
            zip(self.norms[::-1], self.out_layers[::-1], out_features[::-1])
        ):
            features = norm(features)
            out_d = out_layer(features.permute(0, 3, 1, 2))
            outs.append(out_d)
        out = sum(
            F.interpolate(
                x,
                size=outs[0].shape[-2:],
                mode="bilinear",
            )
            for x in outs
        )
        out = out / len(outs)
        # jit complains, fix as list (loose dyn input)
        out_shapes = [int(s) for s in out.shape[1:]]
        out = F.layer_norm(out.float(), out_shapes)
        out = out.clamp(-10.0, 10.0).exp().to(dtype, non_blocking=True)

        for i, (norm, out_layer, features) in enumerate(
            zip(
                self.confidence_norms[::-1],
                self.confidence_out_layers[::-1],
                out_features[::-1],
            )
        ):
            features = norm(features)
            out_c = out_layer(features.permute(0, 3, 1, 2))
            confidences.append(out_c)
        confidence = sum(
            F.interpolate(
                x,
                size=confidences[0].shape[-2:],
                mode="bilinear",
            )
            for x in confidences
        )
        confidence = confidence / len(confidences)
        confidence = torch.sigmoid(confidence)

        return out, confidence

    def init_latents(self, features, shapes):
        # Generate latents with init as pooled features
        features_channels = torch.cat(features, dim=-1)
        features_16 = self.features_channel_cat(features_channels)
        latents_16 = features_16 + self.to_latents(
            flat_interpolate(features_16, old=self.shapes, new=shapes, antialias=False)
        )
        return latents_16

    def forward(
        self, features: torch.Tensor, rays_hr: torch.Tensor, pos_embed, level_embed
    ) -> torch.Tensor:
        B = features.shape[0]
        features = features.unbind(dim=-1)
        shapes = self.shapes

        # camera_embedding
        rays_embeddings = self.project_rays(rays_hr, shapes)

        # Init latents
        init_latents_16 = self.init_latents(features, shapes)

        # Aggregate features: F -> D
        latents_16 = self.aggregate_16(
            init_latents_16,
            context=torch.cat(features, dim=1),
            pos_embed_context=pos_embed + level_embed,
        )

        # Aggregate camera: D -> D|E
        latents_16 = self.prompt_camera(latents_16, context=rays_embeddings[0])

        # Decode depth
        out = self.decode_depth(latents_16, rays_embeddings, shapes)

        return out


class Decoder(nn.Module):
    def __init__(
        self,
        config,
    ):
        super().__init__()
        self.build(config)
        self.apply(self._init_weights)

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
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    def get_adapted_features(self, features_flat, splits):
        features_flat_cat = torch.cat(features_flat, dim=-1)
        features_projected = self.input_adapter(
            features_flat_cat, splits
        )  # list [b hw c] shapes
        features = torch.chunk(features_projected, len(splits), dim=-1)
        return features

    def run_camera(self, cls_tokens, features, pos_embed, original_shapes, rays_gt):
        # get cls tokens projections
        cls_tokens_splits = torch.tensor(
            [x.shape[-1] for x in cls_tokens],
            device=features.device,
            requires_grad=False,
            dtype=features.dtype,
        )
        cls_tokens = torch.cat(cls_tokens, dim=-1)
        cls_tokens = self.camera_token_adapter(cls_tokens, cls_tokens_splits)
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

        rays = (
            rays_gt
            if rays_gt is not None
            else generate_rays(intrinsics, original_shapes)[0]
        )
        return intrinsics, rays

    def run_global(self, cls_tokens, features, rays):
        # get cls tokens projections
        cls_tokens_splits = torch.tensor(
            [x.shape[-1] for x in cls_tokens],
            device=features.device,
            requires_grad=False,
            dtype=torch.float32,
        )
        cls_tokens = torch.cat(cls_tokens, dim=-1)
        cls_tokens = self.global_token_adapter(cls_tokens, cls_tokens_splits)
        cls_tokens = torch.cat(
            torch.chunk(cls_tokens, len(cls_tokens_splits), dim=-1), dim=1
        )

        scale, shift = self.global_layer(
            features=features, rays=rays, cls_tokens=cls_tokens
        )

        return scale, shift

    def forward(self, inputs, image_metas) -> torch.Tensor:
        B, C, H, W = inputs["image"].shape
        device = inputs["image"].device

        # get features in b n d format
        # level shapes, the shape per level, for swin like [[128, 128], [64, 64],...], for vit [[32,32]] -> mult times resolutions
        level_shapes = sorted(
            list(set([tuple([x.shape[1], x.shape[2]]) for x in inputs["features"]]))
        )[::-1]
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
            for x, input_shape in zip(inputs["features"], input_shapes)
        ]
        features_splits = torch.tensor(
            [x.shape[-1] for x in features_flat],
            device=device,
            requires_grad=False,
            dtype=torch.float32,
        )
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
        dummy_tensor = torch.zeros(
            B, 1, common_shape[0], common_shape[1], device=device, requires_grad=False
        )
        pos_embed = self.pos_embed(dummy_tensor)
        pos_embed = rearrange(pos_embed, "b c h w -> b (h w) c").repeat(
            1, self.num_resolutions, 1
        )

        self.camera_layer.set_shapes(common_shape)
        intrinsics, rays = self.run_camera(
            inputs["camera_tokens"],
            features=features,
            pos_embed=pos_embed + level_embed,
            original_shapes=(H, W),
            rays_gt=inputs.get("rays"),
        )

        self.global_layer.set_shapes(common_shape)
        self.global_layer.set_original_shapes((H, W))
        scale, shift = self.run_global(
            inputs["global_tokens"], features=features, rays=rays
        )

        # run bulk of the model
        self.depth_layer.set_shapes(common_shape)
        self.depth_layer.set_original_shapes((H, W))
        out_normalized, confidence = self.depth_layer(
            features=features,
            rays_hr=rays,
            pos_embed=pos_embed,
            level_embed=level_embed,
        )
        # shift is scale invariant if we do (x + mu) * sigma
        out = (out_normalized + shift) * scale

        outputs = {
            "depth": out.clamp(min=1e-3),
            "depth_ssi": out_normalized,
            "confidence": confidence,
            "K": intrinsics,
            "scale_shift": (scale, shift),
            "rays": rays,
        }
        return outputs

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"latents_pos", "level_embeds"}

    def build(self, config):
        input_dims = config["model"]["pixel_encoder"]["embed_dims"]
        hidden_dim = config["model"]["pixel_decoder"]["hidden_dim"]
        expansion = config["model"]["expansion"]
        num_heads = config["model"]["num_heads"]
        dropout = config["model"]["pixel_decoder"]["dropout"]
        depths_encoder = config["model"]["pixel_encoder"]["depths"]
        depth = config["model"]["pixel_decoder"]["depths"]
        depths_encoder = config["model"]["pixel_encoder"]["depths"]
        self.downsample = 4
        self.num_resolutions = len(depths_encoder)

        self.slices_encoder = list(zip([d - 1 for d in depths_encoder], depths_encoder))
        cls_token_input_dims = [input_dims[i] for i in [-1, -2, -3, -4]]
        input_dims = [input_dims[d - 1] for d in depths_encoder]

        # # camera layer
        self.camera_layer = CameraHead(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            expansion=expansion,
            dropout=dropout,
        )

        # # scale shift layer
        self.global_layer = GlobalHead(
            hidden_dim=hidden_dim,
            camera_dim=96,
            num_heads=num_heads,
            expansion=expansion,
            dropout=dropout,
        )

        # # adapt from encoder features, just project
        self.input_adapter = ListAdapter(input_dims, hidden_dim)
        self.camera_token_adapter = ListAdapter(cls_token_input_dims, hidden_dim)
        self.global_token_adapter = ListAdapter(cls_token_input_dims[:2], hidden_dim)

        self.depth_layer = DepthHead(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            expansion=expansion,
            depths=depth,
            dropout=dropout,
            camera_dim=96,
            num_resolutions=self.num_resolutions,
        )

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
