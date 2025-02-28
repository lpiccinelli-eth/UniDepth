"""
Author: Luigi Piccinelli
Licensed under the CC-BY NC 4.0 license (http://creativecommons.org/licenses/by-nc/4.0/)
"""

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .layer_scale import LayerScale
from .mlp import MLP


class SimpleAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        dropout: float = 0.0,
        cosine: bool = False,
        context_dim: int | None = None,
    ):
        super().__init__()
        self.dropout = dropout
        self.num_heads = num_heads
        self.hidden_dim = dim
        context_dim = context_dim or dim

        self.kv = nn.Linear(context_dim, dim * 2, bias=False)
        self.q = nn.Linear(dim, dim, bias=False)
        self.norm_attnx = nn.LayerNorm(dim)
        self.norm_attnctx = nn.LayerNorm(context_dim)
        self.cosine = cosine
        self.out = nn.Linear(dim, dim)

    def forward(
        self,
        x: torch.Tensor,
        attn_bias: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
        pos_embed: torch.Tensor | None = None,
        pos_embed_context: torch.Tensor | None = None,
        rope: nn.Module | None = None,
    ) -> torch.Tensor:
        context = x if context is None else context
        x = self.norm_attnx(x)
        context = self.norm_attnctx(context)
        k, v = rearrange(
            self.kv(context), "b n (kv h d) -> b h n d kv", h=self.num_heads, kv=2
        ).unbind(dim=-1)
        q = rearrange(self.q(x), "b n (h d) -> b h n d", h=self.num_heads)

        if rope is not None:
            q = rope(q)
            k = rope(k)
        else:
            if pos_embed is not None:
                pos_embed = rearrange(
                    pos_embed, "b n (h d) -> b h n d", h=self.num_heads
                )
                q = q + pos_embed
            if pos_embed_context is not None:
                pos_embed_context = rearrange(
                    pos_embed_context, "b n (h d) -> b h n d", h=self.num_heads
                )
                k = k + pos_embed_context

        if self.cosine:
            q, k = map(partial(F.normalize, p=2, dim=-1), (q, k))  # cosine sim
        x = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout, attn_mask=attn_bias
        )
        x = rearrange(x, "b h n d -> b n (h d)")
        x = self.out(x)
        return x


class AttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        expansion: int = 4,
        dropout: float = 0.0,
        cosine: bool = False,
        gated: bool = False,
        layer_scale: float = 1.0,
        context_dim: int | None = None,
        use_bias: bool = True,
    ):
        super().__init__()
        self.dropout = dropout
        self.num_heads = num_heads
        self.hidden_dim = dim
        context_dim = context_dim or dim
        self.mlp = MLP(dim, expansion=expansion, dropout=dropout, gated=gated)
        self.kv = nn.Linear(context_dim, dim * 2, bias=use_bias)
        self.q = nn.Linear(dim, dim, bias=use_bias)
        self.norm_attnx = nn.LayerNorm(dim)
        self.norm_attnctx = nn.LayerNorm(context_dim)
        self.cosine = cosine
        self.out = nn.Linear(dim, dim, bias=use_bias)
        self.ls1 = LayerScale(dim, layer_scale) if layer_scale > 0.0 else nn.Identity()
        self.ls2 = LayerScale(dim, layer_scale) if layer_scale > 0.0 else nn.Identity()

    def attn(
        self,
        x: torch.Tensor,
        attn_bias: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
        pos_embed: torch.Tensor | None = None,
        pos_embed_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.norm_attnx(x)
        context = self.norm_attnctx(context)
        k, v = rearrange(
            self.kv(context), "b n (kv h d) -> b h n d kv", h=self.num_heads, kv=2
        ).unbind(dim=-1)
        q = rearrange(self.q(x), "b n (h d) -> b h n d", h=self.num_heads)

        if pos_embed is not None:
            pos_embed = rearrange(pos_embed, "b n (h d) -> b h n d", h=self.num_heads)
            q = q + pos_embed
        if pos_embed_context is not None:
            pos_embed_context = rearrange(
                pos_embed_context, "b n (h d) -> b h n d", h=self.num_heads
            )
            k = k + pos_embed_context

        if self.cosine:
            q, k = map(partial(F.normalize, p=2, dim=-1), (q, k))  # cosine sim

        x = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout, attn_mask=attn_bias
        )
        x = rearrange(x, "b h n d -> b n (h d)")
        x = self.out(x)
        return x

    def forward(
        self,
        x: torch.Tensor,
        attn_bias: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
        pos_embed: torch.Tensor | None = None,
        pos_embed_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        context = x if context is None else context
        x = (
            self.ls1(
                self.attn(
                    x,
                    attn_bias=attn_bias,
                    context=context,
                    pos_embed=pos_embed,
                    pos_embed_context=pos_embed_context,
                )
            )
            + x
        )
        x = self.ls2(self.mlp(x)) + x
        return x


class AttentionLayer(nn.Module):
    def __init__(
        self,
        num_blocks: int,
        dim: int,
        num_heads: int = 4,
        expansion: int = 4,
        dropout: float = 0.0,
        cosine: bool = False,
        gated: bool = False,
        layer_scale: float = 1.0,
        context_dim: int | None = None,
        use_bias: bool = True,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                AttentionBlock(
                    dim=dim,
                    num_heads=num_heads,
                    expansion=expansion,
                    dropout=dropout,
                    cosine=cosine,
                    gated=gated,
                    layer_scale=layer_scale,
                    context_dim=context_dim,
                    use_bias=use_bias,
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        pos_embed: torch.Tensor | None = None,
        pos_embed_context: torch.Tensor | None = None,
        attn_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(
                x,
                context=context,
                pos_embed=pos_embed,
                pos_embed_context=pos_embed_context,
                attn_bias=attn_bias,
            )
        return x


class AttentionDecoderBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        expansion: int = 4,
        dropout: float = 0.0,
        cosine: bool = False,
        gated: bool = False,
        layer_scale: float = 1.0,
        context_dim: int | None = None,
        single_head_ca: bool = True,
    ):
        super().__init__()
        self.dropout = dropout
        self.num_heads = num_heads
        self.hidden_dim = dim
        self.single_head_ca = single_head_ca
        context_dim = context_dim or dim
        self.mlp = MLP(dim, expansion=expansion, dropout=dropout, gated=gated)
        self.kv_ca = nn.Linear(context_dim, dim * 2)
        self.q_ca = nn.Linear(dim, dim)
        self.kv_sa = nn.Linear(dim, dim * 2)
        self.q_sa = nn.Linear(dim, dim)
        self.norm_x_sa = nn.LayerNorm(dim)
        self.norm_x_ca = nn.LayerNorm(dim)
        self.norm_ctx_ca = nn.LayerNorm(context_dim)
        self.cosine = cosine
        self.out_ca = nn.Linear(dim, dim)
        self.out_sa = nn.Linear(dim, dim)
        self.ls1 = LayerScale(dim, layer_scale) if layer_scale > 0.0 else nn.Identity()
        self.ls2 = LayerScale(dim, layer_scale) if layer_scale > 0.0 else nn.Identity()
        self.ls3 = LayerScale(dim, layer_scale) if layer_scale > 0.0 else nn.Identity()

    def cross_attn(
        self,
        x: torch.Tensor,
        attn_bias: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
        pos_embed: torch.Tensor | None = None,
        pos_embed_context: torch.Tensor | None = None,
        rope: nn.Module | None = None,
    ) -> torch.Tensor:
        num_heads = 1 if self.single_head_ca else self.num_heads
        x = self.norm_x_ca(x)
        context = self.norm_ctx_ca(context)
        k, v = rearrange(
            self.kv_ca(context), "b n (kv h d) -> b h n d kv", h=num_heads, kv=2
        ).unbind(dim=-1)
        q = rearrange(self.q_ca(x), "b n (h d) -> b h n d", h=num_heads)

        if rope is not None:
            q = rope(q)
            k = rope(k)
        else:
            if pos_embed is not None:
                pos_embed = rearrange(pos_embed, "b n (h d) -> b h n d", h=num_heads)
                q = q + pos_embed
            if pos_embed_context is not None:
                pos_embed_context = rearrange(
                    pos_embed_context, "b n (h d) -> b h n d", h=num_heads
                )
                k = k + pos_embed_context

        if self.cosine:
            q, k = map(partial(F.normalize, p=2, dim=-1), (q, k))  # cosine sim
        x = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout, attn_mask=attn_bias
        )
        x = rearrange(x, "b h n d -> b n (h d)")
        x = self.out_ca(x)
        return x

    def self_attn(
        self,
        x: torch.Tensor,
        attn_bias: torch.Tensor | None = None,
        pos_embed: torch.Tensor | None = None,
        rope: nn.Module | None = None,
    ) -> torch.Tensor:
        x = self.norm_x_sa(x)
        k, v = rearrange(
            self.kv_sa(x), "b n (kv h d) -> b h n d kv", h=self.num_heads, kv=2
        ).unbind(dim=-1)
        q = rearrange(self.q_sa(x), "b n (h d) -> b h n d", h=self.num_heads)

        if rope is not None:
            q = rope(q)
            k = rope(k)
        elif pos_embed is not None:
            pos_embed = rearrange(pos_embed, "b n (h d) -> b h n d", h=self.num_heads)
            q = q + pos_embed

        if self.cosine:
            q, k = map(partial(F.normalize, p=2, dim=-1), (q, k))  # cosine sim
        x = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout, attn_mask=attn_bias
        )
        x = rearrange(x, "b h n d -> b n (h d)")
        x = self.out_sa(x)
        return x

    def forward(
        self,
        x: torch.Tensor,
        attn_bias: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
        pos_embed: torch.Tensor | None = None,
        pos_embed_context: torch.Tensor | None = None,
        rope: nn.Module | None = None,
    ) -> torch.Tensor:
        context = x if context is None else context
        x = (
            self.ls1(
                self.cross_attn(
                    x,
                    rope=rope,
                    attn_bias=attn_bias,
                    context=context,
                    pos_embed=pos_embed,
                    pos_embed_context=pos_embed_context,
                )
            )
            + x
        )
        x = (
            self.ls2(
                self.self_attn(x, rope=rope, attn_bias=attn_bias, pos_embed=pos_embed)
            )
            + x
        )
        x = self.ls3(self.mlp(x)) + x
        return x
