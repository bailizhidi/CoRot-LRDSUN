"""Small permutation-equivariant Physics-Attention / Transolver-style core.

This is intentionally not a vendored Transolver project.  It provides only a
standard self-attention + residual MLP block with an explicit boolean
allow-mask.  No absolute token-index embedding is used.
"""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


class PhysicsAttention(nn.Module):
    """Multi-head self-attention over ``[N,d]`` or ``[B,N,d]`` tokens."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        dim = int(dim)
        num_heads = int(num_heads)
        if dim <= 0 or num_heads <= 0 or dim % num_heads:
            raise ValueError(f"dim={dim} must be positive and divisible by num_heads={num_heads}")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(float(dropout))

    def forward(
        self,
        tokens: torch.Tensor,
        attention_allow: torch.Tensor | None = None,
    ) -> torch.Tensor:
        squeeze = False
        if tokens.ndim == 2:
            tokens = tokens.unsqueeze(0)
            squeeze = True
        if tokens.ndim != 3 or tokens.shape[-1] != self.dim:
            raise ValueError(f"tokens must be [N,{self.dim}] or [B,N,{self.dim}], got {tuple(tokens.shape)}")
        b, n, _ = tokens.shape
        qkv = self.qkv(tokens).reshape(b, n, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if attention_allow is not None:
            mask = attention_allow.to(device=scores.device, dtype=torch.bool)
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            if mask.ndim != 3 or mask.shape[-2:] != (n, n):
                raise ValueError(
                    f"attention_allow must be [N,N] or [B,N,N], got {tuple(mask.shape)}"
                )
            if mask.shape[0] not in (1, b):
                raise ValueError(f"attention mask batch {mask.shape[0]} incompatible with B={b}")
            if mask.shape[0] == 1 and b != 1:
                mask = mask.expand(b, -1, -1)
            if not torch.all(mask.any(dim=-1)):
                raise ValueError("every token must attend to at least one allowed token")
            scores = scores.masked_fill(~mask[:, None, :, :], torch.finfo(scores.dtype).min)

        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        out = torch.matmul(weights, v)
        out = out.transpose(1, 2).reshape(b, n, self.dim)
        out = self.proj(out)
        out = self.dropout(out)
        return out.squeeze(0) if squeeze else out


class PhysicsAttentionBlock(nn.Module):
    """Pre-norm attention + residual feed-forward block."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        hidden = max(int(round(float(dim) * float(mlp_ratio))), int(dim))
        self.norm1 = nn.LayerNorm(dim)
        self.attn = PhysicsAttention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden, dim),
            nn.Dropout(float(dropout)),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        attention_allow: torch.Tensor | None = None,
    ) -> torch.Tensor:
        tokens = tokens + self.attn(self.norm1(tokens), attention_allow)
        tokens = tokens + self.ffn(self.norm2(tokens))
        return tokens


class PhysicsAttentionStack(nn.Module):
    """A configurable stack with the same explicit group mask contract."""

    def __init__(
        self,
        dim: int,
        depth: int = 2,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        if int(depth) <= 0:
            raise ValueError("depth must be positive")
        self.blocks = nn.ModuleList(
            PhysicsAttentionBlock(dim, num_heads, mlp_ratio, dropout)
            for _ in range(int(depth))
        )

    def forward(
        self,
        tokens: torch.Tensor,
        attention_allow: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for block in self.blocks:
            tokens = block(tokens, attention_allow)
        return tokens


__all__ = ["PhysicsAttention", "PhysicsAttentionBlock", "PhysicsAttentionStack"]
