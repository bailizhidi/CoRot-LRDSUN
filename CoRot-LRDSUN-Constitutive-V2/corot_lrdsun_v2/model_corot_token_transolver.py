"""V2 CoRot local-token constitutive model.

The V1 edge encoder, center encoder, constitutive trunk, and output heads keep
their parameter names so a V1 checkpoint can warm-start those components.
New token projection, Physics-Attention, and adapter parameters are initialized
by V2; the adapter starts at zero, making the initial forward close to the V1
trunk path while the new block learns.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from .physics_attention import PhysicsAttentionStack
from .token_data import TokenBatch, encode_shared_h_geo
from .token_grouping import assert_single_group
from .v1_compat import CONTEXT_DIM, EDGE_DIM, LE_DIM, STATE_DIM, v1_mlp


class CoRotTokenTransolver(nn.Module):
    """One material-state token sequence for one FE sample and one time step."""

    def __init__(
        self,
        hidden_dim: int = 192,
        token_dim: int = 192,
        attention_depth: int = 2,
        attention_heads: int = 4,
        attention_mlp_ratio: float = 2.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        h = int(hidden_dim)
        d = int(token_dim)
        self.hidden_dim = h
        self.token_dim = d

        # Names and dimensions match corot_lrdsun.model.CoRotLRDSUN where the
        # component is reused.  This makes V1 warm-start loading explicit.
        self.edge_encoder = v1_mlp(EDGE_DIM, h, h, depth=3, dropout=dropout)
        self.center_encoder = v1_mlp(
            STATE_DIM + CONTEXT_DIM,
            h,
            h,
            depth=3,
            dropout=dropout,
        )
        self.query = nn.Linear(h, h)
        self.key = nn.Linear(h, h)
        self.value = nn.Linear(h, h)

        # h_geo is [pool_mean, pool_max] (2H), while state/context and the V1
        # center latent are retained in the token input.
        self.token_in = nn.Linear(3 * h + STATE_DIM + CONTEXT_DIM, d)
        self.physics_attention = PhysicsAttentionStack(
            d,
            depth=attention_depth,
            num_heads=attention_heads,
            mlp_ratio=attention_mlp_ratio,
            dropout=dropout,
        )
        self.token_adapter = nn.Linear(d, 4 * h)
        nn.init.zeros_(self.token_adapter.weight)
        nn.init.zeros_(self.token_adapter.bias)

        # Existing V1 constitutive trunk and heads.
        self.trunk = v1_mlp(h * 4, h * 2, h, depth=3, dropout=dropout)
        self.delta8_head = v1_mlp(h, h, 8, depth=2)
        self.plastic_logit_head = v1_mlp(h, h, 1, depth=2)
        self.peeq_mag_head = v1_mlp(h, h, 1, depth=2)
        self.le_head = v1_mlp(h, h, LE_DIM, depth=2)

    def _device_batch(self, batch: TokenBatch) -> TokenBatch:
        device = next(self.parameters()).device
        if batch.state.device != device:
            return batch.to(device)
        return batch

    def _local_features(self, batch: TokenBatch):
        """Return V1 local features, with edge encoding done once per center."""
        edge_n = batch.edge_unique_n
        edge_mask = batch.edge_mask
        he_unique = self.edge_encoder(edge_n)  # [M_unique,K,H]
        center_index = batch.edge_center_index
        he = he_unique.index_select(0, center_index)  # [N_token,K,H]
        mask = edge_mask.index_select(0, center_index)

        hc = self.center_encoder(torch.cat([batch.state_n, batch.context_n], dim=-1))
        q = self.query(hc)[:, None, :]
        k = self.key(he)
        v = self.value(he)
        score = (q * k).sum(-1) / (k.shape[-1] ** 0.5)
        score = score.masked_fill(~mask, -1.0e4)
        attn = torch.softmax(score, dim=1)
        attn = attn * mask.float()
        attn = attn / attn.sum(dim=1, keepdim=True).clamp_min(1.0e-8)
        pool_attn = (attn[..., None] * v).sum(dim=1)

        denom = mask.sum(dim=1, keepdim=True).clamp_min(1).float()
        pool_mean = (he * mask[..., None].float()).sum(dim=1) / denom
        hmasked = he.masked_fill(~mask[..., None], -1.0e4)
        pool_max = hmasked.max(dim=1).values

        # Mean/max are edge-only and therefore shared by the two surfaces of a
        # geometric center.  The state-conditioned attention remains available
        # for the V1 trunk path.  The explicit helper is also the token_data
        # API used by pipeline tests and future data-only experiments.
        h_geo, material_features = encode_shared_h_geo(batch, he_unique=he_unique)
        x_base = torch.cat([hc, pool_attn, pool_mean, pool_max], dim=-1)
        token_in = torch.cat([material_features, hc], dim=-1)
        return token_in, x_base

    def token_sequence(
        self,
        batch: TokenBatch,
        attention_allow: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return updated material tokens ``[N_token, token_dim]``."""
        batch = self._device_batch(batch)
        assert_single_group(batch)
        token_in, _ = self._local_features(batch)
        tokens = self.token_in(token_in)
        return self.physics_attention(tokens, attention_allow)

    def forward(
        self,
        batch: TokenBatch,
        attention_allow: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if not isinstance(batch, TokenBatch):
            raise TypeError("CoRotTokenTransolver.forward expects a TokenBatch")
        batch = self._device_batch(batch)
        assert_single_group(batch)
        token_in, x_base = self._local_features(batch)
        tokens = self.token_in(token_in)
        tokens = self.physics_attention(tokens, attention_allow)
        x_fused = x_base + self.token_adapter(tokens)
        h = self.trunk(x_fused)
        return {
            "delta8_norm": self.delta8_head(h),
            "plastic_logit": self.plastic_logit_head(h),
            "peeq_mag_raw": self.peeq_mag_head(h),
            "le_norm": self.le_head(h),
        }

    @staticmethod
    def peeq_scaled_soft(out: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.sigmoid(out["plastic_logit"]) * F.softplus(out["peeq_mag_raw"])

    @staticmethod
    def peeq_scaled_hard(
        out: dict[str, torch.Tensor],
        threshold: float = 0.5,
    ) -> torch.Tensor:
        gate = (torch.sigmoid(out["plastic_logit"]) >= threshold).float()
        return gate * F.softplus(out["peeq_mag_raw"])

    def load_v1_checkpoint(
        self,
        checkpoint: str | Path | dict[str, Any],
        *,
        map_location: str | torch.device = "cpu",
    ) -> dict[str, list[str]]:
        """Warm-start matching V1 modules and report new/missing keys."""
        if isinstance(checkpoint, (str, Path)):
            checkpoint = torch.load(checkpoint, map_location=map_location)
        state = checkpoint.get("model", checkpoint)
        own = self.state_dict()
        matched = {
            key: value
            for key, value in state.items()
            if key in own and tuple(value.shape) == tuple(own[key].shape)
        }
        result = self.load_state_dict(matched, strict=False)
        return {
            "loaded": sorted(matched),
            "missing_after_warm_start": sorted(result.missing_keys),
            "unexpected_in_checkpoint": sorted(
                key for key in state.keys() if key not in matched
            ),
        }


__all__ = ["CoRotTokenTransolver"]
