from __future__ import annotations

import math
import torch
from torch import nn
import torch.nn.functional as F

from .contracts import STATE_DIM, EDGE_DIM, CONTEXT_DIM, LE_DIM


def mlp(din: int, dh: int, dout: int, depth: int = 3, dropout: float = 0.0) -> nn.Sequential:
    layers = []
    d = din
    for _ in range(max(1, depth - 1)):
        layers += [nn.Linear(d, dh), nn.SiLU(), nn.LayerNorm(dh)]
        if dropout > 0: layers.append(nn.Dropout(dropout))
        d = dh
    layers.append(nn.Linear(d, dout))
    return nn.Sequential(*layers)


class CoRotLRDSUN(nn.Module):
    """Local constitutive state-update network.

    Center input: 9D material history only.
    Neighbor input: co-rotational kinematics only.
    Output: dS4, dPE4, nonnegative dPEEQ, plus LE4 auxiliary prediction.
    """
    def __init__(self, hidden_dim: int = 192, dropout: float = 0.0):
        super().__init__()
        h = int(hidden_dim)
        self.edge_encoder = mlp(EDGE_DIM, h, h, depth=3, dropout=dropout)
        self.center_encoder = mlp(STATE_DIM + CONTEXT_DIM, h, h, depth=3, dropout=dropout)
        self.query = nn.Linear(h, h)
        self.key = nn.Linear(h, h)
        self.value = nn.Linear(h, h)
        self.trunk = mlp(h * 4, h * 2, h, depth=3, dropout=dropout)
        self.delta8_head = mlp(h, h, 8, depth=2)
        self.plastic_logit_head = mlp(h, h, 1, depth=2)
        self.peeq_mag_head = mlp(h, h, 1, depth=2)
        self.le_head = mlp(h, h, LE_DIM, depth=2)

    def forward(self, state_n: torch.Tensor, context_n: torch.Tensor, edge_n: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        # edge_n [B,K,F], mask [B,K]
        he = self.edge_encoder(edge_n)
        hc = self.center_encoder(torch.cat([state_n, context_n], dim=-1))
        q = self.query(hc)[:, None, :]
        k = self.key(he)
        v = self.value(he)
        score = (q * k).sum(-1) / math.sqrt(k.shape[-1])
        score = score.masked_fill(~mask, -1.0e4)
        attn = torch.softmax(score, dim=1)
        attn = attn * mask.float()
        attn = attn / attn.sum(dim=1, keepdim=True).clamp_min(1.0e-8)
        pool_attn = (attn[..., None] * v).sum(dim=1)
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1).float()
        pool_mean = (he * mask[..., None].float()).sum(dim=1) / denom
        hmasked = he.masked_fill(~mask[..., None], -1.0e4)
        pool_max = hmasked.max(dim=1).values
        h = self.trunk(torch.cat([hc, pool_attn, pool_mean, pool_max], dim=-1))
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
    def peeq_scaled_hard(out: dict[str, torch.Tensor], threshold: float = 0.5) -> torch.Tensor:
        gate = (torch.sigmoid(out["plastic_logit"]) >= threshold).float()
        return gate * F.softplus(out["peeq_mag_raw"])
