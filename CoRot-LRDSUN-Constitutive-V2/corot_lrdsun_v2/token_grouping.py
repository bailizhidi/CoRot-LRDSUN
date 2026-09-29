"""Explicit sample/time grouping for Physics-Attention.

The V1 leading dimension is a material-state mini-batch.  V2 must additionally
carry the semantic group key ``(sample_id, time_index)`` so a future packed
batch cannot silently mix unrelated FE samples or frames.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch


@dataclass(frozen=True)
class TokenGroupKey:
    sample_id: int
    time_index: int


@dataclass
class PackedTokenGroups:
    tokens: torch.Tensor          # [N_total, d]
    group_keys: tuple[TokenGroupKey, ...]  # one key per token row
    group_ptr: torch.Tensor       # [G+1]
    attention_allow: torch.Tensor # [N_total,N_total], True = attention allowed

    @property
    def num_tokens(self) -> int:
        return int(self.tokens.shape[0])

    @property
    def num_groups(self) -> int:
        return int(self.group_ptr.numel() - 1)


def key_for_batch(batch) -> TokenGroupKey:
    return TokenGroupKey(int(batch.sample_id), int(batch.time_index))


def assert_single_group(batch) -> None:
    """Fail fast for the normal one-sample/one-time attention contract."""
    key = key_for_batch(batch)
    if not isinstance(key.sample_id, int) or not isinstance(key.time_index, int):
        raise TypeError("sample_id and time_index must be scalar integers")


def build_block_diagonal_mask(
    group_keys: Iterable[TokenGroupKey],
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Build a boolean allow-mask that prevents cross-group attention."""
    keys = tuple(group_keys)
    if not keys:
        return torch.zeros((0, 0), dtype=torch.bool, device=device)
    sample = torch.tensor([k.sample_id for k in keys], device=device)
    time = torch.tensor([k.time_index for k in keys], device=device)
    return (sample[:, None] == sample[None, :]) & (time[:, None] == time[None, :])


def pack_token_groups(
    groups: Iterable[tuple[TokenGroupKey, torch.Tensor]],
    *,
    device: torch.device | str | None = None,
) -> PackedTokenGroups:
    """Pack groups and produce a mandatory block-diagonal attention mask.

    This function is the only supported way to concatenate multiple token
    sequences.  It never relies on ordering to enforce isolation.
    """
    groups = tuple(groups)
    if not groups:
        raise ValueError("At least one token group is required")
    tensors = []
    keys_per_token: list[TokenGroupKey] = []
    ptr = [0]
    dim = None
    for key, tokens in groups:
        if tokens.ndim != 2:
            raise ValueError(f"tokens for {key} must be [N,d], got {tuple(tokens.shape)}")
        if tokens.shape[0] == 0:
            raise ValueError(f"empty token group is not allowed: {key}")
        if dim is None:
            dim = int(tokens.shape[1])
        elif int(tokens.shape[1]) != dim:
            raise ValueError("all token groups must have the same feature dimension")
        tensors.append(tokens)
        keys_per_token.extend([key] * int(tokens.shape[0]))
        ptr.append(ptr[-1] + int(tokens.shape[0]))
    packed = torch.cat(tensors, dim=0)
    if device is not None:
        packed = packed.to(device)
    mask = build_block_diagonal_mask(keys_per_token, device=packed.device)
    return PackedTokenGroups(
        tokens=packed,
        group_keys=tuple(keys_per_token),
        group_ptr=torch.tensor(ptr, dtype=torch.long, device=packed.device),
        attention_allow=mask,
    )


def assert_no_sample_or_frame_mixing(group_keys: Iterable[TokenGroupKey]) -> None:
    """Validate that a sequence contains exactly one sample and one frame."""
    keys = tuple(group_keys)
    if not keys:
        raise ValueError("empty attention sequence")
    samples = {k.sample_id for k in keys}
    times = {k.time_index for k in keys}
    if len(samples) != 1 or len(times) != 1:
        raise AssertionError(
            "attention sequence mixes sample/time groups: "
            f"samples={sorted(samples)}, times={sorted(times)}"
        )


__all__ = [
    "TokenGroupKey",
    "PackedTokenGroups",
    "key_for_batch",
    "assert_single_group",
    "build_block_diagonal_mask",
    "pack_token_groups",
    "assert_no_sample_or_frame_mixing",
]
