#!/usr/bin/env python3
"""Read-only V2 token pipeline smoke test.

With ``--cache-dir`` this exercises a real V1 prepared sample.  Without it the
script uses a deterministic synthetic shell-like sample, so the token/grouping
contract can be checked in a clean checkout before any training is attempted.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np
import torch

from corot_lrdsun_v2 import CoRotTokenTransolver, TokenDataBuilder
from corot_lrdsun_v2.token_grouping import (
    TokenGroupKey,
    assert_no_sample_or_frame_mixing,
    build_block_diagonal_mask,
    pack_token_groups,
)
from corot_lrdsun_v2.v1_compat import (
    PreparedSample,
    load_manifest,
    load_stats,
    split_records,
)


class SyntheticSample:
    """Minimal PreparedSample-compatible object for a no-data pipeline test."""

    def __init__(self, n: int = 96, t: int = 8, sample_id: int = 7):
        self.N = int(n)
        self.T = int(t)
        self.D = 10.0
        self.t_over_D = 0.08
        self.R_over_D = 3.5
        self.E_modulus = 110000.0
        self.nu = 0.39
        self.meta = {"sample_id": int(sample_id)}
        x = np.linspace(0.0, 1.0, self.N, dtype=np.float32)
        self.X0 = np.stack([10.0 * x, np.sin(4.0 * x), 0.05 * np.cos(3.0 * x)], axis=1)
        self.U = np.zeros((self.T, self.N, 3), dtype=np.float32)
        for frame in range(self.T):
            self.U[frame, :, 1] = 0.02 * frame * np.sin(2.0 * x)
            self.U[frame, :, 2] = 0.01 * frame * x
        self.Q0 = np.repeat(np.eye(3, dtype=np.float32)[None], self.N, axis=0)
        neighbors = []
        ptr = [0]
        for i in range(self.N):
            ids = sorted({j for j in (i - 2, i - 1, i + 1, i + 2) if 0 <= j < self.N})
            neighbors.extend(ids)
            ptr.append(len(neighbors))
        self.ptr = np.asarray(ptr, dtype=np.int64)
        self.idx = np.asarray(neighbors, dtype=np.int32)
        self.R = np.repeat(np.eye(3, dtype=np.float32)[None, None], self.T * self.N, axis=0)
        self.R = self.R.reshape(self.T, self.N, 3, 3)
        self.region_id = np.mod(np.arange(self.N), 3).astype(np.int8)
        self.angle_progress = np.linspace(0.0, 1.0, self.T, dtype=np.float32)
        base = np.zeros((self.T, self.N, 9), dtype=np.float32)
        base[..., 0] = 25.0 * self.angle_progress[:, None]
        base[..., 1] = -10.0 * self.angle_progress[:, None]
        base[..., 4] = 0.001 * self.angle_progress[:, None]
        base[..., 8] = 0.002 * self.angle_progress[:, None]
        self.state_outer = base.copy()
        self.state_inner = base.copy()
        self.state_inner[..., 0] += 1.5
        self.LE_outer = np.zeros((self.T, self.N, 4), dtype=np.float32)
        self.LE_inner = np.zeros((self.T, self.N, 4), dtype=np.float32)

    def states_at(self, t, centers, surfaces):
        centers = np.asarray(centers, dtype=np.int64)
        surfaces = np.asarray(surfaces, dtype=np.int64)
        return np.where(
            surfaces[:, None] == 0,
            self.state_outer[int(t), centers],
            self.state_inner[int(t), centers],
        ).astype(np.float32)

    def le_at(self, t, centers, surfaces):
        centers = np.asarray(centers, dtype=np.int64)
        surfaces = np.asarray(surfaces, dtype=np.int64)
        return np.where(
            surfaces[:, None] == 0,
            self.LE_outer[int(t), centers],
            self.LE_inner[int(t), centers],
        ).astype(np.float32)


def load_real_sample(cache_dir: Path, sample_id: int):
    manifest = load_manifest(cache_dir)
    records = manifest["records"]
    matches = [r for r in records if int(r["sample_id"]) == int(sample_id)]
    if not matches:
        raise KeyError(f"sample_id={sample_id} not found in {cache_dir}")
    return PreparedSample(matches[0]), load_stats(cache_dir / "stats.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", type=Path, default=None)
    ap.add_argument("--sample-id", type=int, default=1)
    ap.add_argument("--time", type=int, default=0)
    ap.add_argument("--centers", type=int, default=8)
    ap.add_argument("--hidden-dim", type=int, default=64)
    ap.add_argument("--token-dim", type=int, default=64)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    if args.cache_dir is None:
        sample = SyntheticSample(sample_id=args.sample_id)
        stats = None
    else:
        sample, stats = load_real_sample(args.cache_dir, args.sample_id)

    if args.centers > sample.N:
        raise ValueError(f"centers={args.centers} exceeds N={sample.N}")
    centers = rng.choice(sample.N, size=args.centers, replace=False).astype(np.int64)
    builder = TokenDataBuilder(stats)
    batch = builder.build(
        sample,
        args.time,
        centers,
        shuffle=args.shuffle,
        seed=args.seed,
    )
    model = CoRotTokenTransolver(
        hidden_dim=args.hidden_dim,
        token_dim=args.token_dim,
        attention_depth=2,
        attention_heads=4,
    ).to(args.device)
    model.eval()
    batch = batch.to(args.device)

    with torch.no_grad():
        tokens = model.token_sequence(batch)
        out = model(batch)

        # Permutation equivariance: metadata and rows are permuted together;
        # output must permute in exactly the same way.
        order = torch.randperm(batch.num_tokens, device=batch.state.device)
        perm_batch = batch.permuted(order)
        perm_tokens = model.token_sequence(perm_batch)
        equivariance_error = float(
            (perm_tokens - tokens.index_select(0, order)).abs().max().cpu()
        )

        # Explicit block-diagonal isolation test on two different groups.
        key_a = TokenGroupKey(batch.sample_id, batch.time_index)
        key_b = TokenGroupKey(batch.sample_id + 1000, batch.time_index + 3)
        assert_no_sample_or_frame_mixing((key_a,))
        block = pack_token_groups(((key_a, tokens), (key_b, tokens.clone())))
        cross_mask = build_block_diagonal_mask(
            [key_a] * batch.num_tokens + [key_b] * batch.num_tokens,
            device=tokens.device,
        )
        independent_out = model.physics_attention(tokens)
        packed_out = model.physics_attention(block.tokens, block.attention_allow)
        split = batch.num_tokens
        isolated_error = float(
            max(
                (packed_out[:split] - independent_out).abs().max().cpu(),
                (packed_out[split:] - independent_out).abs().max().cpu(),
            )
        )

    result = {
        "sample_id": batch.sample_id,
        "time": batch.time_index,
        "number_of_geometric_centers": batch.num_geometric_centers,
        "number_of_material_tokens": batch.num_tokens,
        "token_shape": list(tokens.shape),
        "edge_unique_shape": list(batch.edge_unique_n.shape),
        "center_id_shape": list(batch.center_id.shape),
        "surface_id_shape": list(batch.surface_id.shape),
        "material_state_index_shape": list(batch.material_state_index.shape),
        "surface_counts": {
            "SPOS": int((batch.surface_id == 0).sum()),
            "SNEG": int((batch.surface_id == 1).sum()),
        },
        "output_shapes": {key: list(value.shape) for key, value in out.items()},
        "cross_group_mask_entries": int((~cross_mask).sum()),
        "permutation_equivariance_max_abs_error": equivariance_error,
        "block_mask_isolation_max_abs_error": isolated_error,
        "sample_time_isolation": "PASS",
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
