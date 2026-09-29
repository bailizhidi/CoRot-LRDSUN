#!/usr/bin/env python3
"""Real-data four-rank DDP smoke for the V2 token model.

This is a graph and communication test only.  It creates no optimizer and
never writes a checkpoint.  Each rank samples its own 1024 geometric centers
from real V1 sample 119, producing 2048 material tokens per rank.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
sys.dont_write_bytecode = True

from corot_lrdsun_v2 import CoRotTokenTransolver, TokenBatch, TokenDataBuilder  # noqa: E402
from corot_lrdsun_v2.v1_compat import (  # noqa: E402
    PreparedSample,
    balanced_centers,
    compute_losses,
    load_manifest,
    load_stats,
    setup_ddp,
    split_records,
)


V1_PREFIXES = (
    "edge_encoder.",
    "center_encoder.",
    "query.",
    "key.",
    "value.",
    "trunk.",
    "delta8_head.",
    "plastic_logit_head.",
    "peeq_mag_head.",
    "le_head.",
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Real V2 four-rank DDP smoke")
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--sample-id", type=int, default=119)
    ap.add_argument("--time", type=int, default=0)
    ap.add_argument("--centers-per-rank", type=int, default=1024)
    ap.add_argument("--hidden-dim", type=int, default=192)
    ap.add_argument("--token-dim", type=int, default=192)
    ap.add_argument("--attention-depth", type=int, default=2)
    ap.add_argument("--attention-heads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=119)
    return ap.parse_args()


def loss_config(stats: dict[str, Any]) -> dict[str, float]:
    return {
        "w_delta8": 1.0,
        "w_peeq": 1.0,
        "w_gate": 0.20,
        "w_le": 0.25,
        "w_mises": 0.25,
        "plastic_threshold": float(stats.get("plastic_threshold", 1.0e-10)),
    }


def load_sample(cache_dir: Path, sample_id: int) -> tuple[PreparedSample, dict[str, Any]]:
    manifest = load_manifest(cache_dir)
    records = [r for r in manifest["records"] if int(r["sample_id"]) == int(sample_id)]
    if not records:
        raise KeyError(f"sample_id={sample_id} not found in {cache_dir}")
    stats_path = cache_dir / "stats.json"
    if not stats_path.is_file():
        raise FileNotFoundError(stats_path)
    return PreparedSample(records[0]), load_stats(stats_path)


def checkpoint_report(model: CoRotTokenTransolver, checkpoint: Path) -> dict[str, Any]:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    report = model.load_v1_checkpoint(ckpt, map_location="cpu")
    status = {
        prefix[:-1]: {
            "loaded": sum(k.startswith(prefix) for k in report["loaded"]),
            "missing": sum(
                k.startswith(prefix) for k in report["missing_after_warm_start"]
            ),
        }
        for prefix in V1_PREFIXES
    }
    return {
        "checkpoint_model_keys": len(ckpt.get("model", ckpt)),
        "loaded_keys": len(report["loaded"]),
        "missing_keys": report["missing_after_warm_start"],
        "unexpected_keys": report["unexpected_in_checkpoint"],
        "v1_component_status": status,
        "physics_attention_keys_in_checkpoint": [
            k for k in ckpt.get("model", ckpt) if k.startswith("physics_attention.")
        ],
    }


def main() -> None:
    args = parse_args()
    rank, local_rank, world_size, device = setup_ddp()
    if world_size != 4:
        raise RuntimeError(f"expected world_size=4, got {world_size}")
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)

    try:
        sample, stats = load_sample(args.cache_dir, args.sample_id)
        if args.time < 0 or args.time + 1 >= sample.T:
            raise ValueError(f"invalid time={args.time} for sample T={sample.T}")
        if args.centers_per_rank > sample.N:
            raise ValueError(
                f"centers_per_rank={args.centers_per_rank} exceeds N={sample.N}"
            )

        model = CoRotTokenTransolver(
            hidden_dim=args.hidden_dim,
            token_dim=args.token_dim,
            attention_depth=args.attention_depth,
            attention_heads=args.attention_heads,
        ).to(device)
        ckpt_report = checkpoint_report(model, args.checkpoint)
        v1_loaded = all(
            item["loaded"] > 0 and item["missing"] == 0
            for item in ckpt_report["v1_component_status"].values()
        )
        ddp_model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )

        rng = np.random.default_rng(args.seed + rank * 1_000_003)
        centers = balanced_centers(sample.region_id, args.centers_per_rank, rng)
        builder = TokenDataBuilder(stats)
        batch = builder.build(sample, args.time, centers, shuffle=False).to(device)
        assert isinstance(batch, TokenBatch)
        assert batch.num_geometric_centers == args.centers_per_rank
        assert batch.num_tokens == 2 * args.centers_per_rank
        assert tuple(batch.state.shape) == (2 * args.centers_per_rank, 9)
        assert tuple(batch.context_n.shape) == (2 * args.centers_per_rank, 6)

        torch.cuda.reset_peak_memory_stats(device)
        ddp_model.zero_grad(set_to_none=True)
        dist.barrier()
        out = ddp_model(batch)
        forward = all(torch.isfinite(value).all() for value in out.values())
        loss, loss_parts, _ = compute_losses(
            out,
            batch.as_loss_batch(),
            stats,
            loss_config(stats),
        )
        loss_ok = bool(torch.isfinite(loss).item())
        loss.backward()
        backward = loss_ok

        unused = [
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and parameter.grad is None
        ]
        grad_values = [
            parameter.grad.detach().float()
            for parameter in model.parameters()
            if parameter.grad is not None
        ]
        if not grad_values:
            raise RuntimeError("no gradients were produced")
        grad_checksum = torch.stack(
            [value.double().sum() for value in grad_values]
        ).sum()
        grad_l2 = torch.sqrt(
            torch.stack([value.double().square().sum() for value in grad_values]).sum()
        )
        grad_pair = torch.stack([grad_checksum, grad_l2]).to(device=device)
        gathered = [torch.zeros_like(grad_pair) for _ in range(world_size)]
        dist.all_gather(gathered, grad_pair)
        gathered = torch.stack(gathered)
        grad_diff = float((gathered - gathered[0]).abs().max().cpu())
        gradient_sync = bool(grad_diff <= 1.0e-5 * max(1.0, abs(float(gathered[0, 0].cpu()))))

        memory = torch.tensor(
            [
                torch.cuda.memory_allocated(device) / 1024**2,
                torch.cuda.memory_reserved(device) / 1024**2,
                torch.cuda.max_memory_allocated(device) / 1024**2,
                torch.cuda.max_memory_reserved(device) / 1024**2,
            ],
            dtype=torch.float64,
            device=device,
        )
        memory_all = [torch.zeros_like(memory) for _ in range(world_size)]
        dist.all_gather(memory_all, memory)
        memory_all = torch.stack(memory_all).cpu().tolist()

        flags = torch.tensor(
            [int(forward), int(loss_ok), int(backward), int(gradient_sync), int(not unused)],
            dtype=torch.int32,
            device=device,
        )
        dist.all_reduce(flags, op=dist.ReduceOp.MIN)
        if rank == 0:
            rank_memory = {
                str(i): {
                    "allocated_mb": round(row[0], 3),
                    "reserved_mb": round(row[1], 3),
                    "peak_allocated_mb": round(row[2], 3),
                    "peak_reserved_mb": round(row[3], 3),
                }
                for i, row in enumerate(memory_all)
            }
            result = {
                "status": "PASS"
                if bool(flags.all().item()) and v1_loaded
                else "FAIL",
                "real_data": True,
                "backend": dist.get_backend(),
                "world_size": world_size,
                "ranks": list(range(world_size)),
                "sample": args.sample_id,
                "time": args.time,
                "geometric_centers_per_rank": args.centers_per_rank,
                "tokens_per_rank": [2 * args.centers_per_rank] * world_size,
                "token_shape_per_rank": [2 * args.centers_per_rank, args.token_dim],
                "forward": bool(flags[0].item()),
                "loss": bool(flags[1].item()),
                "backward": bool(flags[2].item()),
                "gradient_sync": {
                    "pass": bool(flags[3].item()),
                    "max_checksum_l2_abs_diff": grad_diff,
                    "rank0_grad_l2": float(gathered[0, 1]),
                },
                "unused_parameters": {
                    "pass": bool(flags[4].item()),
                    "rank0_names": unused,
                },
                "loss_components_rank0": {
                    key: float(value) for key, value in loss_parts.items()
                },
                "checkpoint_warm_start": {
                    "v1_encoder_trunk_heads_loaded": v1_loaded,
                    **ckpt_report,
                },
                "peak_memory": rank_memory,
                "optimizer_step_called": False,
                "checkpoint_saved": False,
                "ready_for_pilot_training": bool(flags.all().item()) and v1_loaded,
            }
            print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
