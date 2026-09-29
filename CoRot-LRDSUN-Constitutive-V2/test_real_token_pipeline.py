#!/usr/bin/env python3
"""Step 1.5: read-only integration smoke for real V1 cache/checkpoints.

This script deliberately does not create an optimizer or call ``step``.  For
each requested geometric-center count it performs:

    V1 PreparedSample -> paired token data -> V2 forward -> V1 loss -> backward

The attention sequence is one sample and one frame.  The script is intended to
be run from the V2 directory, but also adds that directory to ``sys.path`` so
an absolute path invocation works.  No synthetic fallback is provided here:
this is a real-data integration test.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
# Avoid creating __pycache__ files in the read-only V1 tree while importing it.
sys.dont_write_bytecode = True

from corot_lrdsun_v2 import CoRotTokenTransolver, TokenBatch, TokenDataBuilder  # noqa: E402
from corot_lrdsun_v2.v1_compat import (  # noqa: E402
    PreparedSample,
    V1_ROOT,
    balanced_centers,
    compute_losses,
    load_manifest,
    load_stats,
)


EXPECTED_V1_PREFIXES = (
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


def _default_checkpoint() -> Path | None:
    """Resolve the V1 pointer file if it exists in the checked-out project."""
    pointer = V1_ROOT / "finetune_5step_latest.txt"
    if not pointer.is_file():
        return None
    text = pointer.read_text(encoding="utf-8").strip()
    if not text:
        return None
    candidate = Path(text) / "best_val_5step.pt"
    return candidate


def _parse_args() -> argparse.Namespace:
    env_cache = os.environ.get("PREPARED_CACHE_V1")
    env_ckpt = os.environ.get("V1_CHECKPOINT")
    ap = argparse.ArgumentParser(
        description="Real V1 cache/checkpoint -> V2 token/forward/loss/backward smoke"
    )
    ap.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(env_cache) if env_cache else None,
        help="V1 prepared_cache_prod_v1 directory (or PREPARED_CACHE_V1)",
    )
    ap.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(env_ckpt) if env_ckpt else _default_checkpoint(),
        help="V1 best_val_5step.pt (or V1_CHECKPOINT)",
    )
    ap.add_argument("--stats", type=Path, default=None)
    ap.add_argument("--sample-id", type=int, default=119)
    ap.add_argument("--time", type=int, default=0)
    ap.add_argument("--centers", type=int, nargs="+", default=[1024, 2048])
    ap.add_argument("--hidden-dim", type=int, default=None)
    ap.add_argument("--token-dim", type=int, default=None)
    ap.add_argument("--attention-depth", type=int, default=2)
    ap.add_argument("--attention-heads", type=int, default=4)
    ap.add_argument("--attention-mlp-ratio", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=119)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-backward", action="store_true", help="forward/loss only; not the default")
    return ap.parse_args()


def _require_real_inputs(args: argparse.Namespace) -> None:
    missing = []
    if args.cache_dir is None:
        missing.append("--cache-dir (or PREPARED_CACHE_V1)")
    elif not args.cache_dir.is_dir():
        missing.append(f"cache directory does not exist: {args.cache_dir}")
    if args.checkpoint is None:
        missing.append("--checkpoint (or V1_CHECKPOINT)")
    elif not args.checkpoint.is_file():
        missing.append(f"checkpoint does not exist: {args.checkpoint}")
    if missing:
        raise FileNotFoundError(
            "Real integration inputs are unavailable. Missing: " + "; ".join(missing)
        )


def _load_sample(
    cache_dir: Path,
    sample_id: int,
    stats_path: Path | None = None,
) -> tuple[PreparedSample, dict[str, Any]]:
    manifest = load_manifest(cache_dir)
    records = [r for r in manifest["records"] if int(r["sample_id"]) == int(sample_id)]
    if not records:
        available = sorted(int(r["sample_id"]) for r in manifest["records"])
        raise KeyError(f"sample_id={sample_id} not found; available range/count={available[:5]}.../{len(available)}")
    stats_path = stats_path or (cache_dir / "stats.json")
    if not stats_path.is_file():
        raise FileNotFoundError(f"V1 stats file not found: {stats_path}")
    return PreparedSample(records[0]), load_stats(stats_path)


def _load_checkpoint(path: Path) -> dict[str, Any]:
    # V1 checkpoints are ordinary torch dictionaries with a ``model`` key.
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch versions before the weights_only argument.
        return torch.load(path, map_location="cpu")


def _model_config(args: argparse.Namespace, checkpoint: dict[str, Any]) -> tuple[int, int]:
    cfg = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    hidden = int(args.hidden_dim if args.hidden_dim is not None else cfg.get("hidden_dim", 192))
    token = int(args.token_dim if args.token_dim is not None else cfg.get("token_dim", hidden))
    return hidden, token


def _checkpoint_report(model: CoRotTokenTransolver, checkpoint: dict[str, Any]) -> dict[str, Any]:
    state = checkpoint.get("model", checkpoint)
    if not isinstance(state, dict):
        raise TypeError("checkpoint must contain a state-dict under key 'model'")

    # Snapshot the new attention block to make the "new parameters are not
    # loaded from V1" property explicit.
    attention_before = {
        k: v.detach().clone()
        for k, v in model.state_dict().items()
        if k.startswith("physics_attention.")
    }
    report = model.load_v1_checkpoint(checkpoint, map_location="cpu")
    attention_unchanged = all(
        torch.equal(model.state_dict()[key], value)
        for key, value in attention_before.items()
    )
    loaded = report["loaded"]
    prefix_status = {
        prefix[:-1]: {
            "loaded": sum(k.startswith(prefix) for k in loaded),
            "missing": sum(
                k.startswith(prefix) for k in report["missing_after_warm_start"]
            ),
        }
        for prefix in EXPECTED_V1_PREFIXES
    }
    return {
        "checkpoint_model_keys": len(state),
        "loaded_keys": loaded,
        "missing_keys": report["missing_after_warm_start"],
        "unexpected_keys": report["unexpected_in_checkpoint"],
        "v1_component_status": prefix_status,
        "physics_attention_keys_in_checkpoint": [
            k for k in state if k.startswith("physics_attention.")
        ],
        "physics_attention_initialized_new_and_unchanged": attention_unchanged,
    }


def _loss_config(stats: dict[str, Any]) -> dict[str, float]:
    return {
        "w_delta8": 1.0,
        "w_peeq": 1.0,
        "w_gate": 0.20,
        "w_le": 0.25,
        "w_mises": 0.25,
        "plastic_threshold": float(stats.get("plastic_threshold", 1.0e-10)),
    }


def _memory_snapshot(device: torch.device) -> dict[str, float | None]:
    if device.type != "cuda":
        return {"allocated_mb": None, "reserved_mb": None}
    return {
        "allocated_mb": round(torch.cuda.memory_allocated(device) / 1024**2, 3),
        "reserved_mb": round(torch.cuda.memory_reserved(device) / 1024**2, 3),
    }


def _run_one_count(
    *,
    model: CoRotTokenTransolver,
    builder: TokenDataBuilder,
    sample: PreparedSample,
    stats: dict[str, Any],
    cfg: dict[str, float],
    centers_count: int,
    time_index: int,
    seed: int,
    device: torch.device,
    do_backward: bool,
) -> dict[str, Any]:
    if centers_count > sample.N:
        raise ValueError(f"centers={centers_count} exceeds sample N={sample.N}")
    rng = np.random.default_rng(seed + centers_count)
    geometric_centers = balanced_centers(sample.region_id, centers_count, rng)
    batch = builder.build(sample, time_index, geometric_centers, shuffle=False).to(device)
    assert isinstance(batch, TokenBatch), "TokenDataBuilder.build must return TokenBatch"
    assert batch.num_geometric_centers == int(centers_count)
    assert batch.num_tokens == 2 * int(centers_count)
    assert tuple(batch.state.shape) == (2 * int(centers_count), 9)
    assert tuple(batch.context_n.shape) == (2 * int(centers_count), 6)
    assert tuple(batch.edge_center_index.shape) == (2 * int(centers_count),)
    model.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    status = "ok"
    error = None
    loss_parts: dict[str, float] = {}
    grad_norm = None
    out = None
    loss = None
    try:
        out = model(batch)
        forward_memory = _memory_snapshot(device)
        loss, parts, _ = compute_losses(out, batch.as_loss_batch(), stats, cfg)
        loss_parts = {key: float(value) for key, value in parts.items()}
        finite_loss = bool(torch.isfinite(loss).item())
        if not finite_loss:
            raise FloatingPointError(f"non-finite loss: {float(loss.detach())}")
        if do_backward:
            loss.backward()
            squared = [
                parameter.grad.detach().float().square().sum()
                for parameter in model.parameters()
                if parameter.grad is not None
            ]
            grad_norm = float(torch.sqrt(torch.stack(squared).sum()).cpu()) if squared else 0.0
        backward_memory = _memory_snapshot(device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        peak = {
            "peak_allocated_mb": round(torch.cuda.max_memory_allocated(device) / 1024**2, 3)
            if device.type == "cuda"
            else None,
            "peak_reserved_mb": round(torch.cuda.max_memory_reserved(device) / 1024**2, 3)
            if device.type == "cuda"
            else None,
        }
    except torch.cuda.OutOfMemoryError as exc:
        status = "cuda_oom"
        error = str(exc).splitlines()[0]
        forward_memory = _memory_snapshot(device)
        backward_memory = _memory_snapshot(device)
        peak = {
            "peak_allocated_mb": round(torch.cuda.max_memory_allocated(device) / 1024**2, 3),
            "peak_reserved_mb": round(torch.cuda.max_memory_reserved(device) / 1024**2, 3),
        }
    finally:
        # Keep the next center count independent and release the graph before
        # the next full attention matrix is allocated.
        del out, loss, batch
        model.zero_grad(set_to_none=True)
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return {
        "geometric_centers": int(centers_count),
        "material_tokens": int(2 * centers_count),
        "token_shape": [int(2 * centers_count), int(model.token_dim)],
        "state_shape": [int(2 * centers_count), 9],
        "output_shapes": {
            "delta8_norm": [int(2 * centers_count), 8],
            "plastic_logit": [int(2 * centers_count), 1],
            "peeq_mag_raw": [int(2 * centers_count), 1],
            "le_norm": [int(2 * centers_count), 4],
        },
        "status": status,
        "error": error,
        "loss_components": loss_parts,
        "forward_memory": forward_memory,
        "backward_memory": backward_memory,
        **peak,
        "backward_graph_checked": bool(do_backward and status == "ok"),
        "grad_l2": grad_norm,
    }


def main() -> None:
    args = _parse_args()
    try:
        _require_real_inputs(args)
        assert args.cache_dir is not None and args.checkpoint is not None
        sample, stats = _load_sample(args.cache_dir, args.sample_id, args.stats)
        checkpoint = _load_checkpoint(args.checkpoint)
        hidden_dim, token_dim = _model_config(args, checkpoint)
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("--device=cuda requested but CUDA is unavailable")
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        model = CoRotTokenTransolver(
            hidden_dim=hidden_dim,
            token_dim=token_dim,
            attention_depth=args.attention_depth,
            attention_heads=args.attention_heads,
            attention_mlp_ratio=args.attention_mlp_ratio,
        ).to(device)
        checkpoint_report = _checkpoint_report(model, checkpoint)
        v1_components_loaded = all(
            item["loaded"] > 0 and item["missing"] == 0
            for item in checkpoint_report["v1_component_status"].values()
        )
        builder = TokenDataBuilder(stats)
        cfg = _loss_config(stats)
        counts = []
        for count in args.centers:
            counts.append(
                _run_one_count(
                    model=model,
                    builder=builder,
                    sample=sample,
                    stats=stats,
                    cfg=cfg,
                    centers_count=int(count),
                    time_index=args.time,
                    seed=args.seed,
                    device=device,
                    do_backward=not args.no_backward,
                )
            )

        result = {
            "status": "PASS"
            if v1_components_loaded and all(row["status"] == "ok" for row in counts)
            else "PARTIAL",
            "real_data": True,
            "cache_dir": str(args.cache_dir),
            "checkpoint": str(args.checkpoint),
            "sample": int(args.sample_id),
            "time": int(args.time),
            "sample_nodes": int(sample.N),
            "sample_frames": int(sample.T),
            "device": str(device),
            "model": {
                "hidden_dim": hidden_dim,
                "token_dim": token_dim,
                "attention_depth": args.attention_depth,
                "attention_heads": args.attention_heads,
            },
            "checkpoint_report": checkpoint_report,
            "v1_encoder_trunk_heads_loaded": v1_components_loaded,
            "runs": counts,
            "optimizer_step_called": False,
            "ready_for_formal_transolver_training": all(
                row["status"] == "ok" and row["backward_graph_checked"] for row in counts
            ) and v1_components_loaded,
        }
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except Exception as exc:
        # A missing external cache/checkpoint is an environment result, not a
        # synthetic success.  Keep the output machine-readable and actionable.
        print(
            json.dumps(
                {
                    "status": "BLOCKED_MISSING_REAL_INPUT" if isinstance(exc, FileNotFoundError) else "FAIL",
                    "real_data": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "optimizer_step_called": False,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        raise SystemExit(2 if isinstance(exc, FileNotFoundError) else 1)


if __name__ == "__main__":
    main()
