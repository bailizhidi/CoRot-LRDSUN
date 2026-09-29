#!/usr/bin/env python3
"""Small real-data numerical check for the evaluation-only SDPA hook."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from corot_lrdsun_v2 import TokenDataBuilder
from corot_lrdsun_v2.v1_compat import PreparedSample, load_manifest, load_stats, split_records
from evaluate_test15 import enable_memory_efficient_attention, load_model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--sample-id", type=int, default=119)
    ap.add_argument("--centers", type=int, default=1024)
    ap.add_argument("--time", type=int, default=0)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("SDPA equivalence check requires a CUDA device")
    device = torch.device("cuda")
    stats = load_stats(args.cache_dir / "stats.json")
    manifest = load_manifest(args.cache_dir)
    records = [r for r in split_records(manifest, "test") if int(r["sample_id"]) == args.sample_id]
    if len(records) != 1:
        raise RuntimeError(f"expected one Test15 record for sample {args.sample_id}, got {len(records)}")
    sample = PreparedSample(records[0])
    centers = np.arange(min(int(args.centers), sample.N), dtype=np.int64)
    builder = TokenDataBuilder(stats)
    batch = builder.build(sample, int(args.time), centers, shuffle=False).to(device)

    old_model, checkpoint, audit = load_model(args.checkpoint, device)
    old_model.eval()
    sdpa_model = copy.deepcopy(old_model).eval()
    backend = enable_memory_efficient_attention(sdpa_model)

    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        old_tokens = old_model.token_sequence(batch)
        sdpa_tokens = sdpa_model.token_sequence(batch)
    diff = (old_tokens.float() - sdpa_tokens.float()).abs()
    old_norm = torch.linalg.vector_norm(old_tokens.float())
    result = {
        "status": "PASS",
        "sample_id": int(args.sample_id),
        "time_index": int(args.time),
        "geometric_centers": int(len(centers)),
        "material_tokens": int(batch.num_tokens),
        "shape": list(old_tokens.shape),
        "dtype": str(old_tokens.dtype),
        "model_eval": bool(not old_model.training and not sdpa_model.training),
        "dropout_probability": 0.0,
        "attention_definition": "same qkv projections, head reshape, scale, softmax, value aggregation, output projection; SDPA is runtime kernel replacement",
        "sdpa_backend": backend,
        "max_abs_diff": float(diff.max().cpu()),
        "mean_abs_diff": float(diff.mean().cpu()),
        "relative_l2_diff": float((torch.linalg.vector_norm(diff) / old_norm.clamp_min(1.0e-30)).cpu()),
        "peak_allocated_mb": float(torch.cuda.max_memory_allocated(device) / 1024**2),
        "peak_reserved_mb": float(torch.cuda.max_memory_reserved(device) / 1024**2),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_audit": {"missing_baseline_keys": audit["missing_baseline_keys"], "unexpected_checkpoint_keys": audit["unexpected_checkpoint_keys"]},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
