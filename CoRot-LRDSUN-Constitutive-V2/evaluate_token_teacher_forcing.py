#!/usr/bin/env python3
"""V2 teacher-forcing evaluator: one complete token sequence per sample/time."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from corot_lrdsun_v2 import CoRotTokenTransolver, TokenDataBuilder
from corot_lrdsun_v2.v1_compat import (
    PreparedSample,
    delta8_from_norm,
    le_from_norm,
    load_manifest,
    load_stats,
)


def mises(s4: np.ndarray) -> np.ndarray:
    saa, stt, srr, sat = [s4[..., i] for i in range(4)]
    vm2 = 0.5 * ((saa - stt) ** 2 + (stt - srr) ** 2 + (srr - saa) ** 2) + 3.0 * sat**2
    return np.sqrt(np.maximum(vm2, 0.0))


def predicted_state(model, out, state, stats, hard_gate=True, threshold=0.5):
    d8 = delta8_from_norm(out["delta8_norm"].float(), stats)
    p = model.peeq_scaled_hard(out, threshold) if hard_gate else model.peeq_scaled_soft(out)
    dpeeq = p.float() * float(stats["peeq_delta_scale"])
    return torch.cat([state[:, :8].float() + d8, state[:, 8:9].float() + dpeeq], dim=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--split", choices=("val", "test"), default="test")
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--transition-stride", type=int, default=1)
    ap.add_argument("--centers-per-step", type=int, default=0, help="0 = all geometric centers")
    ap.add_argument("--soft-gate", action="store_true")
    ap.add_argument("--gate-threshold", type=float, default=0.5)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    ck = torch.load(args.checkpoint, map_location=device)
    stats = ck.get("stats") or load_stats(args.cache_dir / "stats.json")
    cfg = ck.get("config", {})
    model = CoRotTokenTransolver(
        hidden_dim=int(cfg.get("hidden_dim", 192)),
        token_dim=int(cfg.get("token_dim", cfg.get("hidden_dim", 192))),
        attention_depth=int(cfg.get("attention_depth", 2)),
        attention_heads=int(cfg.get("attention_heads", 4)),
    ).to(device)
    model.load_v1_checkpoint(ck)
    model.eval()

    manifest = load_manifest(args.cache_dir)
    records = [r for r in manifest["records"] if str(r["split"]).lower() == args.split]
    if args.max_samples > 0:
        records = records[: args.max_samples]
    summaries = []
    with torch.no_grad():
        for rec in records:
            sample = PreparedSample(rec)
            centers = np.arange(sample.N, dtype=np.int64)
            if args.centers_per_step > 0:
                centers = centers[: min(args.centers_per_step, sample.N)]
            builder = TokenDataBuilder(stats)
            sse_s = sse_vm = sse_p = 0.0
            count = 0
            for t in range(0, sample.T - 1, args.transition_stride):
                batch = builder.build(sample, t, centers, shuffle=False).to(device)
                out = model(batch)
                pred = predicted_state(
                    model,
                    out,
                    batch.state,
                    stats,
                    hard_gate=not args.soft_gate,
                    threshold=args.gate_threshold,
                )
                true = batch.next_state
                dp = (pred - true).cpu().numpy()
                sse_s += float(np.square(dp[:, :8]).sum())
                sse_p += float(np.square(dp[:, 8]).sum())
                sse_vm += float(np.square(mises(pred[:, :4].cpu().numpy()) - mises(true[:, :4].cpu().numpy())).sum())
                count += int(pred.shape[0])
            summaries.append({
                "sample_id": int(rec["sample_id"]),
                "num_predictions": count,
                "rmse_state8": float(np.sqrt(sse_s / max(count * 8, 1))),
                "rmse_peeq": float(np.sqrt(sse_p / max(count, 1))),
                "rmse_mises": float(np.sqrt(sse_vm / max(count, 1))),
            })

    result = {
        "mode": "token_teacher_forcing",
        "split": args.split,
        "checkpoint": str(args.checkpoint),
        "sample_count": len(summaries),
        "samples": summaries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
