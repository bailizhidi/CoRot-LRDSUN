#!/usr/bin/env python3
"""V2 autoregressive rollout with all material tokens in one spatial sequence."""

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
    load_manifest,
    load_stats,
)


def mises(s4: np.ndarray) -> np.ndarray:
    saa, stt, srr, sat = [s4[..., i] for i in range(4)]
    vm2 = 0.5 * ((saa - stt) ** 2 + (stt - srr) ** 2 + (srr - saa) ** 2) + 3.0 * sat**2
    return np.sqrt(np.maximum(vm2, 0.0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--split", choices=("val", "test"), default="test")
    ap.add_argument("--max-samples", type=int, default=0)
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
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []

    with torch.no_grad():
        for rec in records:
            sample = PreparedSample(rec)
            centers = np.arange(sample.N, dtype=np.int64)
            if args.centers_per_step > 0:
                centers = centers[: min(args.centers_per_step, sample.N)]
            pred = np.stack(
                [
                    np.asarray(sample.state_outer[0], dtype=np.float32),
                    np.asarray(sample.state_inner[0], dtype=np.float32),
                ],
                axis=0,
            )[:, centers].copy()
            builder = TokenDataBuilder(stats)
            sse_s = sse_vm = sse_p = 0.0
            count = 0
            frame_rows = []

            for t in range(sample.T - 1):
                override = np.stack([pred[0], pred[1]], axis=1).reshape(-1, 9)
                batch = builder.build(
                    sample,
                    t,
                    centers,
                    shuffle=False,
                    state_override=override,
                ).to(device)
                out = model(batch)
                d8 = delta8_from_norm(out["delta8_norm"].float(), stats)
                p_scaled = (
                    model.peeq_scaled_hard(out, args.gate_threshold)
                    if not args.soft_gate
                    else model.peeq_scaled_soft(out)
                ).float()
                pred_tokens = torch.cat(
                    [
                        batch.state[:, :8].float() + d8,
                        batch.state[:, 8:9].float() + p_scaled * float(stats["peeq_delta_scale"]),
                    ],
                    dim=1,
                ).cpu().numpy()

                next_pred = np.empty_like(pred)
                for surface in (0, 1):
                    rows = np.flatnonzero(batch.surface_id.cpu().numpy() == surface)
                    # shuffle=False makes the rows pairwise canonical, but use
                    # metadata rather than relying on row position.
                    for row in rows:
                        center = int(batch.center_id[row])
                        where = np.flatnonzero(centers == center)
                        if len(where) != 1:
                            raise RuntimeError("center mapping is not one-to-one in rollout subset")
                        next_pred[surface, where[0]] = pred_tokens[row]
                pred = next_pred

                gt = np.stack(
                    [
                        np.asarray(sample.state_outer[t + 1], dtype=np.float32)[centers],
                        np.asarray(sample.state_inner[t + 1], dtype=np.float32)[centers],
                    ],
                    axis=0,
                ).reshape(-1, 9)
                dp = pred.reshape(-1, 9).astype(np.float64) - gt.astype(np.float64)
                sse_s += float(np.square(dp[:, :8]).sum())
                sse_p += float(np.square(dp[:, 8]).sum())
                sse_vm += float(np.square(mises(pred.reshape(-1, 9)[:, :4]) - mises(gt[:, :4])).sum())
                count += int(dp.shape[0])
                frame_rows.append({
                    "frame": t + 1,
                    "rmse_state8": float(np.sqrt(np.square(dp[:, :8]).mean())),
                    "rmse_peeq": float(np.sqrt(np.square(dp[:, 8]).mean())),
                    "rmse_mises": float(np.sqrt(np.square(mises(pred.reshape(-1, 9)[:, :4]) - mises(gt[:, :4])).mean())),
                })

            summary = {
                "sample_id": int(rec["sample_id"]),
                "num_predictions": count,
                "rmse_state8": float(np.sqrt(sse_s / max(count * 8, 1))),
                "rmse_peeq": float(np.sqrt(sse_p / max(count, 1))),
                "rmse_mises": float(np.sqrt(sse_vm / max(count, 1))),
                "frames": frame_rows,
            }
            summaries.append(summary)
            (args.output_dir / f"sample_{int(rec['sample_id']):04d}_summary.json").write_text(
                json.dumps(summary, indent=2), encoding="utf-8"
            )

    result = {
        "mode": "token_autoregressive_rollout",
        "split": args.split,
        "checkpoint": str(args.checkpoint),
        "sample_count": len(summaries),
        "samples": summaries,
    }
    (args.output_dir / "rollout_summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
