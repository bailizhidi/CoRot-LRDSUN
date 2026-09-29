#!/usr/bin/env python3
"""Protocol-matched Test15 evaluation for the V2 token model.

This evaluator intentionally reuses the read-only V1 ``MetricAccumulator``
and ``PreparedSample`` contract.  Each model call contains every material
state of exactly one sample and one time step, so Physics Attention cannot
mix samples or frames.  No training, optimizer, or checkpoint mutation is
performed here.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from types import MethodType
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from corot_lrdsun_v2 import (
    CoRotTokenTransolver,
    TokenDataBuilder,
    surface_center_to_canonical_states,
    token_rows_to_surface_center_states,
)
from corot_lrdsun_v2.v1_compat import (
    PreparedSample,
    delta8_from_norm,
    le_from_norm,
    load_manifest,
    load_stats,
    split_records,
)

# This is an evaluation-only import of the unchanged V1 metric contract.
from corot_lrdsun.metrics import MetricAccumulator
from corot_lrdsun_v2.physics_attention import PhysicsAttention


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
SCALAR_NAMES = ("S", "Mises", "PEEQ", "LE")


def scalar_metrics(metric: dict[str, Any]) -> dict[str, float]:
    """The four requested scalars, derived from the V1 metric dictionary.

    ``S`` and ``LE`` are the arithmetic means over their four reported
    components (the full component arrays remain in every JSON output).
    Mises and PEEQ use V1's direct scalar MAE fields.
    """

    return {
        "S": float(np.mean(metric["mae_S_components_MPa"])),
        "Mises": float(metric["mae_S_mises_MPa"]),
        "PEEQ": float(metric["mae_PEEQ"]),
        "LE": float(np.mean(metric["mae_LE_components"])),
    }


def metric_row(sample_id: int, metric: dict[str, Any]) -> dict[str, Any]:
    row = {"sample_id": int(sample_id)}
    row.update({f"mae_{key}": value for key, value in scalar_metrics(metric).items()})
    return row


def distribution(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for key in SCALAR_NAMES:
        vals = [float(row[f"mae_{key}"]) for row in rows]
        out[key] = {
            "mean": float(statistics.mean(vals)) if vals else float("nan"),
            "median": float(statistics.median(vals)) if vals else float("nan"),
            "std": float(statistics.pstdev(vals)) if len(vals) > 1 else 0.0,
        }
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def predicted_state_and_le(model, out, state, stats, threshold: float):
    d8 = delta8_from_norm(out["delta8_norm"].float(), stats)
    p_scaled = model.peeq_scaled_hard(out, threshold).float()
    dpeeq = p_scaled * float(stats["peeq_delta_scale"])
    pred_state = torch.cat(
        [state[:, :8].float() + d8, state[:, 8:9].float() + dpeeq], dim=1
    )
    pred_le = le_from_norm(out["le_norm"].float(), stats)
    return pred_state, pred_le


def checkpoint_audit(model: CoRotTokenTransolver, checkpoint: dict[str, Any]) -> dict[str, Any]:
    state = checkpoint.get("model", checkpoint)
    report = model.load_v1_checkpoint(checkpoint)
    own = model.state_dict()
    v1_keys = [key for key in own if key.startswith(V1_PREFIXES)]
    new_keys = [key for key in own if not key.startswith(V1_PREFIXES)]
    loaded = set(report["loaded"])
    return {
        "checkpoint_epoch": checkpoint.get("epoch"),
        "loaded_v1_keys": sorted(key for key in report["loaded"] if key in v1_keys),
        "loaded_key_count": len(report["loaded"]),
        "missing_baseline_keys": sorted(key for key in v1_keys if key not in loaded),
        "unexpected_checkpoint_keys": sorted(report["unexpected_in_checkpoint"]),
        "v2_new_keys": sorted(new_keys),
        "v2_new_key_count": len(new_keys),
        "checkpoint_state_key_count": len(state),
    }


def load_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg = checkpoint.get("config", {})
    model = CoRotTokenTransolver(
        hidden_dim=int(cfg.get("hidden_dim", 192)),
        token_dim=int(cfg.get("token_dim", cfg.get("hidden_dim", 192))),
        attention_depth=int(cfg.get("attention_depth", 2)),
        attention_heads=int(cfg.get("attention_heads", 4)),
        attention_mlp_ratio=float(cfg.get("attention_mlp_ratio", 2.0)),
        dropout=float(cfg.get("dropout", 0.0)),
    ).to(device)
    audit = checkpoint_audit(model, checkpoint)
    model.eval()
    if audit["missing_baseline_keys"]:
        raise RuntimeError(
            "V1 baseline keys missing while loading Test15 checkpoint: "
            + repr(audit["missing_baseline_keys"])
        )
    if audit["unexpected_checkpoint_keys"]:
        raise RuntimeError(
            "Unexpected checkpoint model keys: "
            + repr(audit["unexpected_checkpoint_keys"])
        )
    return model, checkpoint, audit


def enable_memory_efficient_attention(model: CoRotTokenTransolver) -> str:
    """Use PyTorch SDPA without materializing an N-by-N score matrix.

    Test15 follows V1's full-field evaluation and some samples contain tens of
    thousands of geometric centers.  The learned attention definition is
    unchanged (same q/k/v projections, scale, softmax, and output projection);
    only the runtime kernel is changed from an explicit score tensor to
    PyTorch's Flash/memory-efficient scaled-dot-product attention.  The normal
    model implementation remains untouched and this hook is evaluation-only.
    """

    def sdpa_forward(self: PhysicsAttention, tokens: torch.Tensor, attention_allow=None):
        squeeze = False
        if tokens.ndim == 2:
            tokens = tokens.unsqueeze(0)
            squeeze = True
        if tokens.ndim != 3 or tokens.shape[-1] != self.dim:
            raise ValueError(
                f"tokens must be [N,{self.dim}] or [B,N,{self.dim}], got {tuple(tokens.shape)}"
            )
        if attention_allow is not None:
            # The Test15 evaluator always sends one complete sample/time group
            # and therefore has no mask.  Refuse a dense mask rather than
            # silently recreating the O(N^2) memory failure.
            raise RuntimeError("memory-efficient Test15 attention requires attention_allow=None")
        b, n, _ = tokens.shape
        qkv = self.qkv(tokens).reshape(b, n, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=float(self.dropout.p) if self.training else 0.0,
            is_causal=False,
            scale=self.scale,
        )
        out = out.transpose(1, 2).reshape(b, n, self.dim)
        out = self.proj(out)
        out = self.dropout(out)
        return out.squeeze(0) if squeeze else out

    count = 0
    for module in model.modules():
        if isinstance(module, PhysicsAttention):
            module.forward = MethodType(sdpa_forward, module)
            count += 1
    if count == 0:
        raise RuntimeError("No PhysicsAttention modules found for Test15 SDPA hook")
    return f"torch_sdpa_memory_efficient ({count} attention modules)"


def protocol_audit(records, cache_dir: Path, checkpoint: Path, v1_result_dir: Path) -> dict[str, Any]:
    sample_rows = []
    for rec in records:
        sample = PreparedSample(rec)
        sample_rows.append(
            {
                "sample_id": int(rec["sample_id"]),
                "frames": int(sample.T),
                "transitions": int(sample.T - 1),
                "geometric_centers": int(sample.N),
                "material_states_per_frame": int(2 * sample.N),
            }
        )
    max_centers = max(row["geometric_centers"] for row in sample_rows)
    max_tokens = max(row["material_states_per_frame"] for row in sample_rows)
    return {
        "protocol": "V1 Test15 full-field protocol, reused by V2",
        "test_sample_ids": [int(rec["sample_id"]) for rec in records],
        "number_of_test_samples": len(records),
        "samples": sample_rows,
        "number_of_frames": "181 state frames (frame 0 initialization plus frames 1..180)",
        "material_states_evaluated": "both surfaces for every geometric center",
        "surface_treatment": "surface 0=SPOS/outer, surface 1=SNEG/inner; no aggregation before metrics",
        "spatial_sampling": "all geometric centers in each PreparedSample; no subset or resampling",
        "training_tokens_per_sequence": 2048,
        "training_geometric_centers_per_sequence": 1024,
        "test_full_field_geometric_centers_max": max_centers,
        "test_full_field_tokens_max": max_tokens,
        "full_field_attention_sequence_length_shift": True,
        "teacher_forcing": "predict t+1 from GT z_t, then restore GT z_{t+1} for the next call",
        "rollout_initialization": "GT z_0=[S4,PE4,PEEQ] for both surfaces",
        "rollout": "autoregressive z_0 -> z_hat_1 -> ... -> z_hat_180; no reset or correction",
        "recursive_state": "[S4, PE4, PEEQ] (9 values)",
        "auxiliary_output": "LE4 predicted and scored, never fed back recursively",
        "mises": "analytical reconstruction from predicted S4; no Mises head",
        "metric_reduction": "V1 MetricAccumulator; pooled totals plus per-sample mean/median/std",
        "physical_units": {"S": "MPa", "Mises": "MPa", "PEEQ": "dimensionless", "LE": "dimensionless"},
        "cache_dir": str(cache_dir),
        "checkpoint": str(checkpoint),
        "v1_result_dir": str(v1_result_dir),
    }


def evaluate_teacher_forcing(model, records, stats, device, threshold: float):
    global_metric = MetricAccumulator()
    rows = []
    frame_acc: dict[int, MetricAccumulator] = {}
    with torch.no_grad():
        for rec in records:
            sample = PreparedSample(rec)
            centers = np.arange(sample.N, dtype=np.int64)
            builder = TokenDataBuilder(stats)
            sample_metric = MetricAccumulator()
            for t in range(sample.T - 1):
                batch = builder.build(sample, t, centers, shuffle=False).to(device)
                out = model(batch)
                pred_state, pred_le = predicted_state_and_le(
                    model, out, batch.state, stats, threshold
                )
                pred_state_np = pred_state.cpu().numpy()
                pred_le_np = pred_le.cpu().numpy()
                true_state_np = batch.next_state.cpu().numpy()
                true_le_np = batch.le_next.cpu().numpy()
                gate = torch.sigmoid(out["plastic_logit"].float()).cpu().numpy()
                delta_peeq = batch.delta_peeq.cpu().numpy()
                target = frame_acc.setdefault(t + 1, MetricAccumulator())
                target.update(pred_state_np, true_state_np, pred_le_np, true_le_np, gate, delta_peeq)
                sample_metric.update(pred_state_np, true_state_np, pred_le_np, true_le_np, gate, delta_peeq)
                global_metric.update(pred_state_np, true_state_np, pred_le_np, true_le_np, gate, delta_peeq)
            rows.append(metric_row(int(rec["sample_id"]), sample_metric.as_dict()))
    frame_rows = []
    for frame in sorted(frame_acc):
        frame_rows.append({"frame": frame, **{f"mae_{k}": v for k, v in scalar_metrics(frame_acc[frame].as_dict()).items()}})
    return global_metric, rows, frame_rows


def evaluate_rollout(model, records, stats, device, threshold: float):
    global_metric = MetricAccumulator()
    rows = []
    frame_acc: dict[int, MetricAccumulator] = {}
    per_sample_frames: dict[int, list[dict[str, Any]]] = {}
    with torch.no_grad():
        for rec in records:
            sample = PreparedSample(rec)
            centers = np.arange(sample.N, dtype=np.int64)
            builder = TokenDataBuilder(stats)
            pred = np.stack(
                [np.asarray(sample.state_outer[0], dtype=np.float32), np.asarray(sample.state_inner[0], dtype=np.float32)],
                axis=0,
            ).copy()
            sample_metric = MetricAccumulator()
            sample_frame_rows = []
            for t in range(sample.T - 1):
                # TokenDataBuilder's canonical order is [outer_0, inner_0,
                # outer_1, inner_1, ...].  ``pred`` is stored as [surface,
                # center, component], so flatten the surface dimension only
                # after interleaving the two surfaces per center.
                override = surface_center_to_canonical_states(pred)
                batch = builder.build(sample, t, centers, shuffle=False, state_override=override).to(device)
                expected_center = torch.as_tensor(np.repeat(centers, 2), device=device, dtype=torch.long)
                expected_surface = torch.as_tensor(np.tile(np.asarray([0, 1], dtype=np.int64), len(centers)), device=device, dtype=torch.long)
                expected_material = 2 * expected_center + expected_surface
                if not torch.equal(batch.center_id, expected_center) or not torch.equal(batch.surface_id, expected_surface) or not torch.equal(batch.material_state_index, expected_material):
                    raise RuntimeError(f"canonical token metadata mismatch at sample={rec['sample_id']} frame={t}")
                out = model(batch)
                pred_state, pred_le = predicted_state_and_le(model, out, batch.state, stats, threshold)
                pred_state_np = pred_state.cpu().numpy()
                pred_le_np = pred_le.cpu().numpy()
                canonical = batch.material_state_index.cpu().numpy()
                next_pred = token_rows_to_surface_center_states(
                    pred_state_np,
                    batch.center_position.cpu().numpy(),
                    batch.surface_id.cpu().numpy(),
                    len(centers),
                )
                next_le = token_rows_to_surface_center_states(
                    pred_le_np,
                    batch.center_position.cpu().numpy(),
                    batch.surface_id.cpu().numpy(),
                    len(centers),
                )
                pred = next_pred
                gt_state_surface = np.stack(
                    [np.asarray(sample.state_outer[t + 1], dtype=np.float32)[centers], np.asarray(sample.state_inner[t + 1], dtype=np.float32)[centers]],
                    axis=0,
                )
                gt_le_surface = np.stack(
                    [np.asarray(sample.LE_outer[t + 1], dtype=np.float32)[centers], np.asarray(sample.LE_inner[t + 1], dtype=np.float32)[centers]],
                    axis=0,
                )
                pred_metric = surface_center_to_canonical_states(pred)
                next_le_metric = surface_center_to_canonical_states(next_le)
                gt_state = surface_center_to_canonical_states(gt_state_surface)
                gt_le = surface_center_to_canonical_states(gt_le_surface)
                gate = torch.sigmoid(out["plastic_logit"].float()).cpu().numpy()
                delta_peeq = batch.delta_peeq.cpu().numpy()
                fm = frame_acc.setdefault(t + 1, MetricAccumulator())
                sm = MetricAccumulator()
                sm.update(pred_metric, gt_state, next_le_metric, gt_le, gate, delta_peeq)
                fm.update(pred_metric, gt_state, next_le_metric, gt_le, gate, delta_peeq)
                sample_metric.update(pred_metric, gt_state, next_le_metric, gt_le, gate, delta_peeq)
                global_metric.update(pred_metric, gt_state, next_le_metric, gt_le, gate, delta_peeq)
                sample_frame_rows.append({"frame": t + 1, **{f"mae_{k}": v for k, v in scalar_metrics(sm.as_dict()).items()}})
            sid = int(rec["sample_id"])
            rows.append(metric_row(sid, sample_metric.as_dict()))
            per_sample_frames[sid] = sample_frame_rows
    frame_rows = []
    for frame in sorted(frame_acc):
        frame_rows.append({"frame": frame, **{f"mae_{k}": v for k, v in scalar_metrics(frame_acc[frame].as_dict()).items()}})
    return global_metric, rows, frame_rows, per_sample_frames


def load_v1_results(v1_result_dir: Path):
    tf_path = v1_result_dir / "test_teacher_forcing.json"
    roll_path = v1_result_dir / "test_rollout" / "rollout_summary.json"
    if not tf_path.is_file() or not roll_path.is_file():
        raise FileNotFoundError(f"V1 Test15 results missing: {tf_path}, {roll_path}")
    return json.loads(tf_path.read_text()), json.loads(roll_path.read_text())


def comparison_rows(v1_tf, v1_roll, v2_tf_metric, v2_roll_metric, v2_tf_rows, v2_roll_rows):
    def pooled(d):
        return scalar_metrics(d["global"])

    v1_modes = {"teacher_forcing": pooled(v1_tf), "rollout": pooled(v1_roll)}
    v2_modes = {"teacher_forcing": scalar_metrics(v2_tf_metric.as_dict()), "rollout": scalar_metrics(v2_roll_metric.as_dict())}
    rows = []
    for mode in ("teacher_forcing", "rollout"):
        for key in SCALAR_NAMES:
            v1 = v1_modes[mode][key]
            v2 = v2_modes[mode][key]
            rows.append({"mode": mode, "metric": key, "v1": v1, "v2": v2, "absolute_change_v2_minus_v1": v2 - v1, "relative_change_percent": 100.0 * (v2 - v1) / v1 if v1 != 0 else float("nan")})
    v1_tf_by = {int(row["sample_id"]): row for row in v1_tf["samples"]}
    v1_roll_by = {int(row["sample_id"]): row["metrics"] for row in v1_roll["samples"]}
    v2_tf_by = {int(row["sample_id"]): row for row in v2_tf_rows}
    v2_roll_by = {int(row["sample_id"]): row for row in v2_roll_rows}
    per_sample = []
    better = {"teacher_forcing": {key: 0 for key in SCALAR_NAMES}, "rollout": {key: 0 for key in SCALAR_NAMES}}
    worse = {"teacher_forcing": {key: 0 for key in SCALAR_NAMES}, "rollout": {key: 0 for key in SCALAR_NAMES}}
    for sid in sorted(v2_roll_by):
        for mode, v1d, v2d in (
            ("teacher_forcing", v1_tf_by[sid], v2_tf_by[sid]),
            ("rollout", v1_roll_by[sid], v2_roll_by[sid]),
        ):
            v1s = scalar_metrics(v1d)
            v2s = {key: float(v2d[f"mae_{key}"]) for key in SCALAR_NAMES}
            if mode == "rollout":
                row = {"sample_id": sid}
                for key in SCALAR_NAMES:
                    row[f"v1_rollout_{key}"] = v1s[key]
                    row[f"v2_rollout_{key}"] = v2s[key]
                per_sample.append(row)
            for key in SCALAR_NAMES:
                if v2s[key] < v1s[key]:
                    better[mode][key] += 1
                elif v2s[key] > v1s[key]:
                    worse[mode][key] += 1
    return rows, per_sample, {"better": better, "worse": worse}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--v1-result-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--gate-threshold", type=float, default=0.5)
    ap.add_argument("--max-samples", type=int, default=0, help="debug only; 0 evaluates all 15 Test15 samples")
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    t0 = time.time()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    manifest = load_manifest(args.cache_dir)
    records = split_records(manifest, "test")
    if args.max_samples > 0:
        records = records[: args.max_samples]
    expected_ids = [0, 10, 16, 20, 27, 40, 41, 50, 69, 71, 79, 81, 84, 114, 119]
    if args.max_samples <= 0 and [int(r["sample_id"]) for r in records] != expected_ids:
        raise RuntimeError(f"Test15 IDs differ from V1 protocol: {[r['sample_id'] for r in records]}")

    stats = load_stats(args.cache_dir / "stats.json")
    model, checkpoint, ck_audit = load_model(args.checkpoint, device)
    attention_backend = enable_memory_efficient_attention(model)
    ck_audit["evaluation_attention_backend"] = attention_backend
    protocol = protocol_audit(records, args.cache_dir, args.checkpoint, args.v1_result_dir)
    (args.output_dir / "evaluation_protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    (args.output_dir / "checkpoint_audit.json").write_text(json.dumps(ck_audit, indent=2), encoding="utf-8")
    v1_tf, v1_roll = load_v1_results(args.v1_result_dir)

    tf_metric, tf_rows, tf_frame_rows = evaluate_teacher_forcing(model, records, stats, device, args.gate_threshold)
    roll_metric, roll_rows, roll_frame_rows, roll_sample_frames = evaluate_rollout(model, records, stats, device, args.gate_threshold)

    tf_summary = {"mode": "teacher_forcing", "checkpoint": str(args.checkpoint), "checkpoint_epoch": checkpoint.get("epoch"), "pooled": scalar_metrics(tf_metric.as_dict()), "pooled_v1_metric": tf_metric.as_dict(), "per_sample_distribution": distribution(tf_rows), "sample_count": len(tf_rows)}
    roll_summary = {"mode": "181_frame_autoregressive_rollout", "checkpoint": str(args.checkpoint), "checkpoint_epoch": checkpoint.get("epoch"), "pooled": scalar_metrics(roll_metric.as_dict()), "pooled_v1_metric": roll_metric.as_dict(), "per_sample_distribution": distribution(roll_rows), "sample_count": len(roll_rows), "frames": roll_frame_rows}
    (args.output_dir / "teacher_forcing_summary.json").write_text(json.dumps(tf_summary, indent=2), encoding="utf-8")
    (args.output_dir / "rollout_summary.json").write_text(json.dumps(roll_summary, indent=2), encoding="utf-8")
    write_csv(args.output_dir / "teacher_forcing_per_sample.csv", tf_rows)
    write_csv(args.output_dir / "rollout_per_sample.csv", roll_rows)
    write_csv(args.output_dir / "teacher_forcing_per_frame.csv", tf_frame_rows)
    write_csv(args.output_dir / "rollout_per_frame.csv", roll_frame_rows)

    comparison, per_sample, counts = comparison_rows(v1_tf, v1_roll, tf_metric, roll_metric, tf_rows, roll_rows)
    write_csv(args.output_dir / "v1_v2_comparison.csv", comparison)
    write_csv(args.output_dir / "v1_v2_per_sample_rollout.csv", per_sample)
    comparison_summary = {"pooled_comparison": comparison, "per_sample_rollout": per_sample, "counts": counts}
    (args.output_dir / "v1_v2_comparison.json").write_text(json.dumps(comparison_summary, indent=2), encoding="utf-8")
    for sid, rows in roll_sample_frames.items():
        write_csv(args.output_dir / f"sample_{sid:04d}_rollout_per_frame.csv", rows)

    finite = all(np.isfinite(float(value)) for summary in (tf_summary["pooled"], roll_summary["pooled"]) for value in summary.values())
    memory = {}
    if device.type == "cuda":
        memory = {
            "peak_allocated_mb": float(torch.cuda.max_memory_allocated(device) / 1024**2),
            "peak_reserved_mb": float(torch.cuda.max_memory_reserved(device) / 1024**2),
        }
    final = {"status": "PASS" if finite else "FAIL", "nan_inf": not finite, "gpu_memory": memory, "checkpoint_audit": ck_audit, "test_sample_ids": [int(r["sample_id"]) for r in records], "teacher_forcing": tf_summary["pooled"], "rollout": roll_summary["pooled"], "v1_vs_v2_counts": counts, "seconds": time.time() - t0}
    (args.output_dir / "evaluation_status.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
