#!/usr/bin/env python3
"""Compare teacher-forcing and rollout frame-1 paths on identical inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from corot_lrdsun.metrics import MetricAccumulator
from corot_lrdsun_v2 import TokenDataBuilder
from corot_lrdsun_v2.v1_compat import (
    PreparedSample,
    delta8_from_norm,
    le_from_norm,
    load_manifest,
    load_stats,
    split_records,
)
from evaluate_test15 import enable_memory_efficient_attention, load_model


def diff(a: Any, b: Any) -> dict[str, Any]:
    if isinstance(a, torch.Tensor):
        if not isinstance(b, torch.Tensor) or tuple(a.shape) != tuple(b.shape):
            return {"same": False, "shape_a": list(a.shape), "shape_b": list(getattr(b, "shape", ())), "max_abs_diff": None}
        if a.dtype == torch.bool or b.dtype == torch.bool or a.dtype in (torch.int64, torch.int32, torch.int16, torch.int8):
            same = bool(torch.equal(a, b))
            return {"same": same, "shape": list(a.shape), "max_abs_diff": 0.0 if same else 1.0}
        d = (a.float() - b.float()).abs()
        return {"same": bool(torch.equal(a, b)), "shape": list(a.shape), "max_abs_diff": float(d.max().cpu()), "mean_abs_diff": float(d.mean().cpu()), "relative_l2_diff": float((torch.linalg.vector_norm(d) / torch.linalg.vector_norm(a.float()).clamp_min(1.0e-30)).cpu())}
    aa = np.asarray(a)
    bb = np.asarray(b)
    if aa.shape != bb.shape:
        return {"same": False, "shape_a": list(aa.shape), "shape_b": list(bb.shape), "max_abs_diff": None}
    if aa.dtype == np.bool_ or np.issubdtype(aa.dtype, np.integer):
        same = bool(np.array_equal(aa, bb))
        return {"same": same, "shape": list(aa.shape), "max_abs_diff": 0.0 if same else 1.0}
    dd = np.abs(aa.astype(np.float64) - bb.astype(np.float64))
    return {"same": bool(np.array_equal(aa, bb)), "shape": list(aa.shape), "max_abs_diff": float(dd.max()) if dd.size else 0.0, "mean_abs_diff": float(dd.mean()) if dd.size else 0.0, "relative_l2_diff": float(np.linalg.norm(dd) / max(np.linalg.norm(aa.astype(np.float64)), 1.0e-30))}


def metric_scalars(metric: dict[str, Any]) -> dict[str, float]:
    return {
        "S": float(np.mean(metric["mae_S_components_MPa"])),
        "Mises": float(metric["mae_S_mises_MPa"]),
        "PEEQ": float(metric["mae_PEEQ"]),
        "LE": float(np.mean(metric["mae_LE_components"])),
    }


def prediction(model, batch, stats, threshold=0.5):
    out = model(batch)
    d8 = delta8_from_norm(out["delta8_norm"].float(), stats)
    p_scaled = model.peeq_scaled_hard(out, threshold).float()
    pred_state = torch.cat([batch.state[:, :8].float() + d8, batch.state[:, 8:9].float() + p_scaled * float(stats["peeq_delta_scale"])], dim=1)
    pred_le = le_from_norm(out["le_norm"].float(), stats)
    return out, pred_state, pred_le


def audit_case(model, builder, sample, stats, device, centers: np.ndarray, threshold: float) -> dict[str, Any]:
    # Both paths are explicitly constructed from the same GT z_0 and the same
    # t=0 transition.  The rollout path uses the exact state_override route
    # used by evaluate_token_rollout.py.
    z0 = np.stack([np.asarray(sample.state_outer[0])[centers], np.asarray(sample.state_inner[0])[centers]], axis=1).reshape(-1, 9).astype(np.float32, copy=True)
    tf = builder.build(sample, 0, centers, shuffle=False).to(device)
    ro = builder.build(sample, 0, centers, shuffle=False, state_override=z0).to(device)
    fields = ("edge_unique_n", "edge_unique", "edge_mask", "edge_center_index", "state_n", "context_n", "state", "gt_state_t", "next_state", "delta8", "delta_peeq", "le_next", "center_id", "surface_id", "material_state_index", "center_position", "geometric_centers")
    batch_diff = {name: diff(getattr(tf, name), getattr(ro, name)) for name in fields}
    batch_diff["sample_id"] = {"tf": int(tf.sample_id), "rollout": int(ro.sample_id), "same": int(tf.sample_id) == int(ro.sample_id)}
    batch_diff["time_index"] = {"tf": int(tf.time_index), "rollout": int(ro.time_index), "same": int(tf.time_index) == int(ro.time_index)}

    with torch.no_grad():
        tf_out, tf_state, tf_le = prediction(model, tf, stats, threshold)
        ro_out, ro_state, ro_le = prediction(model, ro, stats, threshold)
    output_diff = {key: diff(tf_out[key], ro_out[key]) for key in ("delta8_norm", "plastic_logit", "peeq_mag_raw", "le_norm")}
    output_diff["updated_state"] = diff(tf_state, ro_state)
    output_diff["updated_le"] = diff(tf_le, ro_le)
    tf_s4 = tf_state[:, :4]
    ro_s4 = ro_state[:, :4]
    tf_vm = torch.sqrt(torch.clamp(0.5 * ((tf_s4[:, 0] - tf_s4[:, 1]) ** 2 + (tf_s4[:, 1] - tf_s4[:, 2]) ** 2 + (tf_s4[:, 2] - tf_s4[:, 0]) ** 2) + 3.0 * tf_s4[:, 3] ** 2, min=0.0))
    ro_vm = torch.sqrt(torch.clamp(0.5 * ((ro_s4[:, 0] - ro_s4[:, 1]) ** 2 + (ro_s4[:, 1] - ro_s4[:, 2]) ** 2 + (ro_s4[:, 2] - ro_s4[:, 0]) ** 2) + 3.0 * ro_s4[:, 3] ** 2, min=0.0))
    output_diff["updated_peeq"] = diff(tf_state[:, 8:9], ro_state[:, 8:9])
    output_diff["Mises"] = diff(tf_vm, ro_vm)

    tf_metric = MetricAccumulator()
    ro_metric = MetricAccumulator()
    tf_np = tf_state.cpu().numpy(); ro_np = ro_state.cpu().numpy()
    tf_le_np = tf_le.cpu().numpy(); ro_le_np = ro_le.cpu().numpy()
    true_state = tf.next_state.cpu().numpy(); true_le = tf.le_next.cpu().numpy()
    tf_gate = torch.sigmoid(tf_out["plastic_logit"].float()).cpu().numpy()
    ro_gate = torch.sigmoid(ro_out["plastic_logit"].float()).cpu().numpy()
    delta_peeq = tf.delta_peeq.cpu().numpy()
    tf_metric.update(tf_np, true_state, tf_le_np, true_le, tf_gate, delta_peeq)
    ro_metric.update(ro_np, true_state, ro_le_np, true_le, ro_gate, delta_peeq)
    metric_tf = metric_scalars(tf_metric.as_dict())
    metric_ro = metric_scalars(ro_metric.as_dict())
    metric_diff = {key: metric_ro[key] - metric_tf[key] for key in metric_tf}

    return {
        "geometric_centers": int(len(centers)),
        "material_tokens": int(tf.num_tokens),
        "center_ids_equal": bool(torch.equal(tf.center_id, ro.center_id)),
        "material_state_indices_equal": bool(torch.equal(tf.material_state_index, ro.material_state_index)),
        "frame_index": {"tf_input": 0, "tf_target": 1, "rollout_input": 0, "rollout_target": 1, "same": True},
        "z0_initialization": {"source": "GT state_outer[0]/state_inner[0]", "max_abs_diff_rollout_override_vs_gt": float(np.max(np.abs(z0 - tf.gt_state_t.cpu().numpy()))), "S4_equal": bool(np.array_equal(z0[:, :4], tf.gt_state_t.cpu().numpy()[:, :4])), "PE4_equal": bool(np.array_equal(z0[:, 4:8], tf.gt_state_t.cpu().numpy()[:, 4:8])), "PEEQ_equal": bool(np.array_equal(z0[:, 8:9], tf.gt_state_t.cpu().numpy()[:, 8:9]))},
        "token_batch_diff": batch_diff,
        "prediction_diff": output_diff,
        "tf_frame1_metrics": metric_tf,
        "rollout_frame1_metrics": metric_ro,
        "metric_difference_rollout_minus_tf": metric_diff,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--sample-id", type=int, default=119)
    ap.add_argument("--centers", type=int, default=1024)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("first-step audit requires CUDA")
    device = torch.device("cuda")
    stats = load_stats(args.cache_dir / "stats.json")
    records = [r for r in split_records(load_manifest(args.cache_dir), "test") if int(r["sample_id"]) == args.sample_id]
    if len(records) != 1:
        raise RuntimeError(f"expected exactly one sample {args.sample_id}")
    sample = PreparedSample(records[0])
    model, checkpoint, audit = load_model(args.checkpoint, device)
    attention_backend = enable_memory_efficient_attention(model)
    model.eval()
    builder = TokenDataBuilder(stats)
    c1024 = np.arange(min(int(args.centers), sample.N), dtype=np.int64)
    cfull = np.arange(sample.N, dtype=np.int64)
    cases = {"C1024": audit_case(model, builder, sample, stats, device, c1024, 0.5), "full_field": audit_case(model, builder, sample, stats, device, cfull, 0.5)}
    result = {
        "status": "PASS" if all(all(v.get("same", True) for v in case["token_batch_diff"].values() if isinstance(v, dict) and "same" in v) and all(v.get("max_abs_diff", 0.0) <= 5.0e-5 for v in case["prediction_diff"].values()) for case in cases.values()) else "FAIL",
        "sample_id": int(args.sample_id),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_audit": {"missing_baseline_keys": audit["missing_baseline_keys"], "unexpected_checkpoint_keys": audit["unexpected_checkpoint_keys"]},
        "attention_backend": attention_backend,
        "training_path_contract": "TF and rollout frame 1 both build t=0 -> t+1 from GT z0, with identical all-center IDs and paired surfaces",
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
