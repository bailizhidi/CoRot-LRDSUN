#!/usr/bin/env python3
"""Full-field frame-1 tensor equivalence gate for all Test15 samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from audit_first_step_equivalence import audit_case, enable_memory_efficient_attention, load_model
from corot_lrdsun_v2 import TokenDataBuilder, surface_center_to_canonical_states
from corot_lrdsun_v2.v1_compat import PreparedSample, load_manifest, load_stats, split_records


def surface_regression() -> dict:
    n = 7
    values = np.zeros((2, n, 2), dtype=np.float32)
    for i in range(n):
        values[0, i] = (1000 + i, 1000 + i + 0.25)
        values[1, i] = (2000 + i, 2000 + i + 0.25)
    canonical = surface_center_to_canonical_states(values)
    expected = np.stack([values[0], values[1]], axis=1).reshape(-1, 2)
    center_id = np.repeat(np.arange(n, dtype=np.int64), 2)
    surface_id = np.tile(np.asarray([0, 1], dtype=np.int64), n)
    material_state_index = 2 * center_id + surface_id
    return {
        "status": "PASS" if np.array_equal(canonical, expected) and np.array_equal(material_state_index, np.arange(2 * n)) else "FAIL",
        "input_shape": list(values.shape),
        "canonical_shape": list(canonical.shape),
        "canonical_values": canonical.tolist(),
        "expected_values": expected.tolist(),
        "center_id": center_id.tolist(),
        "surface_id": surface_id.tolist(),
        "material_state_index": material_state_index.tolist(),
        "mapping": "SPOS_0,SNEG_0,SPOS_1,SNEG_1,...",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("all-sample first-step audit requires CUDA")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    stats = load_stats(args.cache_dir / "stats.json")
    records = split_records(load_manifest(args.cache_dir), "test")
    model, checkpoint, audit = load_model(args.checkpoint, device)
    backend = enable_memory_efficient_attention(model)
    builder = TokenDataBuilder(stats)
    rows = []
    for rec in records:
        sample = PreparedSample(rec)
        centers = np.arange(sample.N, dtype=np.int64)
        case = audit_case(model, builder, sample, stats, device, centers, 0.5)
        max_prediction_diff = max(float(v.get("max_abs_diff", 0.0)) for v in case["prediction_diff"].values())
        batch_same = all(bool(v.get("same", True)) for v in case["token_batch_diff"].values() if isinstance(v, dict) and "same" in v)
        row = {
            "sample_id": int(rec["sample_id"]),
            "geometric_centers": int(sample.N),
            "material_tokens": int(2 * sample.N),
            "max_abs_diff": max_prediction_diff,
            "mean_abs_diff": max(float(v.get("mean_abs_diff", 0.0)) for v in case["prediction_diff"].values()),
            "relative_L2_diff": max(float(v.get("relative_l2_diff", 0.0)) for v in case["prediction_diff"].values()),
            "token_batch_equal": batch_same,
            "z0_equal": bool(case["z0_initialization"]["max_abs_diff_rollout_override_vs_gt"] == 0.0),
            "center_ids_equal": bool(case["center_ids_equal"]),
            "material_state_ids_equal": bool(case["material_state_indices_equal"]),
            "frame_indexing_equal": bool(case["frame_index"]["same"]),
            "tf_frame1_metrics": case["tf_frame1_metrics"],
            "rollout_frame1_metrics": case["rollout_frame1_metrics"],
        }
        row["status"] = "PASS" if row["max_abs_diff"] <= 5.0e-5 and all(row[k] for k in ("token_batch_equal", "z0_equal", "center_ids_equal", "material_state_ids_equal", "frame_indexing_equal")) else "FAIL"
        rows.append(row)
    surface = surface_regression()
    (args.output_dir / "surface_order_regression_test.json").write_text(json.dumps(surface, indent=2), encoding="utf-8")
    result = {
        "status": "PASS" if surface["status"] == "PASS" and all(row["status"] == "PASS" for row in rows) else "FAIL",
        "sample_count": len(rows),
        "all_samples_pass": all(row["status"] == "PASS" for row in rows),
        "surface_order_regression": surface,
        "attention_backend": backend,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_audit": {"missing_baseline_keys": audit["missing_baseline_keys"], "unexpected_checkpoint_keys": audit["unexpected_checkpoint_keys"]},
        "samples": rows,
    }
    (args.output_dir / "first_step_all_samples.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
