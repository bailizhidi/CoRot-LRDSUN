#!/usr/bin/env python3
"""Matched-token Test15 diagnostic with deterministic full coverage.

Every attention call contains at most 1024 geometric centers / 2048 paired
material tokens.  Groups are fixed per sample for the entire rollout and all
centers are updated exactly once per frame.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from corot_lrdsun.metrics import MetricAccumulator
from corot_lrdsun_v2 import (
    TokenDataBuilder,
    surface_center_to_canonical_states,
    token_rows_to_surface_center_states,
)
from corot_lrdsun_v2.v1_compat import PreparedSample, load_manifest, load_stats, split_records
from evaluate_test15 import (
    enable_memory_efficient_attention,
    load_model,
    predicted_state_and_le,
    scalar_metrics,
    write_csv,
)


MAX_CENTERS = 1024
REGION_RATIOS = (0.25, 0.50, 0.25)


def deterministic_groups(sample: PreparedSample, max_centers: int = MAX_CENTERS) -> list[np.ndarray]:
    """Partition every center once, approximately matching V1's region ratios."""
    region = np.asarray(sample.region_id).reshape(-1)
    pools = [list(np.flatnonzero(region == r).astype(np.int64)) for r in range(3)]
    # Unknown region labels are assigned to the largest remaining pool later.
    known = sum(len(pool) for pool in pools)
    if known < sample.N:
        unknown = [int(x) for x in np.flatnonzero(~np.isin(region, [0, 1, 2]))]
        pools[1].extend(unknown)
    groups: list[np.ndarray] = []
    remaining = int(sample.N)
    while remaining:
        target = min(int(max_centers), remaining)
        quotas = [int(round(target * ratio)) for ratio in REGION_RATIOS[:2]]
        quotas.append(target - sum(quotas))
        selected: list[int] = []
        for rid, quota in enumerate(quotas):
            take = min(quota, len(pools[rid]))
            selected.extend(pools[rid][:take])
            del pools[rid][:take]
        deficit = target - len(selected)
        while deficit:
            candidates = sorted(range(3), key=lambda rid: (-len(pools[rid]), rid))
            rid = next((rid for rid in candidates if pools[rid]), None)
            if rid is None:
                raise RuntimeError("center partition lost coverage")
            take = min(deficit, len(pools[rid]))
            selected.extend(pools[rid][:take])
            del pools[rid][:take]
            deficit -= take
        group = np.asarray(sorted(selected), dtype=np.int64)
        if group.size != target or np.unique(group).size != target:
            raise RuntimeError("invalid deterministic center group")
        groups.append(group)
        remaining -= target
    flat = np.concatenate(groups) if groups else np.empty(0, dtype=np.int64)
    if not np.array_equal(np.sort(flat), np.arange(sample.N, dtype=np.int64)):
        raise RuntimeError("center partition is not a full one-to-one cover")
    return groups


def mapping_record(sample: PreparedSample, groups: list[np.ndarray]) -> dict[str, Any]:
    return {
        "sample_id": int(sample.meta["sample_id"]),
        "total_geometric_centers": int(sample.N),
        "full_field_material_tokens": int(2 * sample.N),
        "max_group_geometric_centers": MAX_CENTERS,
        "max_group_material_tokens": 2 * MAX_CENTERS,
        "number_of_groups": len(groups),
        "group_sizes": [int(len(group)) for group in groups],
        "last_group_size": int(len(groups[-1])) if groups else 0,
        "fraction_centers_in_short_last_group": float(len(groups[-1]) / MAX_CENTERS) if groups and len(groups[-1]) < MAX_CENTERS else 1.0,
        "groups": [{"group_id": gid, "center_ids": group.tolist()} for gid, group in enumerate(groups)],
    }


def assert_batch_metadata(batch, centers: np.ndarray, device: torch.device) -> None:
    expected_center = torch.as_tensor(np.repeat(centers, 2), device=device, dtype=torch.long)
    expected_surface = torch.as_tensor(np.tile(np.asarray([0, 1], dtype=np.int64), len(centers)), device=device, dtype=torch.long)
    expected_position = torch.as_tensor(np.repeat(np.arange(len(centers), dtype=np.int64), 2), device=device, dtype=torch.long)
    expected_material = 2 * expected_center + expected_surface
    if not torch.equal(batch.center_id, expected_center) or not torch.equal(batch.surface_id, expected_surface) or not torch.equal(batch.center_position, expected_position) or not torch.equal(batch.material_state_index, expected_material):
        raise RuntimeError(f"matched canonical metadata mismatch sample={batch.sample_id} time={batch.time_index}")


def initialize_surface_state(sample: PreparedSample) -> np.ndarray:
    return np.stack([np.asarray(sample.state_outer[0], dtype=np.float32), np.asarray(sample.state_inner[0], dtype=np.float32)], axis=0).copy()


def initialize_surface_le(sample: PreparedSample) -> np.ndarray:
    return np.zeros((2, sample.N, 4), dtype=np.float32)


def evaluate_mode(model, records, stats, device, groups_by_sample, mode: str, max_frames: int | None = None):
    global_metric = MetricAccumulator()
    sample_rows = []
    frame_acc: dict[int, MetricAccumulator] = {}
    sample_frames: dict[int, list[dict[str, Any]]] = {}
    finite = True
    coverage_ok = True
    duplicate_ok = True
    with torch.no_grad():
        for rec in records:
            sample = PreparedSample(rec)
            groups = groups_by_sample[int(rec["sample_id"])]
            builder = TokenDataBuilder(stats)
            centers_all = np.arange(sample.N, dtype=np.int64)
            pred = initialize_surface_state(sample)
            sample_metric = MetricAccumulator()
            rows_for_sample = []
            stop = sample.T - 1 if max_frames is None else min(sample.T - 1, int(max_frames))
            for t in range(stop):
                if mode == "rollout":
                    next_pred = np.empty_like(pred)
                    next_le = np.empty((2, sample.N, 4), dtype=np.float32)
                    seen = np.zeros(sample.N, dtype=np.int8)
                    gate_global = np.empty((2 * sample.N, 1), dtype=np.float32)
                    delta_global = np.empty((2 * sample.N, 1), dtype=np.float32)
                for group in groups:
                    if mode == "teacher_forcing":
                        override = None
                    else:
                        override = surface_center_to_canonical_states(pred[:, group])
                    batch = builder.build(sample, t, group, shuffle=False, state_override=override).to(device)
                    assert_batch_metadata(batch, group, device)
                    out = model(batch)
                    pred_state, pred_le = predicted_state_and_le(model, out, batch.state, stats, 0.5)
                    if not torch.isfinite(pred_state).all() or not torch.isfinite(pred_le).all():
                        finite = False
                        raise FloatingPointError(f"non-finite matched prediction sample={rec['sample_id']} frame={t}")
                    pred_state_np = pred_state.cpu().numpy()
                    pred_le_np = pred_le.cpu().numpy()
                    true_state = batch.next_state.cpu().numpy()
                    true_le = batch.le_next.cpu().numpy()
                    gate = torch.sigmoid(out["plastic_logit"].float()).cpu().numpy()
                    delta_peeq = batch.delta_peeq.cpu().numpy()
                    if mode == "teacher_forcing":
                        metric = frame_acc.setdefault(t + 1, MetricAccumulator())
                        metric.update(pred_state_np, true_state, pred_le_np, true_le, gate, delta_peeq)
                        sample_metric.update(pred_state_np, true_state, pred_le_np, true_le, gate, delta_peeq)
                        global_metric.update(pred_state_np, true_state, pred_le_np, true_le, gate, delta_peeq)
                    else:
                        positions = batch.center_position.cpu().numpy()
                        surfaces = batch.surface_id.cpu().numpy()
                        if np.any(seen[group]):
                            duplicate_ok = False
                            raise RuntimeError(f"duplicate matched center update sample={rec['sample_id']} frame={t}")
                        seen[group] = 1
                        next_pred[:, group] = token_rows_to_surface_center_states(pred_state_np, positions, surfaces, len(group))
                        next_le[:, group] = token_rows_to_surface_center_states(pred_le_np, positions, surfaces, len(group))
                        material = batch.material_state_index.cpu().numpy()
                        gate_global[material] = gate
                        delta_global[material] = delta_peeq
                if mode == "rollout":
                    coverage_ok = coverage_ok and bool(np.all(seen == 1))
                    if not np.all(seen == 1):
                        raise RuntimeError(f"matched groups did not cover all centers sample={rec['sample_id']} frame={t}")
                    pred = next_pred
                    gt_surface = np.stack([np.asarray(sample.state_outer[t + 1], dtype=np.float32), np.asarray(sample.state_inner[t + 1], dtype=np.float32)], axis=0)
                    le_surface = np.stack([np.asarray(sample.LE_outer[t + 1], dtype=np.float32), np.asarray(sample.LE_inner[t + 1], dtype=np.float32)], axis=0)
                    pred_metric = surface_center_to_canonical_states(pred)
                    le_metric = surface_center_to_canonical_states(next_le)
                    gt_metric = surface_center_to_canonical_states(gt_surface)
                    gt_le = surface_center_to_canonical_states(le_surface)
                    metric = frame_acc.setdefault(t + 1, MetricAccumulator())
                    metric.update(pred_metric, gt_metric, le_metric, gt_le, gate_global, delta_global)
                    sample_metric.update(pred_metric, gt_metric, le_metric, gt_le, gate_global, delta_global)
                    global_metric.update(pred_metric, gt_metric, le_metric, gt_le, gate_global, delta_global)
                frame_metric = frame_acc[t + 1].as_dict()
                rows_for_sample.append({"frame": t + 1, **{f"mae_{key}": value for key, value in scalar_metrics(frame_metric).items()}})
            sample_id = int(rec["sample_id"])
            sample_rows.append({"sample_id": sample_id, **{f"mae_{key}": value for key, value in scalar_metrics(sample_metric.as_dict()).items()}})
            sample_frames[sample_id] = rows_for_sample
    frame_rows = [{"frame": frame, **{f"mae_{key}": value for key, value in scalar_metrics(frame_acc[frame].as_dict()).items()}} for frame in sorted(frame_acc)]
    return {
        "metric": global_metric,
        "sample_rows": sample_rows,
        "frame_rows": frame_rows,
        "sample_frames": sample_frames,
        "finite": finite,
        "coverage_ok": coverage_ok,
        "duplicate_ok": duplicate_ok,
    }


def distribution(rows):
    return {key: {"mean": float(statistics.mean([row[f"mae_{key}"] for row in rows])), "median": float(statistics.median([row[f"mae_{key}"] for row in rows])), "std": float(statistics.pstdev([row[f"mae_{key}"] for row in rows]))} for key in ("S", "Mises", "PEEQ", "LE")}


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv_rows(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_summary(path: Path, mode: str, result: dict[str, Any], checkpoint, records):
    metric = result["metric"].as_dict()
    summary = {"mode": mode, "checkpoint": str(checkpoint), "checkpoint_epoch": 19, "pooled": scalar_metrics(metric), "pooled_v1_metric": metric, "per_sample_distribution": distribution(result["sample_rows"]), "sample_count": len(result["sample_rows"]), "frames": result["frame_rows"], "finite": result["finite"], "coverage_ok": result["coverage_ok"], "duplicate_ok": result["duplicate_ok"]}
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def three_way(out_dir: Path, matched_tf, matched_roll):
    full_dir = out_dir.parent / "test15_surface_order_fixed"
    v1_dir = Path("/data/home/scxk573/run/lsz/ysy/physicsnemo/my_experiments/CoRot-LRDSUN-Constitutive-V1/outputs/finetune_5step/run_20260920_143444_1610485")
    full_tf = load_json(full_dir / "teacher_forcing_summary.json")
    full_roll = load_json(full_dir / "rollout_summary.json")
    v1_tf = load_json(v1_dir / "test_teacher_forcing.json")
    v1_roll = load_json(v1_dir / "test_rollout" / "rollout_summary.json")
    rows = []
    for mode, v1d, fulld, mat in (("teacher_forcing", v1_tf["global"], full_tf["pooled"], matched_tf["pooled"]), ("rollout", v1_roll["global"], full_roll["pooled"], matched_roll["pooled"])):
        v1s = scalar_metrics(v1d)
        for key in ("S", "Mises", "PEEQ", "LE"):
            full = float(fulld[key]); matched = float(mat[key]); base = float(v1s[key])
            rows.append({"mode": mode, "metric": key, "v1": base, "v2_full_field": full, "v2_matched_token": matched, "matched_vs_full_relative_percent": 100.0 * (matched - full) / full if full else float("nan"), "matched_vs_v1_relative_percent": 100.0 * (matched - base) / base if base else float("nan")})
    write_csv(out_dir / "three_way_comparison.csv", rows)
    return rows


def per_sample_diagnostic(out_dir: Path):
    full_rows = {int(row["sample_id"]): row for row in load_csv_rows(out_dir.parent / "test15_surface_order_fixed" / "rollout_per_sample.csv")}
    matched_rows = {int(row["sample_id"]): row for row in load_csv_rows(out_dir / "rollout_per_sample.csv")}
    v1_dir = Path("/data/home/scxk573/run/lsz/ysy/physicsnemo/my_experiments/CoRot-LRDSUN-Constitutive-V1/outputs/finetune_5step/run_20260920_143444_1610485")
    v1_roll = load_json(v1_dir / "test_rollout" / "rollout_summary.json")
    v1_rows = {int(row["sample_id"]): scalar_metrics(row["metrics"]) for row in v1_roll["samples"]}
    mapping = load_json(out_dir / "matched_group_mapping.json")
    rows = []
    better_full = {key: 0 for key in ("S", "Mises", "PEEQ", "LE")}
    better_v1 = {key: 0 for key in ("S", "Mises", "PEEQ", "LE")}
    token_counts = []
    gaps = {key: [] for key in ("S", "Mises", "PEEQ", "LE")}
    for sid in sorted(matched_rows):
        full = {key: float(full_rows[sid][f"mae_{key}"]) for key in ("S", "Mises", "PEEQ", "LE")}
        matched = {key: float(matched_rows[sid][f"mae_{key}"]) for key in ("S", "Mises", "PEEQ", "LE")}
        v1 = v1_rows[sid]
        info = mapping[str(sid)]
        row = {"sample_id": sid, "total_geometric_centers": info["total_geometric_centers"], "full_field_material_tokens": info["full_field_material_tokens"], "number_of_matched_groups": info["number_of_groups"], "group_sizes": json.dumps(info["group_sizes"]), "last_group_size": info["last_group_size"]}
        for key in ("S", "Mises", "PEEQ", "LE"):
            row[f"full_field_{key}"] = full[key]
            row[f"matched_{key}"] = matched[key]
            row[f"v1_{key}"] = v1[key]
            row[f"matched_minus_full_{key}"] = matched[key] - full[key]
            row[f"matched_minus_v1_{key}"] = matched[key] - v1[key]
            gaps[key].append(matched[key] - full[key])
            if matched[key] < full[key]:
                better_full[key] += 1
            if matched[key] < v1[key]:
                better_v1[key] += 1
        token_counts.append(info["full_field_material_tokens"])
        rows.append(row)
    write_csv(out_dir / "sequence_size_diagnostic.csv", rows)

    def corr(values):
        x = np.asarray(token_counts, dtype=np.float64)
        y = np.asarray(values, dtype=np.float64)
        return float(np.corrcoef(x, y)[0, 1]) if len(x) > 1 and np.std(x) > 0 and np.std(y) > 0 else 0.0

    def rank(values):
        order = np.argsort(np.argsort(np.asarray(values, dtype=np.float64)))
        return order.astype(np.float64)

    correlations = {}
    for key, values in gaps.items():
        correlations[key] = {"pearson_token_count_vs_matched_minus_full": corr(values), "spearman_token_count_vs_matched_minus_full": corr(rank(values))}
    return {"matched_lower_than_full_field": better_full, "matched_lower_than_v1": better_v1, "correlation": correlations, "sample_count": len(rows)}


def frame_comparison(out_dir: Path):
    full = load_json(out_dir.parent / "test15_surface_order_fixed" / "rollout_summary.json")["frames"]
    matched = load_json(out_dir / "rollout_summary.json")["frames"]
    v1_dir = Path("/data/home/scxk573/run/lsz/ysy/physicsnemo/my_experiments/CoRot-LRDSUN-Constitutive-V1/outputs/finetune_5step/run_20260920_143444_1610485")
    v1 = load_json(v1_dir / "test_rollout" / "rollout_summary.json")
    v1_by_frame = {}
    for sample in v1["samples"]:
        for row in sample["frames"]:
            v1_by_frame.setdefault(int(row["frame"]), []).append(scalar_metrics(row))
    rows = []
    for fr, full_row, matched_row in zip((int(x["frame"]) for x in full), full, matched):
        mean_v1 = {key: float(statistics.mean([x[key] for x in v1_by_frame[fr]])) for key in ("S", "Mises", "PEEQ", "LE")}
        row = {"frame": fr}
        for key in ("S", "Mises", "PEEQ", "LE"):
            row[f"v1_{key}"] = mean_v1[key]
            row[f"v2_full_field_{key}"] = float(full_row[f"mae_{key}"])
            row[f"v2_matched_token_{key}"] = float(matched_row[f"mae_{key}"])
        rows.append(row)
    write_csv(out_dir / "frame_comparison.csv", rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--precheck", action="store_true")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("matched-token evaluation requires CUDA")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    start = time.time()
    stats = load_stats(args.cache_dir / "stats.json")
    records = split_records(load_manifest(args.cache_dir), "test")
    if args.precheck:
        records = [rec for rec in records if int(rec["sample_id"]) == 119]
    model, checkpoint, audit = load_model(args.checkpoint, device)
    backend = enable_memory_efficient_attention(model)
    groups_by_sample = {}
    mapping = {}
    for rec in records:
        sample = PreparedSample(rec)
        groups = deterministic_groups(sample)
        groups_by_sample[int(rec["sample_id"])] = groups
        mapping[str(int(rec["sample_id"]))] = mapping_record(sample, groups)
    (args.output_dir / "matched_group_mapping.json").write_text(json.dumps(mapping, indent=2), encoding="utf-8")
    protocol = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "test_sample_ids": [int(rec["sample_id"]) for rec in records],
        "training_geometric_centers_per_sequence": 1024,
        "training_material_tokens_per_sequence": 2048,
        "max_matched_geometric_centers_per_attention": 1024,
        "max_matched_material_tokens_per_attention": 2048,
        "grouping": "deterministic stratified region-balanced partition, target ratios 25/50/25, no replacement/padding",
        "region_ratios": list(REGION_RATIOS),
        "group_mapping_fixed_across_frames": True,
        "full_field_coverage": True,
        "cross_group_attention": False,
        "recursive_state": "[S4, PE4, PEEQ]",
        "rollout": "GT z0 then autoregressive z1..z180 with no reset",
        "surface_order": "SPOS_0,SNEG_0,SPOS_1,SNEG_1,...",
        "attention_backend": backend,
    }
    (args.output_dir / "matched_protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    tf = evaluate_mode(model, records, stats, device, groups_by_sample, "teacher_forcing", max_frames=1 if args.precheck else None)
    roll = evaluate_mode(model, records, stats, device, groups_by_sample, "rollout", max_frames=1 if args.precheck else None)
    tf_summary = write_summary(args.output_dir / "teacher_forcing_summary.json", "matched_token_teacher_forcing", tf, args.checkpoint, records)
    roll_summary = write_summary(args.output_dir / "rollout_summary.json", "matched_token_rollout", roll, args.checkpoint, records)
    write_csv(args.output_dir / "teacher_forcing_per_sample.csv", tf["sample_rows"])
    write_csv(args.output_dir / "rollout_per_sample.csv", roll["sample_rows"])
    write_csv(args.output_dir / "rollout_per_frame.csv", roll["frame_rows"])
    for sid, rows in roll["sample_frames"].items():
        write_csv(args.output_dir / f"sample_{sid:04d}_rollout_per_frame.csv", rows)
    if not args.precheck:
        three_way(args.output_dir, tf_summary, roll_summary)
        sequence_diagnostic = per_sample_diagnostic(args.output_dir)
        frame_comparison(args.output_dir)
    else:
        sequence_diagnostic = {}
    memory = {"peak_allocated_mb": float(torch.cuda.max_memory_allocated(device) / 1024**2), "peak_reserved_mb": float(torch.cuda.max_memory_reserved(device) / 1024**2)}
    final = {"status": "PASS" if tf["finite"] and roll["finite"] and tf["coverage_ok"] and roll["coverage_ok"] and tf["duplicate_ok"] and roll["duplicate_ok"] else "FAIL", "precheck": bool(args.precheck), "checkpoint_epoch": checkpoint.get("epoch"), "attention_backend": backend, "finite": bool(tf["finite"] and roll["finite"]), "full_coverage": bool(tf["coverage_ok"] and roll["coverage_ok"]), "all_centers_updated_exactly_once": bool(roll["duplicate_ok"]), "gpu_memory": memory, "sequence_diagnostic": sequence_diagnostic, "seconds": time.time() - start, "checkpoint_audit": audit}
    (args.output_dir / ("matched_precheck.json" if args.precheck else "evaluation_status.json")).write_text(json.dumps(final, indent=2), encoding="utf-8")
    print(json.dumps(final, indent=2))
    if final["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
