#!/usr/bin/env python3
"""V2 5-step training entry point.

The first V2 milestone is data/token validation, so this script is provided
for the subsequent training milestone but is not invoked by the smoke test.
It keeps one complete token sequence per model call: no spatial chunking is
performed because chunking would change the nonlocal attention contract.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from corot_lrdsun_v2 import CoRotTokenTransolver, TokenDataBuilder
from corot_lrdsun_v2.token_data import TokenBatch
from corot_lrdsun_v2.v1_compat import (
    PreparedSample,
    balanced_centers,
    compute_losses,
    delta8_from_norm,
    le_to_norm,
    load_manifest,
    load_stats,
    mises_from_s4_torch,
    setup_ddp,
    split_records,
)


HORIZON = 5
STEP_WEIGHTS = (1.0, 0.75, 0.50, 0.35, 0.25)
V1_PREFIXES = (
    "edge_encoder.", "center_encoder.", "query.", "key.", "value.",
    "trunk.", "delta8_head.", "plastic_logit_head.", "peeq_mag_head.", "le_head.",
)


def partition_records(records, world: int, rank: int, epoch: int, seed: int):
    order = list(records)
    random.Random(seed + epoch).shuffle(order)
    if len(order) % world:
        need = world - (len(order) % world)
        order += order[:need]
    return order[rank::world]


def cyclic_window_indices(sample: PreparedSample, epoch: int, n: int, seed: int):
    n_windows = int(sample.T - HORIZON)
    if n_windows <= 0:
        raise RuntimeError(f"T={sample.T} is too short for {HORIZON}-step windows")
    k = min(max(int(n), 1), n_windows)
    cycle_epochs = int(math.ceil(n_windows / k))
    epoch0 = int(epoch) - 1
    cycle_id = epoch0 // cycle_epochs
    slot_id = epoch0 % cycle_epochs
    sample_id = int(sample.meta.get("sample_id", 0))
    rng = np.random.default_rng(int(seed) + sample_id * 1_000_003 + cycle_id * 10_000_019)
    perm = rng.permutation(n_windows)
    return perm[slot_id * k : min((slot_id + 1) * k, n_windows)].astype(np.int64)


def random_window_indices(sample: PreparedSample, n: int, rng: np.random.Generator):
    """Match the V1 validation sampler: unique random valid starts."""
    n_windows = int(sample.T - HORIZON)
    if n_windows <= 0:
        raise RuntimeError(f"T={sample.T} is too short for {HORIZON}-step windows")
    return rng.choice(
        np.arange(n_windows),
        size=min(int(n), n_windows),
        replace=False,
    ).astype(np.int64, copy=False)


def predicted_next_state(out, input_state, stats):
    d8 = delta8_from_norm(out["delta8_norm"].float(), stats)
    p_scaled = torch.sigmoid(out["plastic_logit"].float()) * F.softplus(
        out["peeq_mag_raw"].float()
    )
    dpeeq = p_scaled * float(stats["peeq_delta_scale"])
    return torch.cat([input_state[:, :8].float() + d8, input_state[:, 8:9].float() + dpeeq], dim=1)


def rollout_step_loss(out, batch: TokenBatch, stats, cfg):
    pred_next = predicted_next_state(out, batch.state, stats)
    d8_std = torch.as_tensor(stats["delta8_std"], device=pred_next.device, dtype=torch.float32)
    l_d8 = ((pred_next[:, :8] - batch.next_state[:, :8]) / d8_std).square().mean()
    pscale = float(stats["peeq_delta_scale"])
    l_p = F.smooth_l1_loss(
        (pred_next[:, 8:9] - batch.next_state[:, 8:9]) / pscale,
        torch.zeros_like(pred_next[:, 8:9]),
        beta=0.25,
    )
    active = (batch.delta_peeq > float(cfg["plastic_threshold"])).float()
    pos_weight = torch.tensor(
        [float(stats.get("plastic_pos_weight", 1.0))],
        device=active.device,
        dtype=active.dtype,
    )
    l_gate = F.binary_cross_entropy_with_logits(
        out["plastic_logit"], active, pos_weight=pos_weight
    )
    l_le = F.mse_loss(out["le_norm"], le_to_norm(batch.le_next, stats))
    pred_vm = mises_from_s4_torch(pred_next[:, :4])
    gt_vm = mises_from_s4_torch(batch.next_state[:, :4])
    vm_scale = float(stats["mises_scale"])
    l_vm = F.smooth_l1_loss(pred_vm / vm_scale, gt_vm / vm_scale, beta=0.25)
    total = (
        float(cfg["w_delta8"]) * l_d8
        + float(cfg["w_peeq"]) * l_p
        + float(cfg["w_gate"]) * l_gate
        + float(cfg["w_le"]) * l_le
        + float(cfg["w_mises"]) * l_vm
    )
    detail = {
        "loss": float(total.detach()),
        "delta8": float(l_d8.detach()),
        "peeq": float(l_p.detach()),
        "gate": float(l_gate.detach()),
        "le": float(l_le.detach()),
        "mises": float(l_vm.detach()),
    }
    return total, pred_next, detail


def five_step_forward_loss(model, builder, sample, t, centers, stats, device, cfg, amp):
    pred_state = None
    total = None
    parts = []
    for step in range(HORIZON):
        batch = builder.build(sample, int(t) + step, centers, shuffle=False).to(device)
        if pred_state is not None:
            batch = batch.with_state_override(pred_state)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
            out = model(batch)
            if step == 0:
                loss, detail, _ = compute_losses(out, batch.as_loss_batch(), stats, cfg)
            else:
                loss, pred_state, detail = rollout_step_loss(out, batch, stats, cfg)
        if step == 0:
            pred_state = predicted_next_state(out, batch.state, stats)
        total = loss * STEP_WEIGHTS[step] if total is None else total + loss * STEP_WEIGHTS[step]
        parts.append(detail)
    return total, parts


def raw_model(model):
    return model.module if isinstance(model, DDP) else model


def grad_l2(parameters):
    values = [p.grad.detach().float().square().sum() for p in parameters if p.grad is not None]
    if not values:
        return 0.0
    return float(torch.sqrt(torch.stack(values).sum()).detach().cpu())


def named_grad_l2(model, prefix):
    values = [
        p.grad.detach().float().square().sum()
        for name, p in model.named_parameters()
        if name.startswith(prefix) and p.grad is not None
    ]
    if not values:
        return 0.0
    return float(torch.sqrt(torch.stack(values).sum()).detach().cpu())


def parameter_snapshot(model, prefixes=("physics_attention.", "token_adapter.")):
    return {
        name: p.detach().float().cpu().clone()
        for name, p in model.named_parameters()
        if name.startswith(prefixes)
    }


def parameter_change(before, model):
    rows = []
    for name, old in before.items():
        new = dict(model.named_parameters())[name].detach().float().cpu()
        diff = new - old
        rows.append((name, diff))
    if not rows:
        return {"parameter_count": 0, "l2": 0.0, "max_abs": 0.0, "checksum_before": 0.0, "checksum_after": 0.0}
    return {
        "parameter_count": len(rows),
        "l2": float(torch.sqrt(torch.stack([d.square().sum() for _, d in rows]).sum())),
        "max_abs": float(max(d.abs().max().item() for _, d in rows)),
        "checksum_before": float(sum(old.sum().item() for old in before.values())),
        "checksum_after": float(sum(dict(model.named_parameters())[name].detach().float().cpu().sum().item() for name, _ in rows)),
    }


@torch.no_grad()
def validate_5step(model, records, stats, device, cfg, seed, windows_per_sample, max_samples, amp):
    if not records:
        raise RuntimeError("validation split is empty")
    selected = records if max_samples <= 0 else records[:max_samples]
    rng = np.random.default_rng(int(seed))
    step_sums = [0.0] * HORIZON
    component_sums = [{key: 0.0 for key in ("delta8", "peeq", "gate", "le", "mises")} for _ in range(HORIZON)]
    n_windows = 0
    for rec in selected:
        sample = PreparedSample(rec)
        starts = random_window_indices(sample, windows_per_sample, rng)
        builder = TokenDataBuilder(stats)
        for t in starts:
            centers = balanced_centers(sample.region_id, cfg["centers_per_window"], rng)
            _, parts = five_step_forward_loss(
                model, builder, sample, int(t), centers, stats, device, cfg, amp
            )
            for step in range(HORIZON):
                step_sums[step] += float(parts[step]["loss"])
                for key in component_sums[step]:
                    component_sums[step][key] += float(parts[step][key])
            n_windows += 1
    step_means = [x / max(n_windows, 1) for x in step_sums]
    return {
        "total": float(sum(w * x for w, x in zip(STEP_WEIGHTS, step_means))),
        "step_losses": step_means,
        "step_components": [
            {key: value / max(n_windows, 1) for key, value in row.items()}
            for row in component_sums
        ],
        "samples": len(selected),
        "windows": n_windows,
    }


def gather_memory(device, world):
    values = torch.tensor([
        torch.cuda.memory_allocated(device) / 1024**2,
        torch.cuda.memory_reserved(device) / 1024**2,
        torch.cuda.max_memory_allocated(device) / 1024**2,
        torch.cuda.max_memory_reserved(device) / 1024**2,
    ], dtype=torch.float64, device=device)
    gathered = [torch.zeros_like(values) for _ in range(world)]
    dist.all_gather(gathered, values)
    return {
        str(rank): {
            "allocated_mb": round(row[0], 3),
            "reserved_mb": round(row[1], 3),
            "peak_allocated_mb": round(row[2], 3),
            "peak_reserved_mb": round(row[3], 3),
        }
        for rank, row in enumerate(torch.stack(gathered).cpu().tolist())
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--stats", type=Path, default=None)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--warm-start", type=Path, default=None)
    ap.add_argument("--resume", type=Path, default=None)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--windows-per-sample", type=int, default=15)
    ap.add_argument("--centers-per-window", type=int, default=1024)
    ap.add_argument("--hidden-dim", type=int, default=192)
    ap.add_argument("--token-dim", type=int, default=192)
    ap.add_argument("--attention-depth", type=int, default=2)
    ap.add_argument("--attention-heads", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--min-lr", type=float, default=1e-6)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--w-delta8", type=float, default=1.0)
    ap.add_argument("--w-peeq", type=float, default=1.0)
    ap.add_argument("--w-gate", type=float, default=0.20)
    ap.add_argument("--w-le", type=float, default=0.25)
    ap.add_argument("--w-mises", type=float, default=0.25)
    ap.add_argument("--plastic-threshold", type=float, default=1e-10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--pilot", action="store_true", help="two-epoch instrumented pilot; enforces C1024")
    ap.add_argument("--val-windows-per-sample", type=int, default=18)
    ap.add_argument("--max-val-samples", type=int, default=0, help="0 means all validation samples")
    ap.add_argument("--early-stop-start", type=int, default=18)
    ap.add_argument("--early-stop-patience", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true", help="forward/shape check only; no optimizer step")
    args = ap.parse_args()

    rank, local, world, device = setup_ddp()
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    try:
        records = split_records(load_manifest(args.cache_dir), "train")
        stats = load_stats(args.stats or args.cache_dir / "stats.json")
        args.plastic_threshold = float(stats.get("plastic_threshold", args.plastic_threshold))
        if args.smoke:
            args.epochs = min(args.epochs, 2)
            args.windows_per_sample = min(args.windows_per_sample, 1)
            args.centers_per_window = min(args.centers_per_window, 1024)
        if args.pilot:
            args.epochs = 2
            args.windows_per_sample = 1
            if args.centers_per_window != 1024:
                raise ValueError("pilot training requires centers_per_window=1024")
        if args.warm_start is None and args.resume is None and not args.dry_run:
            raise RuntimeError("--warm-start or --resume is required for V2 training")

        manifest = load_manifest(args.cache_dir)
        val_records = split_records(manifest, "val")
        if not val_records:
            raise RuntimeError("Need a non-empty val split for pilot training")
        model = CoRotTokenTransolver(
            hidden_dim=args.hidden_dim,
            token_dim=args.token_dim,
            attention_depth=args.attention_depth,
            attention_heads=args.attention_heads,
        ).to(device)
        warm_report = None
        if args.warm_start is not None:
            warm_report = model.load_v1_checkpoint(args.warm_start, map_location="cpu")
            component_status = {
                prefix[:-1]: {
                    "loaded": sum(k.startswith(prefix) for k in warm_report["loaded"]),
                    "missing": sum(k.startswith(prefix) for k in warm_report["missing_after_warm_start"]),
                }
                for prefix in V1_PREFIXES
            }
            if any(item["loaded"] == 0 or item["missing"] > 0 for item in component_status.values()):
                raise RuntimeError(f"V1 warm-start baseline keys are incomplete: {component_status}")
            if rank == 0:
                print(json.dumps({"warm_start": warm_report, "v1_component_status": component_status}, indent=2), flush=True)
        if world > 1:
            model = DDP(model, device_ids=[local], output_device=local, find_unused_parameters=False)

        resume_ckpt = None
        start_epoch = 1
        resumed_history = []
        resumed_best_val = float("inf")
        if args.resume is not None:
            resume_ckpt = torch.load(args.resume, map_location=device)
            raw_model(model).load_state_dict(resume_ckpt["model"])
            start_epoch = int(resume_ckpt.get("epoch", 0)) + 1
            resumed_history = list(resume_ckpt.get("history", []))
            resumed_best_val = float(resume_ckpt.get("best_val", float("inf")))

        if args.dry_run:
            rec = records[0]
            sample = PreparedSample(rec)
            rng = np.random.default_rng(args.seed)
            centers = balanced_centers(sample.region_id, args.centers_per_window, rng)
            builder = TokenDataBuilder(stats)
            batch = builder.build(sample, 0, centers, shuffle=False).to(device)
            with torch.no_grad():
                out = model(batch)
                token_shape = list(model.module.token_sequence(batch).shape if isinstance(model, DDP) else model.token_sequence(batch).shape)
            if rank == 0:
                print(json.dumps({
                    "dry_run": True,
                    "sample_id": int(rec["sample_id"]),
                    "time": 0,
                    "geometric_centers": batch.num_geometric_centers,
                    "material_tokens": batch.num_tokens,
                    "token_shape": token_shape,
                    "output_shapes": {k: list(v.shape) for k, v in out.items()},
                }, indent=2), flush=True)
            return

        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(args.epochs, 1), eta_min=args.min_lr)
        if resume_ckpt is not None:
            if "optimizer" in resume_ckpt:
                opt.load_state_dict(resume_ckpt["optimizer"])
            if "scheduler" in resume_ckpt:
                sched.load_state_dict(resume_ckpt["scheduler"])
        cfg = vars(args)
        amp = (not args.no_amp) and device.type == "cuda"
        args.output_dir.mkdir(parents=True, exist_ok=True)
        # Keep the two newly added branches separate in the pilot report.  A
        # combined checksum could show an update while hiding a frozen
        # Physics-Attention block behind the token adapter update.
        initial_attention = parameter_snapshot(raw_model(model), ("physics_attention.",))
        initial_adapter = parameter_snapshot(raw_model(model), ("token_adapter.",))
        history = resumed_history
        best_val = resumed_best_val
        best_name = "best_val_pilot.pt" if args.pilot else "best_val.pt"
        bad_epochs = 0
        for epoch in range(start_epoch, args.epochs + 1):
            model.train()
            local_records = partition_records(records, world, rank, epoch, args.seed)
            epoch_loss = 0.0
            step_sums = [0.0] * HORIZON
            component_sums = [
                {key: 0.0 for key in ("delta8", "peeq", "gate", "le", "mises")}
                for _ in range(HORIZON)
            ]
            grad_pre_sum = 0.0
            grad_post_sum = 0.0
            attention_grad_sum = 0.0
            adapter_grad_sum = 0.0
            trunk_grad_sum = 0.0
            attention_grad_nonzero = 0
            nsteps = 0
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize(device)
            epoch_start = time.time()
            for rec in local_records:
                sample = PreparedSample(rec)
                rng = np.random.default_rng(args.seed + epoch * 100003 + int(rec["sample_id"]) * 1009)
                starts = cyclic_window_indices(sample, epoch, args.windows_per_sample, args.seed)
                builder = TokenDataBuilder(stats)
                for t in starts:
                    centers = balanced_centers(sample.region_id, args.centers_per_window, rng)
                    opt.zero_grad(set_to_none=True)
                    loss, parts = five_step_forward_loss(model, builder, sample, int(t), centers, stats, device, cfg, amp)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"non-finite training loss at epoch={epoch}, sample={rec['sample_id']}, t={t}")
                    loss.backward()
                    grad_pre = grad_l2(model.parameters())
                    attention_grad = named_grad_l2(raw_model(model), "physics_attention.")
                    adapter_grad = named_grad_l2(raw_model(model), "token_adapter.")
                    trunk_grad = named_grad_l2(raw_model(model), "trunk.")
                    grad_pre_sum += grad_pre
                    attention_grad_sum += attention_grad
                    adapter_grad_sum += adapter_grad
                    trunk_grad_sum += trunk_grad
                    attention_grad_nonzero += int(attention_grad > 0.0)
                    if args.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    grad_post = grad_l2(model.parameters())
                    grad_post_sum += grad_post
                    opt.step()
                    epoch_loss += float(loss.detach())
                    for step in range(HORIZON):
                        step_sums[step] += float(parts[step]["loss"])
                        for key in component_sums[step]:
                            component_sums[step][key] += float(parts[step][key])
                    nsteps += 1
            sched.step()
            if dist.is_initialized():
                dist.barrier()
            # Aggregate training metrics over all DDP ranks.
            train_vec = torch.tensor(
                [epoch_loss, *step_sums, grad_pre_sum, grad_post_sum,
                 attention_grad_sum, adapter_grad_sum, trunk_grad_sum,
                 float(attention_grad_nonzero), float(nsteps)],
                dtype=torch.float64,
                device=device,
            )
            dist.all_reduce(train_vec, op=dist.ReduceOp.SUM)
            component_vec = torch.tensor(
                [component_sums[step][key] for step in range(HORIZON) for key in ("delta8", "peeq", "gate", "le", "mises")],
                dtype=torch.float64,
                device=device,
            )
            dist.all_reduce(component_vec, op=dist.ReduceOp.SUM)
            global_steps = max(float(train_vec[-1].item()), 1.0)
            if rank == 0:
                raw = raw_model(model)
                val_model = raw
                val = validate_5step(
                    val_model,
                    val_records,
                    stats,
                    device,
                    cfg,
                    args.seed + epoch * 100003,
                    args.val_windows_per_sample,
                    args.max_val_samples,
                    amp,
                )
            else:
                val = None
            if dist.is_initialized():
                dist.barrier()
            memory = gather_memory(device, world)
            if rank == 0:
                epoch_time = time.time() - epoch_start
                train_steps = [float(train_vec[1 + i].item()) / global_steps for i in range(HORIZON)]
                train_total = float(train_vec[0].item()) / global_steps
                train_components = []
                comp_keys = ("delta8", "peeq", "gate", "le", "mises")
                for step in range(HORIZON):
                    offset = step * len(comp_keys)
                    train_components.append({
                        key: float(component_vec[offset + j].item()) / global_steps
                        for j, key in enumerate(comp_keys)
                    })
                metrics = {
                    "epoch": epoch,
                    "epoch_time_sec": epoch_time,
                    "train_total": train_total,
                    "train_step_losses": train_steps,
                    "train_step_components": train_components,
                    "validation_total": val["total"],
                    "validation_step_losses": val["step_losses"],
                    "validation_step_components": val["step_components"],
                    "validation_samples": val["samples"],
                    "validation_windows": val["windows"],
                    "learning_rate": float(opt.param_groups[0]["lr"]),
                    "gradient_norm_pre_clip": float(train_vec[1 + HORIZON].item()) / global_steps,
                    "gradient_norm_post_clip": float(train_vec[2 + HORIZON].item()) / global_steps,
                    "physics_attention_grad_l2": float(train_vec[3 + HORIZON].item()) / global_steps,
                    "token_adapter_grad_l2": float(train_vec[4 + HORIZON].item()) / global_steps,
                    "v1_trunk_grad_l2": float(train_vec[5 + HORIZON].item()) / global_steps,
                    "physics_attention_nonzero_grad_steps": int(train_vec[6 + HORIZON].item()),
                    "optimizer_steps": int(train_vec[-1].item()),
                    "peak_memory_per_rank": memory,
                    "nan_inf": False,
                }
                history.append(metrics)
                improved = val["total"] < best_val - 1.0e-8
                if improved:
                    best_val = val["total"]
                    bad_epochs = 0
                else:
                    bad_epochs += 1
                ckpt = {
                    "epoch": epoch,
                    "best_val": best_val,
                    "model": model.module.state_dict() if isinstance(model, DDP) else model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "scheduler": sched.state_dict(),
                    "config": {
                        **{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
                        "objective_contract": "5step_differentiable_rollout_v1",
                        "recursive_state_contract": "z=[S4,PE4,PEEQ]",
                        "recursive_detach": False,
                        "best_checkpoint_criterion": "validation_total_min",
                        "surface_contract": "paired_outer_inner",
                    },
                    "stats": stats,
                    "history": history,
                }
                torch.save(ckpt, args.output_dir / "last.pt")
                if improved:
                    torch.save(ckpt, args.output_dir / best_name)
                print(json.dumps(metrics, ensure_ascii=False), flush=True)
            if dist.is_initialized():
                dist.barrier()

        if rank == 0:
            attention_change = parameter_change(initial_attention, raw_model(model))
            adapter_change = parameter_change(initial_adapter, raw_model(model))
            best_record = min(history, key=lambda row: row["validation_total"]) if history else None
            summary_name = "pilot_summary.json" if args.pilot else "formal_summary.json"
            summary = {
                "status": "PASS" if history else "FAIL",
                "pilot": bool(args.pilot),
                "epochs_completed": len(history),
                "world_size": world,
                "centers_per_rank": args.centers_per_window,
                "tokens_per_rank": 2 * args.centers_per_window,
                "step_weights": list(STEP_WEIGHTS),
                "warm_start": warm_report,
                "new_v2_keys": [k for k in raw_model(model).state_dict() if not any(k.startswith(p) for p in V1_PREFIXES)],
                "physics_attention_parameter_change": attention_change,
                "token_adapter_parameter_change": adapter_change,
                "physics_attention_grad_nonzero": any(x["physics_attention_nonzero_grad_steps"] > 0 for x in history),
                "nan_inf": any(x["nan_inf"] for x in history),
                "checkpoint_paths": {
                    "last": str(args.output_dir / "last.pt"),
                    "best": str(args.output_dir / best_name),
                },
                "best_epoch": int(best_record["epoch"]) if best_record else None,
                "best_validation_metric": float(best_record["validation_total"]) if best_record else None,
                "final_epoch_metric": float(history[-1]["validation_total"]) if history else None,
                "gpu_utilization": "not sampled",
                "ready_for_formal_training": bool(
                    len(history) == args.epochs
                    and all(not x["nan_inf"] for x in history)
                    and all(x["physics_attention_nonzero_grad_steps"] > 0 for x in history)
                    and attention_change["l2"] > 0.0
                ),
                "history": history,
            }
            (args.output_dir / summary_name).write_text(
                json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            print(json.dumps({"pilot_summary": summary}, ensure_ascii=False), flush=True)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
