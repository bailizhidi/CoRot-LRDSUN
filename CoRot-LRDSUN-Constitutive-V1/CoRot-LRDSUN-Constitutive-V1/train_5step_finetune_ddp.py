#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
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

from corot_lrdsun.data import PreparedSample, balanced_centers, to_torch
from corot_lrdsun.io import load_manifest, split_records, save_json
from corot_lrdsun.losses import compute_losses
from corot_lrdsun.model import CoRotLRDSUN
from corot_lrdsun.normalization import (
    load_stats,
    normalize_inputs,
    delta8_from_norm,
    le_to_norm,
)
from corot_lrdsun.physics import mises_from_s4_torch
from corot_lrdsun.runtime import (
    setup_ddp,
    barrier,
    raw_model,
    set_seed,
    save_checkpoint,
)

HORIZON = 5
DEFAULT_STEP_WEIGHTS = (1.0, 0.75, 0.50, 0.35, 0.25)


def parse_step_weights(text: str) -> list[float]:
    vals = [float(x.strip()) for x in text.split(",") if x.strip()]
    if len(vals) != HORIZON:
        raise ValueError(
            f"--step-weights must contain exactly {HORIZON} comma-separated values, got {vals}"
        )
    if any(v < 0 for v in vals):
        raise ValueError(f"step weights must be non-negative, got {vals}")
    if vals[0] <= 0:
        raise ValueError("step-1 weight must be > 0")
    return vals


def reduce_pair(total: float, n: int, device: torch.device):
    x = torch.tensor([total, n], device=device, dtype=torch.float64)
    if dist.is_initialized():
        dist.all_reduce(x)
    return float(x[0]), float(x[1])


def partition_records(records, world: int, rank: int, epoch: int, seed: int):
    order = list(records)
    random.Random(seed + epoch).shuffle(order)
    if len(order) % world:
        need = world - (len(order) % world)
        order += order[:need]
    return order[rank::world]


def paired_center_surfaces(c_geom: np.ndarray, rng: np.random.Generator):
    """Each geometric center contributes BOTH shell surfaces.

    1024 geometric centers -> 2048 material-state samples.
    surface 0 = outer/SPOS, surface 1 = inner/SNEG.
    """
    c_geom = np.asarray(c_geom, dtype=np.int64).reshape(-1)
    centers = np.repeat(c_geom, 2)
    surfaces = np.tile(np.asarray([0, 1], dtype=np.int64), len(c_geom))
    perm = rng.permutation(len(centers))
    return centers[perm], surfaces[perm]


def cyclic_window_indices(
    sample: PreparedSample,
    epoch: int,
    windows_per_sample: int,
    seed: int,
):
    """Cyclic coverage for valid 5-step windows t -> ... -> t+5.

    For T=181 there are 176 valid starts: t=0,...,175.
    With 15 windows/sample/epoch, ceil(176/15)=12 epochs/cycle.
    Every valid start appears exactly once in a cycle.
    """
    n_windows = int(sample.T - HORIZON)
    if n_windows <= 0:
        raise RuntimeError(
            f"Need T>={HORIZON + 1} for {HORIZON}-step training, got T={sample.T}"
        )

    k = min(max(int(windows_per_sample), 1), n_windows)
    cycle_epochs = int(math.ceil(n_windows / k))

    epoch0 = int(epoch) - 1
    cycle_id = epoch0 // cycle_epochs
    slot_id = epoch0 % cycle_epochs

    sample_id = int(sample.meta.get("sample_id", 0))
    cycle_seed = int(seed) + sample_id * 1_000_003 + cycle_id * 10_000_019
    rng = np.random.default_rng(cycle_seed)
    perm = rng.permutation(n_windows)

    a = slot_id * k
    b = min(a + k, n_windows)
    starts = perm[a:b].astype(np.int64, copy=False)
    return starts, cycle_id, slot_id, cycle_epochs


def random_window_indices(sample: PreparedSample, n: int, rng: np.random.Generator):
    n_windows = int(sample.T - HORIZON)
    return rng.choice(
        np.arange(n_windows),
        size=min(int(n), n_windows),
        replace=False,
    ).astype(np.int64, copy=False)


def iter_window_batches(
    sample: PreparedSample,
    rng: np.random.Generator,
    starts: np.ndarray,
    centers_per_window: int,
    batch_size: int,
):
    for t in np.asarray(starts, dtype=np.int64).tolist():
        c_geom = balanced_centers(
            sample.region_id,
            centers_per_window,
            rng,
        )
        centers, surfaces = paired_center_surfaces(c_geom, rng)
        for a in range(0, len(centers), batch_size):
            b = min(a + batch_size, len(centers))
            yield int(t), centers[a:b], surfaces[a:b]


def predicted_next_state(out: dict, input_state: torch.Tensor, stats: dict):
    """Differentiable state update using the SOFT plastic gate."""
    pred_d8 = delta8_from_norm(out["delta8_norm"].float(), stats)
    pred_p_scaled = torch.sigmoid(out["plastic_logit"].float()) * F.softplus(
        out["peeq_mag_raw"].float()
    )
    pred_dpeeq = pred_p_scaled * float(stats["peeq_delta_scale"])
    return torch.cat(
        [
            input_state[:, :8].float() + pred_d8,
            input_state[:, 8:9].float() + pred_dpeeq,
        ],
        dim=1,
    )


def rollout_step_loss(
    out: dict,
    batch: dict,
    stats: dict,
    cfg: dict,
):
    """State-consistent loss for rollout steps whose input state is predicted.

    The S/PE and PEEQ terms compare the predicted NEXT STATE directly with the
    GT next state. This keeps gradients connected through all preceding rollout
    steps and implements truncated BPTT across the 5-step window.
    """
    input_state = batch["state"]
    pred_next = predicted_next_state(out, input_state, stats)
    gt_next = batch["next_state"]

    d8_std = torch.as_tensor(
        stats["delta8_std"],
        device=pred_next.device,
        dtype=torch.float32,
    )
    e8n = (pred_next[:, :8] - gt_next[:, :8]) / d8_std
    l_d8 = torch.mean(e8n.square())

    pscale = float(stats["peeq_delta_scale"])
    ep = (pred_next[:, 8:9] - gt_next[:, 8:9]) / pscale
    l_p = F.smooth_l1_loss(ep, torch.zeros_like(ep), beta=0.25)

    active = (batch["delta_peeq"] > float(cfg["plastic_threshold"])).float()
    pos_weight = torch.tensor(
        [float(stats.get("plastic_pos_weight", 1.0))],
        device=active.device,
        dtype=active.dtype,
    )
    l_gate = F.binary_cross_entropy_with_logits(
        out["plastic_logit"],
        active,
        pos_weight=pos_weight,
    )

    gt_len = le_to_norm(batch["le_next"], stats)
    l_le = F.mse_loss(out["le_norm"], gt_len)

    pred_vm = mises_from_s4_torch(pred_next[:, :4])
    gt_vm = mises_from_s4_torch(gt_next[:, :4])
    vm_scale = float(stats["mises_scale"])
    l_vm = F.smooth_l1_loss(
        pred_vm / vm_scale,
        gt_vm / vm_scale,
        beta=0.25,
    )

    total = (
        float(cfg["w_delta8"]) * l_d8
        + float(cfg["w_peeq"]) * l_p
        + float(cfg["w_gate"]) * l_gate
        + float(cfg["w_le"]) * l_le
        + float(cfg["w_mises"]) * l_vm
    )

    parts = {
        "loss": float(total.detach()),
        "delta8": float(l_d8.detach()),
        "peeq": float(l_p.detach()),
        "gate": float(l_gate.detach()),
        "le": float(l_le.detach()),
        "mises": float(l_vm.detach()),
    }
    return total, parts, pred_next


def five_step_forward_loss(
    model,
    sample: PreparedSample,
    t: int,
    centers: np.ndarray,
    surfaces: np.ndarray,
    stats: dict,
    device: torch.device,
    cfg: dict,
    amp_enabled: bool,
):
    """Differentiable 5-step rollout.

    z_t^GT -> zhat_{t+1} -> zhat_{t+2} -> ... -> zhat_{t+5}

    No predicted recursive state is detached. Losses at later steps therefore
    backpropagate through all earlier state updates inside the 5-step window.
    """
    weights = cfg["step_weights"]
    parts_all: list[dict] = []
    pred_states: list[torch.Tensor] = []

    # Step 1: preserve the original V1 one-step objective exactly.
    b_np = sample.make_batch(t, centers, surfaces)
    tb = normalize_inputs(to_torch(b_np, device), stats)

    with torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16,
        enabled=amp_enabled,
    ):
        out = model(tb["state_n"], tb["context_n"], tb["edge_n"], tb["mask"])
        loss1, parts1, aux1 = compute_losses(out, tb, stats, cfg)

    pred_state = torch.cat(
        [
            tb["state"][:, :8].float() + aux1["pred_d8"].float(),
            tb["state"][:, 8:9].float()
            + aux1["pred_p_scaled"].float() * float(stats["peeq_delta_scale"]),
        ],
        dim=1,
    )
    parts_all.append(parts1)
    pred_states.append(pred_state)
    total = float(weights[0]) * loss1

    # Steps 2..5: each step consumes the previous model prediction.
    for step in range(2, HORIZON + 1):
        b_np = sample.make_batch(t + step - 1, centers, surfaces)
        tb = to_torch(b_np, device)
        tb["state"] = pred_state  # IMPORTANT: no detach
        tb = normalize_inputs(tb, stats)

        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=amp_enabled,
        ):
            out = model(tb["state_n"], tb["context_n"], tb["edge_n"], tb["mask"])
            loss_step, parts_step, pred_state = rollout_step_loss(
                out,
                tb,
                stats,
                cfg,
            )

        total = total + float(weights[step - 1]) * loss_step
        parts_all.append(parts_step)
        pred_states.append(pred_state)

    return total, parts_all, pred_states


@torch.no_grad()
def validate_5step(model, records, stats, device, args):
    model.eval()
    rng = np.random.default_rng(args.seed + 99_173)
    cfg = vars(args)
    amp_enabled = (not args.no_amp) and device.type == "cuda"

    totals = {"total": 0.0}
    for step in range(1, HORIZON + 1):
        totals[f"step{step}"] = 0.0
        for k in ("delta8", "peeq", "gate", "le", "mises"):
            totals[f"step{step}_{k}"] = 0.0
    windows = 0

    use = records[: args.max_val_samples] if args.max_val_samples > 0 else records

    for rec in use:
        s = PreparedSample(rec)
        starts = random_window_indices(s, args.val_windows_per_sample, rng)
        for t, centers, surfaces in iter_window_batches(
            s,
            rng,
            starts,
            args.val_centers_per_window,
            args.eval_batch_size,
        ):
            loss, parts_all, _ = five_step_forward_loss(
                model,
                s,
                t,
                centers,
                surfaces,
                stats,
                device,
                cfg,
                amp_enabled,
            )
            totals["total"] += float(loss.detach())
            for step, parts in enumerate(parts_all, start=1):
                totals[f"step{step}"] += parts["loss"]
                for k in ("delta8", "peeq", "gate", "le", "mises"):
                    totals[f"step{step}_{k}"] += parts[k]
            windows += 1
        del s
        gc.collect()

    return {k: v / max(windows, 1) for k, v in totals.items()} | {
        "windows": windows
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--stats", type=Path, default=None)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--warm-start", type=Path, default=None)
    ap.add_argument("--resume", type=Path, default=None)

    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--eval-batch-size", type=int, default=2048)

    ap.add_argument("--windows-per-sample", type=int, default=15)
    ap.add_argument("--centers-per-window", type=int, default=1024)
    ap.add_argument("--val-windows-per-sample", type=int, default=18)
    ap.add_argument("--val-centers-per-window", type=int, default=1024)
    ap.add_argument("--max-val-samples", type=int, default=0)

    ap.add_argument("--hidden-dim", type=int, default=192)
    ap.add_argument("--dropout", type=float, default=0.0)

    # Longer BPTT has a larger aggregate gradient than the 2-step objective;
    # use a more conservative fine-tuning LR while warm-starting from V2.
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--min-lr", type=float, default=1e-6)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--grad-clip", type=float, default=1.0)

    ap.add_argument(
        "--step-weights",
        type=str,
        default=",".join(str(x) for x in DEFAULT_STEP_WEIGHTS),
        help="Comma-separated weights for L1..L5",
    )

    ap.add_argument("--w-delta8", type=float, default=1.0)
    ap.add_argument("--w-peeq", type=float, default=1.0)
    ap.add_argument("--w-gate", type=float, default=0.20)
    ap.add_argument("--w-le", type=float, default=0.25)
    ap.add_argument("--w-mises", type=float, default=0.25)

    ap.add_argument("--plastic-threshold", type=float, default=1e-10)
    ap.add_argument("--early-stop-start", type=int, default=18)
    ap.add_argument("--early-stop-patience", type=int, default=8)

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    args.step_weights = parse_step_weights(args.step_weights)

    if args.warm_start is not None and args.resume is not None:
        raise RuntimeError("Use either --warm-start or --resume, not both")
    if args.warm_start is None and args.resume is None:
        raise RuntimeError("5-step fine-tuning requires --warm-start or --resume")

    rank, local, world, device = setup_ddp()
    set_seed(args.seed, rank)

    try:
        manifest = load_manifest(args.cache_dir)
        train = split_records(manifest, "train")
        val = split_records(manifest, "val")
        if not train or not val:
            raise RuntimeError("Need train and val records in prepared cache")

        stats = load_stats(args.stats or args.cache_dir / "stats.json")
        args.plastic_threshold = float(
            stats.get("plastic_threshold", args.plastic_threshold)
        )

        if args.smoke:
            args.epochs = min(args.epochs, 2)
            args.windows_per_sample = 2
            # Test the formal C2048 spatial/batch contract in smoke.
            args.centers_per_window = min(args.centers_per_window, 2048)
            args.val_windows_per_sample = 2
            args.val_centers_per_window = 512
            args.max_val_samples = min(args.max_val_samples or 2, 2)

        init_path = args.resume if args.resume is not None else args.warm_start
        init_ck = torch.load(init_path, map_location="cpu")
        ck_cfg = init_ck.get("config", {})
        ck_hidden = int(ck_cfg.get("hidden_dim", args.hidden_dim))
        if ck_hidden != args.hidden_dim:
            if rank == 0:
                print(
                    f"hidden_dim overridden from {args.hidden_dim} to warm-start value {ck_hidden}",
                    flush=True,
                )
            args.hidden_dim = ck_hidden

        model = CoRotLRDSUN(args.hidden_dim, args.dropout).to(device)
        model.load_state_dict(init_ck["model"])

        if world > 1:
            model = DDP(
                model,
                device_ids=[local],
                output_device=local,
                find_unused_parameters=False,
            )

        opt = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt,
            T_max=max(args.epochs, 1),
            eta_min=args.min_lr,
        )

        start = 1
        best = float("inf")
        history = []
        bad_epochs = 0

        if args.resume is not None:
            ck = torch.load(args.resume, map_location=device)
            raw_model(model).load_state_dict(ck["model"])
            if "optimizer" in ck:
                opt.load_state_dict(ck["optimizer"])
            if "scheduler" in ck:
                sched.load_state_dict(ck["scheduler"])
            start = int(ck["epoch"]) + 1
            best = float(ck.get("best_val", best))
            history = list(ck.get("history", []))

        if rank == 0:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            cfg_out = {
                k: (str(v) if isinstance(v, Path) else v)
                for k, v in vars(args).items()
            }
            cfg_out.update(
                {
                    "objective_contract": "5step_differentiable_rollout_v1",
                    "state_path": "z_t_GT -> zhat_t+1 -> ... -> zhat_t+5",
                    "horizon": HORIZON,
                    "surface_contract": "paired_outer_inner",
                    "valid_5step_windows_per_T181": 176,
                }
            )
            save_json(args.output_dir / "config.json", cfg_out)

            nwin = int(train[0]["num_frames"]) - HORIZON
            cyc = int(math.ceil(nwin / args.windows_per_sample))
            print(
                f"train={len(train)} val={len(val)} world={world} device={device} start_epoch={start}",
                flush=True,
            )
            print(
                "objective=5step_differentiable "
                f"warm_start={args.warm_start} "
                f"step_weights={args.step_weights}",
                flush=True,
            )
            print(
                f"5step_window_contract: {nwin} valid starts, "
                f"{args.windows_per_sample} per sample per epoch, "
                f"{cyc} epochs/full temporal cycle",
                flush=True,
            )
            print(
                "surface_contract=paired_outer_inner "
                f"geometric_centers_per_window={args.centers_per_window} "
                f"material_states_per_window={2 * args.centers_per_window} "
                f"batch_size={args.batch_size}",
                flush=True,
            )

        amp_enabled = (not args.no_amp) and device.type == "cuda"

        for epoch in range(start, args.epochs + 1):
            model.train()
            local_records = partition_records(train, world, rank, epoch, args.seed)

            sum_loss = 0.0
            sum_steps = [0.0 for _ in range(HORIZON)]
            nsteps = 0
            t0 = time.time()

            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)

            if rank == 0:
                nwin = int(train[0]["num_frames"]) - HORIZON
                cyc_epochs = int(math.ceil(nwin / args.windows_per_sample))
                e0 = epoch - 1
                print(
                    f"epoch={epoch:03d} window_schedule=cyclic "
                    f"cycle={e0 // cyc_epochs + 1} "
                    f"slot={e0 % cyc_epochs + 1}/{cyc_epochs} "
                    f"windows_per_sample={args.windows_per_sample}/{nwin}",
                    flush=True,
                )

            for ir, rec in enumerate(local_records, 1):
                s = PreparedSample(rec)
                sample_id = int(rec.get("sample_id", 0))
                rng = np.random.default_rng(
                    args.seed + epoch * 100_003 + sample_id * 1_009
                )
                starts, _, _, _ = cyclic_window_indices(
                    s,
                    epoch,
                    args.windows_per_sample,
                    args.seed,
                )

                for t, centers, surfaces in iter_window_batches(
                    s,
                    rng,
                    starts,
                    args.centers_per_window,
                    args.batch_size,
                ):
                    opt.zero_grad(set_to_none=True)

                    loss, parts_all, _ = five_step_forward_loss(
                        model,
                        s,
                        t,
                        centers,
                        surfaces,
                        stats,
                        device,
                        vars(args),
                        amp_enabled,
                    )

                    loss.backward()
                    if args.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(),
                            args.grad_clip,
                        )
                    opt.step()

                    sum_loss += float(loss.detach())
                    for j, parts in enumerate(parts_all):
                        sum_steps[j] += float(parts["loss"])
                    nsteps += 1

                del s
                gc.collect()

                if rank == 0 and (ir % 5 == 0 or ir == len(local_records)):
                    step_txt = " ".join(
                        f"L{j+1}={sum_steps[j] / max(nsteps,1):.6e}"
                        for j in range(HORIZON)
                    )
                    print(
                        f"epoch={epoch:03d} sample={ir}/{len(local_records)} "
                        f"loss={sum_loss / max(nsteps,1):.6e} {step_txt}",
                        flush=True,
                    )

            gsum, gsteps = reduce_pair(sum_loss, nsteps, device)
            train_loss = gsum / max(gsteps, 1)
            train_step_losses = []
            for x in sum_steps:
                gx, _ = reduce_pair(x, nsteps, device)
                train_step_losses.append(gx / max(gsteps, 1))

            sched.step()
            barrier()

            stop = False
            if rank == 0:
                v = validate_5step(raw_model(model), val, stats, device, args)
                val_loss = float(v["total"])

                peak_alloc_gb = (
                    torch.cuda.max_memory_allocated(device) / 1024**3
                    if device.type == "cuda"
                    else float("nan")
                )
                peak_reserved_gb = (
                    torch.cuda.max_memory_reserved(device) / 1024**3
                    if device.type == "cuda"
                    else float("nan")
                )

                rec_hist = {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    **{
                        f"train_step{j+1}": train_step_losses[j]
                        for j in range(HORIZON)
                    },
                    "val": v,
                    "lr": float(opt.param_groups[0]["lr"]),
                    "seconds": time.time() - t0,
                    "peak_alloc_gb_rank0": peak_alloc_gb,
                    "peak_reserved_gb_rank0": peak_reserved_gb,
                }
                history.append(rec_hist)
                print(json.dumps(rec_hist, ensure_ascii=False), flush=True)

                improved = val_loss < best - 1e-8
                if improved:
                    best = val_loss
                    bad_epochs = 0
                    save_checkpoint(
                        args.output_dir / "best_val_5step.pt",
                        epoch,
                        best,
                        model,
                        opt,
                        sched,
                        history,
                        vars(args),
                        stats,
                    )
                else:
                    bad_epochs += 1

                save_checkpoint(
                    args.output_dir / "last.pt",
                    epoch,
                    best,
                    model,
                    opt,
                    sched,
                    history,
                    vars(args),
                    stats,
                )
                save_json(args.output_dir / "history.json", history)

                if (
                    epoch >= args.early_stop_start
                    and bad_epochs >= args.early_stop_patience
                ):
                    print(
                        f"EARLY STOP epoch={epoch} best_val={best:.6e} bad_epochs={bad_epochs}",
                        flush=True,
                    )
                    stop = True

            flag = torch.tensor(
                [1 if stop else 0],
                device=device,
                dtype=torch.int32,
            )
            if dist.is_initialized():
                dist.broadcast(flag, 0)
            if int(flag.item()):
                break
            barrier()

    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
