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
    """Cyclic coverage for valid 2-step windows t -> t+1 -> t+2.

    For T=181 there are 179 valid starts: t=0,...,178.
    With 15 windows/sample/epoch, ceil(179/15)=12 epochs/cycle.
    Every valid start appears exactly once in a cycle.
    """
    n_windows = int(sample.T - 2)
    if n_windows <= 0:
        raise RuntimeError(f"Need T>=3 for 2-step training, got T={sample.T}")

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
    n_windows = int(sample.T - 2)
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
    """Differentiable material-state update using the SOFT plastic gate."""
    pred_d8 = delta8_from_norm(out["delta8_norm"].float(), stats)
    pred_p_scaled = torch.sigmoid(out["plastic_logit"].float()) * F.softplus(
        out["peeq_mag_raw"].float()
    )
    pred_dpeeq = pred_p_scaled * float(stats["peeq_delta_scale"])
    next_state = torch.cat(
        [
            input_state[:, :8].float() + pred_d8,
            input_state[:, 8:9].float() + pred_dpeeq,
        ],
        dim=1,
    )
    return next_state


def rollout_step_loss(
    out: dict,
    batch: dict,
    stats: dict,
    cfg: dict,
):
    """State-consistent loss for a rollout step whose input state may be predicted.

    Important difference from ordinary one-step teacher forcing:
    the dS/dPE and PEEQ terms penalize the PREDICTED NEXT STATE against the GT next
    state. Therefore L2 can backpropagate through the predicted state from step 1.
    """
    input_state = batch["state"]
    pred_next = predicted_next_state(out, input_state, stats)
    gt_next = batch["next_state"]

    # State-consistent S4+PE4 loss. The mean of delta normalization cancels in
    # an error difference, so division by the original delta8 std is sufficient.
    d8_std = torch.as_tensor(
        stats["delta8_std"],
        device=pred_next.device,
        dtype=torch.float32,
    )
    e8n = (pred_next[:, :8] - gt_next[:, :8]) / d8_std
    l_d8 = torch.mean(e8n.square())

    # State-consistent PEEQ loss. This also sends gradient back through step 1.
    pscale = float(stats["peeq_delta_scale"])
    ep = (pred_next[:, 8:9] - gt_next[:, 8:9]) / pscale
    l_p = F.smooth_l1_loss(ep, torch.zeros_like(ep), beta=0.25)

    # Plastic activity remains a physical GT transition label.
    active = (
        batch["delta_peeq"] > float(cfg["plastic_threshold"])
    ).float()
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


def two_step_forward_loss(
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
    """Differentiable 2-step rollout:

      z_t^GT -> zhat_{t+1} -> zhat_{t+2}

    Step 2 consumes zhat_{t+1} WITHOUT detach.
    """
    # -------------------------------
    # Step 1: ordinary teacher forcing
    # -------------------------------
    b1_np = sample.make_batch(t, centers, surfaces)
    tb1 = normalize_inputs(to_torch(b1_np, device), stats)

    with torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16,
        enabled=amp_enabled,
    ):
        out1 = model(tb1["state_n"], tb1["context_n"], tb1["edge_n"], tb1["mask"])
        loss1, parts1, aux1 = compute_losses(out1, tb1, stats, cfg)

    # IMPORTANT: no detach here.
    pred_state1 = torch.cat(
        [
            tb1["state"][:, :8].float() + aux1["pred_d8"].float(),
            tb1["state"][:, 8:9].float()
            + aux1["pred_p_scaled"].float() * float(stats["peeq_delta_scale"]),
        ],
        dim=1,
    )

    # -----------------------------------------------
    # Step 2: input is model prediction from step 1
    # -----------------------------------------------
    b2_np = sample.make_batch(t + 1, centers, surfaces)
    tb2 = to_torch(b2_np, device)

    # Replace only the recursive material state. Geometry/context and GT targets
    # are still those of physical transition (t+1)->(t+2).
    tb2["state"] = pred_state1
    tb2 = normalize_inputs(tb2, stats)

    with torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16,
        enabled=amp_enabled,
    ):
        out2 = model(tb2["state_n"], tb2["context_n"], tb2["edge_n"], tb2["mask"])
        loss2, parts2, pred_state2 = rollout_step_loss(out2, tb2, stats, cfg)

    total = loss1 + float(cfg["step2_weight"]) * loss2
    return total, parts1, parts2, pred_state1, pred_state2


@torch.no_grad()
def validate_2step(model, records, stats, device, args):
    model.eval()
    rng = np.random.default_rng(args.seed + 99_173)
    cfg = vars(args)
    amp_enabled = (not args.no_amp) and device.type == "cuda"

    totals = {
        "total": 0.0,
        "step1": 0.0,
        "step2": 0.0,
        "step1_delta8": 0.0,
        "step1_peeq": 0.0,
        "step1_gate": 0.0,
        "step1_le": 0.0,
        "step1_mises": 0.0,
        "step2_delta8": 0.0,
        "step2_peeq": 0.0,
        "step2_gate": 0.0,
        "step2_le": 0.0,
        "step2_mises": 0.0,
    }
    steps = 0

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
            loss, p1, p2, _, _ = two_step_forward_loss(
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
            totals["step1"] += p1["loss"]
            totals["step2"] += p2["loss"]
            for k in ("delta8", "peeq", "gate", "le", "mises"):
                totals[f"step1_{k}"] += p1[k]
                totals[f"step2_{k}"] += p2[k]
            steps += 1
        del s
        gc.collect()

    return {k: v / max(steps, 1) for k, v in totals.items()} | {"steps": steps}


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

    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--min-lr", type=float, default=2e-6)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--grad-clip", type=float, default=1.0)

    ap.add_argument("--step2-weight", type=float, default=0.5)

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

    if args.warm_start is not None and args.resume is not None:
        raise RuntimeError("Use either --warm-start or --resume, not both")
    if args.warm_start is None and args.resume is None:
        raise RuntimeError("2-step fine-tuning requires --warm-start or --resume")

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
            # Keep formal center/batch contract in smoke: 1024 centers -> 2048 states.
            args.centers_per_window = 1024
            args.val_windows_per_sample = 2
            args.val_centers_per_window = 512
            args.max_val_samples = min(args.max_val_samples or 2, 2)

        # Warm-start checkpoint determines architecture.
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

        # Resume means resume a previous 2-step run including optimizer/scheduler.
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
                    "objective_contract": "2step_differentiable_rollout_v1",
                    "state_path": "z_t_GT -> zhat_t+1 -> zhat_t+2",
                    "step_weights": [1.0, float(args.step2_weight)],
                    "surface_contract": "paired_outer_inner",
                    "valid_2step_windows_per_T181": 179,
                }
            )
            save_json(args.output_dir / "config.json", cfg_out)

            nwin = int(train[0]["num_frames"]) - 2
            cyc = int(math.ceil(nwin / args.windows_per_sample))
            print(
                f"train={len(train)} val={len(val)} world={world} device={device} start_epoch={start}",
                flush=True,
            )
            print(
                "objective=2step_differentiable "
                f"warm_start={args.warm_start} "
                f"step_weights=[1.0,{args.step2_weight}]",
                flush=True,
            )
            print(
                f"2step_window_contract: {nwin} valid starts, "
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
            sum_l1 = 0.0
            sum_l2 = 0.0
            nsteps = 0
            t0 = time.time()

            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)

            # Audit-friendly cycle log using first sample's T.
            if rank == 0:
                nwin = int(train[0]["num_frames"]) - 2
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

                    loss, p1, p2, _, _ = two_step_forward_loss(
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
                    sum_l1 += float(p1["loss"])
                    sum_l2 += float(p2["loss"])
                    nsteps += 1

                del s
                gc.collect()

                if rank == 0 and (ir % 5 == 0 or ir == len(local_records)):
                    print(
                        f"epoch={epoch:03d} sample={ir}/{len(local_records)} "
                        f"loss={sum_loss / max(nsteps,1):.6e} "
                        f"L1={sum_l1 / max(nsteps,1):.6e} "
                        f"L2={sum_l2 / max(nsteps,1):.6e}",
                        flush=True,
                    )

            gsum, gsteps = reduce_pair(sum_loss, nsteps, device)
            gl1, _ = reduce_pair(sum_l1, nsteps, device)
            gl2, _ = reduce_pair(sum_l2, nsteps, device)
            train_loss = gsum / max(gsteps, 1)
            train_l1 = gl1 / max(gsteps, 1)
            train_l2 = gl2 / max(gsteps, 1)

            sched.step()
            barrier()

            stop = False
            if rank == 0:
                v = validate_2step(raw_model(model), val, stats, device, args)
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
                    "train_step1": train_l1,
                    "train_step2": train_l2,
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
                        args.output_dir / "best_val_2step.pt",
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
