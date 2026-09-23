#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from corot_lrdsun.data import PreparedSample
from corot_lrdsun.io import load_manifest, split_records
from corot_lrdsun.model import CoRotLRDSUN
from corot_lrdsun.normalization import load_stats
from corot_lrdsun.physics import mises_from_s4_numpy
from corot_lrdsun.runtime import predict_batch


def parse_ks(s: str):
    ks = sorted({int(x.strip()) for x in s.split(",") if x.strip()})
    if not ks:
        raise RuntimeError("No K values supplied")
    if any(k <= 0 for k in ks):
        raise RuntimeError("All K values must be > 0")
    return ks


def rmse_from_sse(sse: float, n: int):
    if n <= 0:
        return float("nan")
    return float(np.sqrt(sse / n))


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)

    ap.add_argument("--split", choices=("val", "test"), default="test")

    ap.add_argument(
        "--ks",
        type=str,
        default="1,2,5,10,20,30,60,180",
    )

    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--device", choices=("cuda", "cpu"), default="cuda")

    ap.add_argument("--soft-gate", action="store_true")
    ap.add_argument("--gate-threshold", type=float, default=0.5)

    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    amp = device.type == "cuda"

    ks = parse_ks(args.ks)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    ck = torch.load(args.checkpoint, map_location=device)

    stats = ck.get("stats") or load_stats(
        args.cache_dir / "stats.json"
    )

    cfg = ck.get("config", {})

    model = CoRotLRDSUN(
        int(cfg.get("hidden_dim", 192)),
        float(cfg.get("dropout", 0.0)),
    ).to(device)

    model.load_state_dict(ck["model"])
    model.eval()

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    manifest = load_manifest(args.cache_dir)
    records = split_records(manifest, args.split)

    print("=" * 90)
    print("K-step reset diagnostic")
    print("device     :", device)
    print("AMP/BF16   :", amp)
    print("split      :", args.split)
    print("samples    :", len(records))
    print("K values   :", ks)
    print("batch size :", args.batch_size)
    print("=" * 90)

    sample_rows = []
    global_rows = []
    frame_rows = []

    with torch.inference_mode():

        for K in ks:

            print()
            print("=" * 90)
            print(f"K = {K}")
            print("=" * 90)

            global_vm_sse = 0.0
            global_vm_n = 0

            global_s4_sse = 0.0
            global_s4_n = 0

            global_peeq_sse = 0.0
            global_peeq_n = 0

            final_vm_sse = 0.0
            final_vm_n = 0

            frame_vm_sse = None
            frame_vm_n = None

            for ir, rec in enumerate(records, 1):

                sid = int(rec["sample_id"])
                s = PreparedSample(rec)

                if frame_vm_sse is None:
                    frame_vm_sse = np.zeros(s.T, dtype=np.float64)
                    frame_vm_n = np.zeros(s.T, dtype=np.int64)

                # ------------------------------------------------------
                # initial state = GT at frame 0
                # [2 surfaces, N, 9]
                # ------------------------------------------------------
                pred = np.stack(
                    [
                        np.asarray(s.state_outer[0], dtype=np.float32),
                        np.asarray(s.state_inner[0], dtype=np.float32),
                    ],
                    axis=0,
                ).copy()

                sample_vm_sse = 0.0
                sample_vm_n = 0

                sample_s4_sse = 0.0
                sample_s4_n = 0

                sample_peeq_sse = 0.0
                sample_peeq_n = 0

                for t in range(s.T - 1):

                    # ==================================================
                    # RESET RULE
                    #
                    # K=1:
                    #   t=0,1,2,3,... always GT -> teacher forcing
                    #
                    # K=180:
                    #   only t=0 reset -> full rollout
                    # ==================================================
                    if t % K == 0:
                        pred[0] = np.asarray(
                            s.state_outer[t],
                            dtype=np.float32,
                        )
                        pred[1] = np.asarray(
                            s.state_inner[t],
                            dtype=np.float32,
                        )

                    next_pred = np.empty_like(pred)

                    for surface in (0, 1):

                        for a in range(0, s.N, args.batch_size):

                            b = min(
                                a + args.batch_size,
                                s.N,
                            )

                            centers = np.arange(
                                a,
                                b,
                                dtype=np.int64,
                            )

                            surfaces = np.full(
                                len(centers),
                                surface,
                                dtype=np.int64,
                            )

                            state_override = pred[
                                surface,
                                centers,
                            ]

                            batch = s.make_batch(
                                t,
                                centers,
                                surfaces,
                                state_override=state_override,
                            )

                            pstate, _, _, _, _ = predict_batch(
                                model,
                                batch,
                                stats,
                                device,
                                amp=amp,
                                hard_gate=not args.soft_gate,
                                gate_threshold=args.gate_threshold,
                            )

                            next_pred[
                                surface,
                                centers,
                            ] = pstate.float().cpu().numpy()

                    pred = next_pred

                    frame = t + 1

                    # --------------------------------------------------
                    # Ground truth at frame t+1
                    # --------------------------------------------------
                    gt = np.stack(
                        [
                            np.asarray(
                                s.state_outer[frame],
                                dtype=np.float32,
                            ),
                            np.asarray(
                                s.state_inner[frame],
                                dtype=np.float32,
                            ),
                        ],
                        axis=0,
                    )

                    # --------------------------------------------------
                    # S4 error
                    # --------------------------------------------------
                    ds4 = (
                        pred[:, :, :4].astype(np.float64)
                        - gt[:, :, :4].astype(np.float64)
                    )

                    s4_sse = float(np.sum(ds4 ** 2))
                    s4_n = int(ds4.size)

                    sample_s4_sse += s4_sse
                    sample_s4_n += s4_n

                    global_s4_sse += s4_sse
                    global_s4_n += s4_n

                    # --------------------------------------------------
                    # PEEQ error
                    # --------------------------------------------------
                    dpeq = (
                        pred[:, :, 8].astype(np.float64)
                        - gt[:, :, 8].astype(np.float64)
                    )

                    peeq_sse = float(np.sum(dpeq ** 2))
                    peeq_n = int(dpeq.size)

                    sample_peeq_sse += peeq_sse
                    sample_peeq_n += peeq_n

                    global_peeq_sse += peeq_sse
                    global_peeq_n += peeq_n

                    # --------------------------------------------------
                    # von Mises
                    # --------------------------------------------------
                    frame_sse = 0.0
                    frame_n = 0

                    for surface in (0, 1):

                        gt_vm = mises_from_s4_numpy(
                            gt[surface, :, :4]
                        ).reshape(-1)

                        pr_vm = mises_from_s4_numpy(
                            pred[surface, :, :4]
                        ).reshape(-1)

                        dvm = (
                            pr_vm.astype(np.float64)
                            - gt_vm.astype(np.float64)
                        )

                        ss = float(np.sum(dvm ** 2))
                        nn = int(dvm.size)

                        frame_sse += ss
                        frame_n += nn

                    frame_vm_sse[frame] += frame_sse
                    frame_vm_n[frame] += frame_n

                    sample_vm_sse += frame_sse
                    sample_vm_n += frame_n

                    global_vm_sse += frame_sse
                    global_vm_n += frame_n

                    if frame == s.T - 1:
                        final_vm_sse += frame_sse
                        final_vm_n += frame_n

                sample_row = {
                    "K": K,
                    "sample_id": sid,
                    "mises_rmse_mpa": rmse_from_sse(
                        sample_vm_sse,
                        sample_vm_n,
                    ),
                    "s4_rmse_mpa": rmse_from_sse(
                        sample_s4_sse,
                        sample_s4_n,
                    ),
                    "peeq_rmse": rmse_from_sse(
                        sample_peeq_sse,
                        sample_peeq_n,
                    ),
                }

                sample_rows.append(sample_row)

                print(
                    f"K={K:3d} "
                    f"[{ir:02d}/{len(records):02d}] "
                    f"sample={sid:04d} "
                    f"MisesRMSE="
                    f"{sample_row['mises_rmse_mpa']:.6f} MPa",
                    flush=True,
                )

            # ----------------------------------------------------------
            # Global result for K
            # ----------------------------------------------------------
            k_sample_values = [
                r["mises_rmse_mpa"]
                for r in sample_rows
                if r["K"] == K
            ]

            global_row = {
                "K": K,
                "global_mises_rmse_mpa": rmse_from_sse(
                    global_vm_sse,
                    global_vm_n,
                ),
                "mean_sample_mises_rmse_mpa": float(
                    np.mean(k_sample_values)
                ),
                "global_s4_rmse_mpa": rmse_from_sse(
                    global_s4_sse,
                    global_s4_n,
                ),
                "global_peeq_rmse": rmse_from_sse(
                    global_peeq_sse,
                    global_peeq_n,
                ),
                "final_frame_mises_rmse_mpa": rmse_from_sse(
                    final_vm_sse,
                    final_vm_n,
                ),
            }

            global_rows.append(global_row)

            # ----------------------------------------------------------
            # frame-wise curve for this K
            # ----------------------------------------------------------
            for frame in range(1, len(frame_vm_sse)):
                frame_rows.append(
                    {
                        "K": K,
                        "frame": frame,
                        "mises_rmse_mpa": rmse_from_sse(
                            frame_vm_sse[frame],
                            int(frame_vm_n[frame]),
                        ),
                    }
                )

            print("-" * 90)
            print(json.dumps(global_row, indent=2))
            print("-" * 90)

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    sample_csv = args.output_dir / "kstep_sample_metrics.csv"
    global_csv = args.output_dir / "kstep_global_metrics.csv"
    frame_csv = args.output_dir / "kstep_frame_metrics.csv"
    json_path = args.output_dir / "kstep_global_metrics.json"

    with sample_csv.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "K",
                "sample_id",
                "mises_rmse_mpa",
                "s4_rmse_mpa",
                "peeq_rmse",
            ],
        )
        w.writeheader()
        w.writerows(sample_rows)

    with global_csv.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=list(global_rows[0].keys()),
        )
        w.writeheader()
        w.writerows(global_rows)

    with frame_csv.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "K",
                "frame",
                "mises_rmse_mpa",
            ],
        )
        w.writeheader()
        w.writerows(frame_rows)

    json_path.write_text(
        json.dumps(global_rows, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 90)
    print("K-STEP RESET DIAGNOSTIC COMPLETE")
    print("global :", global_csv)
    print("sample :", sample_csv)
    print("frame  :", frame_csv)
    print("=" * 90)


if __name__ == "__main__":
    main()
