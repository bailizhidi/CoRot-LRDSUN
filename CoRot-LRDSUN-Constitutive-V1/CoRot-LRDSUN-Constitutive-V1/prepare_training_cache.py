#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

from corot_lrdsun.contracts import PREPARED_CACHE_FORMAT
from corot_lrdsun.geometry import compute_kabsch_rotations_numpy, region_ids_from_reference
from corot_lrdsun.io import scan_raw_npz, save_json


def save_npy(path: Path, arr) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(arr))


def state9(z, surface: str) -> np.ndarray:
    return np.concatenate([
        z[f"S_corot_{surface}"],
        z[f"PE_corot_{surface}"],
        z[f"PEEQ_{surface}"],
    ], axis=2).astype(np.float32)


def prepare_one(rec: dict, out_root: Path, overwrite: bool, kabsch_chunk: int) -> dict:
    raw = Path(rec["raw_path"])
    out = out_root / f"sample_{int(rec['sample_id']):04d}"
    done = out / "PREPARED.ok"
    if done.exists() and not overwrite:
        m = json.loads((out / "meta.json").read_text())
        return m
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with np.load(raw, allow_pickle=False) as z:
        X0 = np.asarray(z["tube_mesh_pos"], dtype=np.float32)
        U = np.asarray(z["U_tube"], dtype=np.float32)
        Q0 = np.asarray(z["Q0"], dtype=np.float32)
        ptr = np.asarray(z["frame_neighbor_ptr"], dtype=np.int64)
        idx = np.asarray(z["frame_neighbor_index"], dtype=np.int32)
        angle = np.asarray(z["angle_progress"], dtype=np.float32)
        outer = state9(z, "outer")
        inner = state9(z, "inner")
        LEo = np.asarray(z["LE_corot_outer"], dtype=np.float32)
        LEi = np.asarray(z["LE_corot_inner"], dtype=np.float32)
        T, N = U.shape[:2]
        R = np.empty((T, N, 3, 3), dtype=np.float32)
        rank2 = np.empty((T, N), dtype=np.float32)
        for t in range(T):
            Rt, sv = compute_kabsch_rotations_numpy(X0, X0 + U[t], ptr, idx, chunk_size=kabsch_chunk)
            R[t] = Rt
            rank2[t] = sv[:, 1] / np.maximum(sv[:, 0], 1e-12)
            if t % 30 == 0 or t == T-1:
                print(f"sample={rec['sample_id']:03d} Kabsch {t+1}/{T}", flush=True)
        region_id = region_ids_from_reference(X0, rec["D_outer"], rec["R_bending"])
        if float(np.min(outer[..., 8])) < -1e-9 or float(np.min(inner[..., 8])) < -1e-9:
            raise RuntimeError(f"Negative PEEQ in {raw}")
        arrays = {
            "X0": X0, "U": U, "Q0": Q0, "ptr": ptr, "idx": idx, "R": R,
            "region_id": region_id, "angle_progress": angle,
            "state_outer": outer, "state_inner": inner,
            "LE_outer": LEo, "LE_inner": LEi,
        }
        for name, arr in arrays.items(): save_npy(out / f"{name}.npy", arr)
        meta = dict(rec)
        meta.update({
            "cache_path": str(out.resolve()),
            "rank2_ratio_min": float(np.min(rank2)),
            "rank2_ratio_p01": float(np.percentile(rank2, 1)),
            "region_counts": {str(k): int(np.sum(region_id == k)) for k in (0,1,2)},
            "prepared_seconds": time.time()-t0,
        })
        save_json(out / "meta.json", meta)
        done.write_text("PASSED\n")
        return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", type=Path, required=True)
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--kabsch-chunk", type=int, default=8192)
    ap.add_argument("--sample-ids", type=str, default="")
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    recs = scan_raw_npz(args.dataset_dir)
    wanted = None
    if args.sample_ids.strip(): wanted = {int(x) for x in args.sample_ids.split(",") if x.strip()}
    selected = [r for r in recs if wanted is None or int(r["sample_id"]) in wanted]
    out_recs = []
    if args.workers <= 1:
        for i, rec in enumerate(selected, 1):
            print(f"[{i}/{len(selected)}] preparing {Path(rec['raw_path']).name}", flush=True)
            out_recs.append(prepare_one(rec, args.cache_dir, args.overwrite, args.kabsch_chunk))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(prepare_one, rec, args.cache_dir, args.overwrite, args.kabsch_chunk): rec for rec in selected}
            done_n = 0
            for fut in as_completed(futs):
                rec = futs[fut]; out_recs.append(fut.result()); done_n += 1
                print(f"prepared {done_n}/{len(selected)} sample={rec['sample_id']}", flush=True)
    # If preparing a subset, merge with already prepared metadata.
    all_meta = []
    for p in sorted(args.cache_dir.glob("sample_*/meta.json")):
        try: all_meta.append(json.loads(p.read_text()))
        except Exception: pass
    all_meta.sort(key=lambda x: int(x["sample_id"]))
    split_counts = {s: sum(str(r["split"]).lower()==s for r in all_meta) for s in ("train","val","test")}
    manifest = {"format": PREPARED_CACHE_FORMAT, "dataset_dir": str(args.dataset_dir.resolve()), "records": all_meta, "split_counts": split_counts}
    save_json(args.cache_dir / "manifest.json", manifest)
    (args.cache_dir / "CACHE_PREPARED.ok").write_text("PASSED\n")
    print(f"Prepared {len(all_meta)} samples; splits={split_counts}; manifest={args.cache_dir/'manifest.json'}")

if __name__ == "__main__": main()
