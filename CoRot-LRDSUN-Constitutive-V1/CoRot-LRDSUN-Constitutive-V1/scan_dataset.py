#!/usr/bin/env python3
from __future__ import annotations
import argparse
from collections import Counter
from pathlib import Path
from corot_lrdsun.io import scan_raw_npz, save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--expect-samples", type=int, default=150)
    ap.add_argument("--expect-train", type=int, default=120)
    ap.add_argument("--expect-val", type=int, default=15)
    ap.add_argument("--expect-test", type=int, default=15)
    args = ap.parse_args()
    recs = scan_raw_npz(args.dataset_dir)
    counts = Counter(r["split"] for r in recs)
    ids=[int(r["sample_id"]) for r in recs]
    dup=sorted({x for x in ids if ids.count(x)>1})
    if dup:
        raise SystemExit(f"Duplicate sample_id values: {dup}")
    print(f"dataset={args.dataset_dir}")
    print(f"samples={len(recs)} split_counts={dict(counts)}")
    for r in recs[:5]:
        print(f"  sample={r['sample_id']:03d} split={r['split']} N={r['num_nodes']} T={r['num_frames']} file={Path(r['raw_path']).name}")
    if args.expect_samples > 0 and len(recs) != args.expect_samples:
        raise SystemExit(f"Expected {args.expect_samples} samples, got {len(recs)}")
    for split, expected in (("train",args.expect_train),("val",args.expect_val),("test",args.expect_test)):
        if expected >= 0 and counts.get(split,0) != expected:
            raise SystemExit(f"Expected {expected} {split} samples, got {counts.get(split,0)}")
    bad = [r for r in recs if r["num_frames"] != 181 or r["split"] not in {"train","val","test"}]
    if bad:
        raise SystemExit(f"Bad records: {bad[:3]}")
    out = args.output or (args.dataset_dir / "raw_manifest.json")
    save_json(out, {"records": recs, "split_counts": dict(counts)})
    print(f"Wrote {out}")

if __name__ == "__main__":
    main()
