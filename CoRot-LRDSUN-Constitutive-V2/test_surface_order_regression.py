#!/usr/bin/env python3
"""Synthetic regression for paired SPOS/SNEG canonical ordering."""

from __future__ import annotations

import json
import argparse
from pathlib import Path

import numpy as np

from corot_lrdsun_v2 import surface_center_to_canonical_states


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "surface_order_regression_test.json")
    args = ap.parse_args()
    n = 7
    # Distinct, easily identifiable rows: surface 0 is 1000+i and surface 1
    # is 2000+i.  A correct canonical flatten must interleave them by center.
    values = np.zeros((2, n, 2), dtype=np.float32)
    for i in range(n):
        values[0, i] = (1000 + i, 1000 + i + 0.25)
        values[1, i] = (2000 + i, 2000 + i + 0.25)
    canonical = surface_center_to_canonical_states(values)
    expected = np.stack([values[0], values[1]], axis=1).reshape(-1, 2)
    center_id = np.repeat(np.arange(n, dtype=np.int64), 2)
    surface_id = np.tile(np.asarray([0, 1], dtype=np.int64), n)
    material_state_index = 2 * center_id + surface_id
    expected_index = np.arange(2 * n, dtype=np.int64)
    result = {
        "status": "PASS" if np.array_equal(canonical, expected) and np.array_equal(center_id, np.repeat(np.arange(n), 2)) and np.array_equal(surface_id, np.tile([0, 1], n)) and np.array_equal(material_state_index, expected_index) else "FAIL",
        "input_shape": list(values.shape),
        "canonical_shape": list(canonical.shape),
        "canonical_values": canonical.tolist(),
        "expected_values": expected.tolist(),
        "center_id": center_id.tolist(),
        "surface_id": surface_id.tolist(),
        "material_state_index": material_state_index.tolist(),
        "mapping": "SPOS_0,SNEG_0,SPOS_1,SNEG_1,...",
    }
    out = args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
