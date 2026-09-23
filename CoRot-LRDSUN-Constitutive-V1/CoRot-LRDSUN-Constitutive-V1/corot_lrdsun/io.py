from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from .contracts import PREPARED_CACHE_FORMAT


def _decode_scalar(v) -> str:
    a = np.asarray(v).reshape(-1)
    if len(a) == 0:
        return ""
    x = a[0]
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="ignore").rstrip("\x00")
    return str(x).rstrip("\x00")


def scan_raw_npz(dataset_dir: Path) -> list[dict]:
    records = []
    for p in sorted(Path(dataset_dir).glob("*.npz")):
        with np.load(p, allow_pickle=False) as z:
            rec = {
                "raw_path": str(p.resolve()),
                "name": p.stem,
                "sample_id": int(np.asarray(z["sample_id"]).reshape(-1)[0]),
                "geometry_id": int(np.asarray(z["geometry_id"]).reshape(-1)[0]),
                "split": _decode_scalar(z["split"]).lower(),
                "num_frames": int(np.asarray(z["num_frames"]).reshape(-1)[0]),
                "num_nodes": int(np.asarray(z["num_tube_nodes"]).reshape(-1)[0]),
                "D_outer": float(np.asarray(z["D_outer"]).reshape(-1)[0]),
                "Thickness": float(np.asarray(z["Thickness"]).reshape(-1)[0]),
                "R_bending": float(np.asarray(z["R_bending"]).reshape(-1)[0]),
                "t_over_D": float(np.asarray(z["t_over_D"]).reshape(-1)[0]),
                "R_over_D": float(np.asarray(z["R_over_D"]).reshape(-1)[0]),
                "E_modulus": float(np.asarray(z["E_modulus"]).reshape(-1)[0]),
                "Poisson_Ratio": float(np.asarray(z["Poisson_Ratio"]).reshape(-1)[0]),
            }
        records.append(rec)
    return records


def save_json(path: Path, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_manifest(cache_dir: Path) -> dict:
    m = load_json(Path(cache_dir) / "manifest.json")
    if m.get("format") != PREPARED_CACHE_FORMAT:
        raise RuntimeError(f"Bad prepared cache format: {m.get('format')!r}")
    return m


def split_records(manifest: dict, split: str) -> list[dict]:
    return [r for r in manifest["records"] if str(r["split"]).lower() == split.lower()]
