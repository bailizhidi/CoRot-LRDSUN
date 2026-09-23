from __future__ import annotations

import json
from pathlib import Path
import torch


def load_stats(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _tensor(stats: dict, key: str, device, dtype=torch.float32):
    return torch.tensor(stats[key], device=device, dtype=dtype)


def normalize_inputs(batch: dict[str, torch.Tensor], stats: dict) -> dict[str, torch.Tensor]:
    dev = batch["state"].device
    edge_mean = _tensor(stats, "edge_mean", dev); edge_std = _tensor(stats, "edge_std", dev)
    state_mean = _tensor(stats, "state_mean", dev); state_std = _tensor(stats, "state_std", dev)
    ctx_mean = _tensor(stats, "context_mean", dev); ctx_std = _tensor(stats, "context_std", dev)
    out = dict(batch)
    out["edge_n"] = (batch["edge"] - edge_mean) / edge_std
    out["state_n"] = (batch["state"] - state_mean) / state_std
    out["context_n"] = (batch["context"] - ctx_mean) / ctx_std
    return out


def delta8_to_norm(x: torch.Tensor, stats: dict) -> torch.Tensor:
    m = _tensor(stats, "delta8_mean", x.device); s = _tensor(stats, "delta8_std", x.device)
    return (x - m) / s


def delta8_from_norm(x: torch.Tensor, stats: dict) -> torch.Tensor:
    m = _tensor(stats, "delta8_mean", x.device); s = _tensor(stats, "delta8_std", x.device)
    return x * s + m


def le_to_norm(x: torch.Tensor, stats: dict) -> torch.Tensor:
    m = _tensor(stats, "le_mean", x.device); s = _tensor(stats, "le_std", x.device)
    return (x - m) / s


def le_from_norm(x: torch.Tensor, stats: dict) -> torch.Tensor:
    m = _tensor(stats, "le_mean", x.device); s = _tensor(stats, "le_std", x.device)
    return x * s + m
