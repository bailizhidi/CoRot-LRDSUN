from __future__ import annotations
import torch
import numpy as np


def mises_from_s4_torch(s: torch.Tensor) -> torch.Tensor:
    saa, stt, srr, sat = s[..., 0], s[..., 1], s[..., 2], s[..., 3]
    vm2 = 0.5 * ((saa-stt)**2 + (stt-srr)**2 + (srr-saa)**2) + 3.0 * sat**2
    return torch.sqrt(torch.clamp(vm2, min=0.0)).unsqueeze(-1)


def mises_from_s4_numpy(s: np.ndarray) -> np.ndarray:
    s = np.asarray(s)
    saa, stt, srr, sat = [s[..., i] for i in range(4)]
    vm2 = 0.5 * ((saa-stt)**2 + (stt-srr)**2 + (srr-saa)**2) + 3.0 * sat**2
    return np.sqrt(np.maximum(vm2, 0.0))[..., None]
