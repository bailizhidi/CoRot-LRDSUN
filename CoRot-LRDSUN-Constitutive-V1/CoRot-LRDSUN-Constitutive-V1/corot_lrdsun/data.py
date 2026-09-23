from __future__ import annotations

from pathlib import Path
import numpy as np
import torch

from .geometry import corot_edge_features_for_centers


class PreparedSample:
    def __init__(self, record: dict):
        self.record = record
        self.dir = Path(record["cache_path"])
        self.meta = record
        mm = "r"
        self.X0 = np.load(self.dir / "X0.npy", mmap_mode=mm)
        self.U = np.load(self.dir / "U.npy", mmap_mode=mm)
        self.Q0 = np.load(self.dir / "Q0.npy", mmap_mode=mm)
        self.ptr = np.load(self.dir / "ptr.npy", mmap_mode=mm)
        self.idx = np.load(self.dir / "idx.npy", mmap_mode=mm)
        self.R = np.load(self.dir / "R.npy", mmap_mode=mm)
        self.region_id = np.load(self.dir / "region_id.npy", mmap_mode=mm)
        self.angle_progress = np.load(self.dir / "angle_progress.npy", mmap_mode=mm)
        self.state_outer = np.load(self.dir / "state_outer.npy", mmap_mode=mm)
        self.state_inner = np.load(self.dir / "state_inner.npy", mmap_mode=mm)
        self.LE_outer = np.load(self.dir / "LE_outer.npy", mmap_mode=mm)
        self.LE_inner = np.load(self.dir / "LE_inner.npy", mmap_mode=mm)
        self.N = int(self.X0.shape[0])
        self.T = int(self.U.shape[0])

    @property
    def D(self) -> float: return float(self.meta["D_outer"])
    @property
    def t_over_D(self) -> float: return float(self.meta["t_over_D"])
    @property
    def R_over_D(self) -> float: return float(self.meta["R_over_D"])
    @property
    def E_modulus(self) -> float: return float(self.meta["E_modulus"])
    @property
    def nu(self) -> float: return float(self.meta["Poisson_Ratio"])

    def states_at(self, t: int, centers: np.ndarray, surfaces: np.ndarray) -> np.ndarray:
        centers = np.asarray(centers, dtype=np.int64)
        surfaces = np.asarray(surfaces, dtype=np.int64)
        out = np.empty((len(centers), 9), dtype=np.float32)
        mo = surfaces == 0; mi = ~mo
        if np.any(mo): out[mo] = self.state_outer[t, centers[mo]]
        if np.any(mi): out[mi] = self.state_inner[t, centers[mi]]
        return out

    def le_at(self, t: int, centers: np.ndarray, surfaces: np.ndarray) -> np.ndarray:
        centers = np.asarray(centers, dtype=np.int64)
        surfaces = np.asarray(surfaces, dtype=np.int64)
        out = np.empty((len(centers), 4), dtype=np.float32)
        mo = surfaces == 0; mi = ~mo
        if np.any(mo): out[mo] = self.LE_outer[t, centers[mo]]
        if np.any(mi): out[mi] = self.LE_inner[t, centers[mi]]
        return out

    def make_batch(
        self,
        t: int,
        centers: np.ndarray,
        surfaces: np.ndarray,
        state_override: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        centers = np.asarray(centers, dtype=np.int64).reshape(-1)
        surfaces = np.asarray(surfaces, dtype=np.int64).reshape(-1)
        # outer/inner share exactly the same geometric kinematics.
        # Compute CoRot edge features once for each unique geometric center,
        # then map them back to the material-state samples.
        unique_centers, inverse = np.unique(
            centers,
            return_inverse=True,
        )

        edge_unique, mask_unique = corot_edge_features_for_centers(
            self.X0,
            self.U,
            self.Q0,
            self.R,
            self.ptr,
            self.idx,
            int(t),
            unique_centers,
            self.D,
        )

        edge = edge_unique[inverse]
        mask = mask_unique[inverse]
        gt_state = self.states_at(t, centers, surfaces)
        state = gt_state if state_override is None else np.asarray(state_override, dtype=np.float32)
        next_state = self.states_at(t + 1, centers, surfaces)
        delta = next_state - gt_state
        le_next = self.le_at(t + 1, centers, surfaces)
        zsurf = np.where(surfaces == 0, +0.5 * self.t_over_D, -0.5 * self.t_over_D).astype(np.float32)
        context = np.stack([
            np.full(len(centers), self.R_over_D, np.float32),
            np.full(len(centers), self.t_over_D, np.float32),
            zsurf,
            np.full(len(centers), float(self.angle_progress[t]), np.float32),
            np.full(len(centers), self.E_modulus, np.float32),
            np.full(len(centers), self.nu, np.float32),
        ], axis=1)
        return {
            "edge": edge,
            "mask": mask,
            "state": state.astype(np.float32),
            "gt_state_t": gt_state.astype(np.float32),
            "next_state": next_state.astype(np.float32),
            "delta8": delta[:, :8].astype(np.float32),
            "delta_peeq": delta[:, 8:9].astype(np.float32),
            "le_next": le_next.astype(np.float32),
            "context": context.astype(np.float32),
            "centers": centers,
            "surfaces": surfaces,
        }


def balanced_centers(region_id: np.ndarray, n: int, rng: np.random.Generator, ratios=(0.25, 0.50, 0.25)) -> np.ndarray:
    n = int(n)
    parts = [int(round(n * ratios[0])), int(round(n * ratios[1]))]
    parts.append(n - sum(parts))
    out = []
    for k, nk in enumerate(parts):
        pool = np.flatnonzero(np.asarray(region_id) == k)
        if len(pool) == 0:
            pool = np.arange(len(region_id))
        replace = nk > len(pool)
        out.append(rng.choice(pool, size=nk, replace=replace))
    ans = np.concatenate(out).astype(np.int64)
    rng.shuffle(ans)
    return ans



def expand_centers_both_surfaces(
    centers: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Expand geometric center nodes to paired shell-surface material states.

    Input:
        centers = [i0, i1, i2, ...]

    Output:
        expanded_centers = [i0, i0, i1, i1, i2, i2, ...]
        surfaces         = [ 0,  1,  0,  1,  0,  1, ...]

    surface 0 = outer / SPOS
    surface 1 = inner / SNEG

    Therefore:
        N geometric centers -> 2*N material-state samples.
    """
    centers = np.asarray(centers, dtype=np.int64).reshape(-1)

    expanded_centers = np.repeat(centers, 2)

    surfaces = np.tile(
        np.asarray([0, 1], dtype=np.int64),
        len(centers),
    )

    return expanded_centers, surfaces


def balanced_surfaces(n: int, rng: np.random.Generator) -> np.ndarray:
    s = np.zeros(int(n), dtype=np.int64)
    s[int(n)//2:] = 1
    rng.shuffle(s)
    return s


def to_torch(batch: dict, device: torch.device) -> dict[str, torch.Tensor]:
    out = {}
    for k, v in batch.items():
        if k in ("centers", "surfaces"):
            continue
        if isinstance(v, np.ndarray):
            if v.dtype == np.bool_:
                out[k] = torch.from_numpy(v).to(device=device, dtype=torch.bool, non_blocking=True)
            else:
                out[k] = torch.from_numpy(np.asarray(v)).to(device=device, dtype=torch.float32, non_blocking=True)
    return out
