"""Token-level data construction for the V2 nonlocal surrogate.

The builder keeps geometric-center data separate from material-state data.
This is important for shell surfaces: one geometric center owns one local
kinematic neighborhood, while SPOS and SNEG receive separate recursive state
and context branches.

No neighbor material state is read by this module.  The only neighbor input is
the V1 CoRot edge feature tensor.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Any

import numpy as np
import torch

from .v1_compat import PreparedSample, corot_edge_features_for_centers


def _stats_array(stats: dict[str, Any] | None, key: str, dim: int) -> np.ndarray:
    if stats is None or key not in stats:
        return np.zeros(dim, dtype=np.float32)
    arr = np.asarray(stats[key], dtype=np.float32).reshape(-1)
    if arr.size != dim:
        raise ValueError(f"stats[{key!r}] must have {dim} entries, got {arr.shape}")
    return arr


def _stats_std(stats: dict[str, Any] | None, key: str, dim: int) -> np.ndarray:
    if stats is None or key not in stats:
        return np.ones(dim, dtype=np.float32)
    arr = np.asarray(stats[key], dtype=np.float32).reshape(-1)
    if arr.size != dim:
        raise ValueError(f"stats[{key!r}] must have {dim} entries, got {arr.shape}")
    return np.maximum(arr, 1.0e-12)


@dataclass
class TokenBatch:
    """One sample/time token sequence plus its provenance.

    ``edge_unique_n`` is indexed by ``edge_center_index``.  It therefore has
    one row per unique geometric center, whereas all state/context/target
    fields have one row per material-state token.  The output row order may be
    permuted; ``center_id``, ``surface_id`` and ``material_state_index`` are
    the only supported way to map predictions back.
    """

    edge_unique_n: torch.Tensor       # [M_unique, K, 12]
    edge_unique: torch.Tensor         # [M_unique, K, 12], unnormalized
    edge_mask: torch.Tensor            # [M_unique, K], bool
    edge_center_index: torch.Tensor    # [N_token], long
    state_n: torch.Tensor              # [N_token, 9]
    context_n: torch.Tensor            # [N_token, 6]
    state: torch.Tensor                # [N_token, 9], current recursive input
    gt_state_t: torch.Tensor           # [N_token, 9]
    next_state: torch.Tensor            # [N_token, 9]
    delta8: torch.Tensor               # [N_token, 8]
    delta_peeq: torch.Tensor           # [N_token, 1]
    le_next: torch.Tensor              # [N_token, 4]
    center_id: torch.Tensor            # [N_token], geometric node id
    surface_id: torch.Tensor           # [N_token], 0=SPOS, 1=SNEG
    material_state_index: torch.Tensor # [N_token], canonical 2*node+surface id
    center_position: torch.Tensor      # [N_token], position in input geometric_centers
    geometric_centers: torch.Tensor    # [M], original requested centers
    sample_id: int
    time_index: int

    @property
    def num_tokens(self) -> int:
        return int(self.state.shape[0])

    @property
    def num_geometric_centers(self) -> int:
        return int(self.geometric_centers.shape[0])

    @property
    def group_key(self) -> tuple[int, int]:
        return int(self.sample_id), int(self.time_index)

    def to(self, device: torch.device | str) -> "TokenBatch":
        device = torch.device(device)
        updates = {}
        for field in fields(self):
            name = field.name
            value = getattr(self, name)
            if isinstance(value, torch.Tensor):
                updates[name] = value.to(device)
        out = replace(self, **updates)
        for name in ("_state_mean", "_state_std"):
            if hasattr(self, name):
                setattr(out, name, getattr(self, name).to(device))
        return out

    def with_state_override(self, state: torch.Tensor) -> "TokenBatch":
        """Replace only the recursive input state, preserving GT targets."""
        if tuple(state.shape) != tuple(self.state.shape):
            raise ValueError(
                f"state override shape {tuple(state.shape)} != {tuple(self.state.shape)}"
            )
        # The statistics are not stored separately in the dataclass.  The
        # builder attaches them as private attributes after construction.
        state_mean = getattr(self, "_state_mean", None)
        state_std = getattr(self, "_state_std", None)
        if state_mean is None or state_std is None:
            raise RuntimeError("TokenBatch cannot normalize an override without builder stats")
        out = replace(self, state=state, state_n=(state - state_mean) / state_std)
        out._state_mean = state_mean
        out._state_std = state_std
        return out

    def permuted(self, order: torch.Tensor | np.ndarray) -> "TokenBatch":
        """Return a row-permuted copy for equivariance/isolation tests."""
        if isinstance(order, np.ndarray):
            order = torch.from_numpy(order.astype(np.int64, copy=False))
        order = order.to(device=self.state.device, dtype=torch.long)
        n = self.num_tokens
        if tuple(order.shape) != (n,):
            raise ValueError(f"order must have shape [{n}], got {tuple(order.shape)}")
        row_names = (
            "edge_center_index", "state_n", "context_n", "state", "gt_state_t",
            "next_state", "delta8", "delta_peeq", "le_next", "center_id",
            "surface_id", "material_state_index", "center_position",
        )
        updates = {name: getattr(self, name).index_select(0, order) for name in row_names}
        out = replace(self, **updates)
        for name in ("_state_mean", "_state_std"):
            if hasattr(self, name):
                setattr(out, name, getattr(self, name))
        return out

    def as_loss_batch(self) -> dict[str, torch.Tensor]:
        """Expose the V1 loss contract without exposing neighbor states."""
        return {
            "state": self.state,
            "gt_state_t": self.gt_state_t,
            "next_state": self.next_state,
            "delta8": self.delta8,
            "delta_peeq": self.delta_peeq,
            "le_next": self.le_next,
        }


class TokenDataBuilder:
    """Build one isolated token sequence for a ``PreparedSample`` and time ``t``."""

    def __init__(self, stats: dict[str, Any] | None = None, dtype=torch.float32):
        self.stats = stats or {}
        self.dtype = dtype
        self._edge_mean = torch.as_tensor(_stats_array(self.stats, "edge_mean", 12), dtype=dtype)
        self._edge_std = torch.as_tensor(_stats_std(self.stats, "edge_std", 12), dtype=dtype)
        self._state_mean = torch.as_tensor(_stats_array(self.stats, "state_mean", 9), dtype=dtype)
        self._state_std = torch.as_tensor(_stats_std(self.stats, "state_std", 9), dtype=dtype)
        self._context_mean = torch.as_tensor(_stats_array(self.stats, "context_mean", 6), dtype=dtype)
        self._context_std = torch.as_tensor(_stats_std(self.stats, "context_std", 6), dtype=dtype)

    def build(
        self,
        sample: PreparedSample,
        t: int,
        geometric_centers: np.ndarray,
        *,
        shuffle: bool = False,
        seed: int | None = None,
        state_override: np.ndarray | torch.Tensor | None = None,
    ) -> TokenBatch:
        """Build ``2M`` material tokens for one sample and one transition.

        ``state_override`` is optional and must be in canonical paired order
        ``[outer(center0), inner(center0), ...]`` before the optional shuffle.
        It is intended for autoregressive rollout; GT targets remain attached
        separately and are never used as model inputs.
        """
        t = int(t)
        if t < 0 or t + 1 >= int(sample.T):
            raise ValueError(f"t={t} is invalid for sample with T={sample.T}")
        c_geom = np.asarray(geometric_centers, dtype=np.int64).reshape(-1)
        if c_geom.size == 0:
            raise ValueError("At least one geometric center is required")
        if np.any(c_geom < 0) or np.any(c_geom >= int(sample.N)):
            raise IndexError(f"geometric center outside [0,{sample.N})")

        unique_centers, inverse_unique = np.unique(c_geom, return_inverse=True)
        edge_unique_np, edge_mask_np = corot_edge_features_for_centers(
            sample.X0,
            sample.U,
            sample.Q0,
            sample.R,
            sample.ptr,
            sample.idx,
            t,
            unique_centers,
            sample.D,
        )

        # Canonical material-state order is [outer, inner] per geometric
        # center.  All metadata is permuted together below.
        centers_c = np.repeat(c_geom, 2)
        surfaces_c = np.tile(np.asarray([0, 1], dtype=np.int64), len(c_geom))
        center_position_c = np.repeat(inverse_unique, 2)
        center_id_c = centers_c.copy()
        material_index_c = 2 * center_id_c + surfaces_c

        gt_state = sample.states_at(t, centers_c, surfaces_c).astype(np.float32, copy=False)
        next_state = sample.states_at(t + 1, centers_c, surfaces_c).astype(np.float32, copy=False)
        le_next = sample.le_at(t + 1, centers_c, surfaces_c).astype(np.float32, copy=False)

        zsurf = np.where(
            surfaces_c == 0,
            +0.5 * float(sample.t_over_D),
            -0.5 * float(sample.t_over_D),
        ).astype(np.float32)
        context = np.stack(
            [
                np.full(len(centers_c), sample.R_over_D, np.float32),
                np.full(len(centers_c), sample.t_over_D, np.float32),
                zsurf,
                np.full(len(centers_c), float(sample.angle_progress[t]), np.float32),
                np.full(len(centers_c), sample.E_modulus, np.float32),
                np.full(len(centers_c), sample.nu, np.float32),
            ],
            axis=1,
        )

        if state_override is None:
            state = gt_state.copy()
        else:
            state = np.asarray(state_override, dtype=np.float32)
            if state.shape != gt_state.shape:
                raise ValueError(
                    "state_override must be canonical [2M,9], "
                    f"got {state.shape}, expected {gt_state.shape}"
                )

        if shuffle:
            rng = np.random.default_rng(seed)
            order = rng.permutation(len(centers_c))
        else:
            order = np.arange(len(centers_c), dtype=np.int64)

        centers = centers_c[order]
        surfaces = surfaces_c[order]
        center_position = center_position_c[order]
        center_id = center_id_c[order]
        material_index = material_index_c[order]
        state = state[order]
        gt_state = gt_state[order]
        next_state = next_state[order]
        le_next = le_next[order]
        context = context[order]

        edge_unique = torch.as_tensor(edge_unique_np, dtype=self.dtype)
        edge_unique_n = (edge_unique - self._edge_mean) / self._edge_std
        state_t = torch.as_tensor(state, dtype=self.dtype)
        gt_state_t = torch.as_tensor(gt_state, dtype=self.dtype)
        next_state_t = torch.as_tensor(next_state, dtype=self.dtype)
        context_t = torch.as_tensor(context, dtype=self.dtype)
        state_n = (state_t - self._state_mean) / self._state_std
        context_n = (context_t - self._context_mean) / self._context_std
        delta = next_state_t - gt_state_t

        out = TokenBatch(
            edge_unique_n=edge_unique_n,
            edge_unique=edge_unique,
            edge_mask=torch.as_tensor(edge_mask_np, dtype=torch.bool),
            edge_center_index=torch.as_tensor(center_position, dtype=torch.long),
            state_n=state_n,
            context_n=context_n,
            state=state_t,
            gt_state_t=gt_state_t,
            next_state=next_state_t,
            delta8=delta[:, :8],
            delta_peeq=delta[:, 8:9],
            le_next=torch.as_tensor(le_next, dtype=self.dtype),
            center_id=torch.as_tensor(center_id, dtype=torch.long),
            surface_id=torch.as_tensor(surfaces, dtype=torch.long),
            material_state_index=torch.as_tensor(material_index, dtype=torch.long),
            center_position=torch.as_tensor(center_position, dtype=torch.long),
            geometric_centers=torch.as_tensor(c_geom, dtype=torch.long),
            sample_id=int(sample.meta.get("sample_id", -1)),
            time_index=t,
        )
        # Keep normalization tensors attached for differentiable rollout state
        # overrides.  They are buffers logically, not model parameters.
        out._state_mean = self._state_mean
        out._state_std = self._state_std
        return out


def encode_shared_h_geo(
    batch: TokenBatch,
    edge_encoder: torch.nn.Module | None = None,
    *,
    he_unique: torch.Tensor | None = None,
):
    """Encode/pool one shared geometric latent per material-state row.

    The encoder is injected so V2 can use the V1 ``edge_encoder`` weights.
    The returned ``h_geo`` is edge-only (mean/max pooling) and therefore the
    same for SPOS/SNEG rows belonging to the same geometric center.
    """
    if he_unique is None:
        if edge_encoder is None:
            raise ValueError("either edge_encoder or precomputed he_unique is required")
        he_unique = edge_encoder(batch.edge_unique_n)  # [M_unique,K,H]
    he = he_unique.index_select(0, batch.edge_center_index)
    mask = batch.edge_mask.index_select(0, batch.edge_center_index)
    denom = mask.sum(dim=1, keepdim=True).clamp_min(1).float()
    pool_mean = (he * mask[..., None].float()).sum(dim=1) / denom
    hmasked = he.masked_fill(~mask[..., None], -1.0e4)
    pool_max = hmasked.max(dim=1).values
    h_geo = torch.cat([pool_mean, pool_max], dim=-1)
    material_features = torch.cat([h_geo, batch.state_n, batch.context_n], dim=-1)
    return h_geo, material_features


def surface_center_to_canonical_states(values: np.ndarray | torch.Tensor):
    """Convert ``[surface, center, ...]`` to paired material-state rows.

    The returned leading order is exactly ``[SPOS_0, SNEG_0, SPOS_1,
    SNEG_1, ...]``.  Keeping this conversion explicit prevents a plain
    reshape from silently producing surface-major ordering.
    """

    if values.ndim < 2 or int(values.shape[0]) != 2:
        raise ValueError(
            "values must have shape [2,num_centers,...], "
            f"got {tuple(values.shape)}"
        )
    tail = tuple(values.shape[2:])
    if isinstance(values, torch.Tensor):
        return values.transpose(0, 1).reshape(-1, *tail)
    array = np.asarray(values)
    return np.swapaxes(array, 0, 1).reshape(-1, *tail)


def token_rows_to_surface_center_states(
    values: np.ndarray,
    center_position: np.ndarray,
    surface_id: np.ndarray,
    num_centers: int,
) -> np.ndarray:
    """Scatter token-row values into explicit ``[surface, center, ...]`` form."""

    array = np.asarray(values)
    positions = np.asarray(center_position, dtype=np.int64).reshape(-1)
    surfaces = np.asarray(surface_id, dtype=np.int64).reshape(-1)
    n = int(num_centers)
    if array.shape[0] != positions.size or positions.size != surfaces.size:
        raise ValueError("values, center_position, and surface_id must have equal row counts")
    if positions.size != 2 * n:
        raise ValueError(f"expected exactly 2*num_centers={2*n} token rows, got {positions.size}")
    if np.any(positions < 0) or np.any(positions >= n) or np.any((surfaces < 0) | (surfaces > 1)):
        raise ValueError("invalid center_position or surface_id")
    key = 2 * positions + surfaces
    if not np.array_equal(np.sort(key), np.arange(2 * n, dtype=np.int64)):
        raise ValueError("token rows do not form a one-to-one paired surface/center mapping")
    out = np.empty((2, n, *array.shape[1:]), dtype=array.dtype)
    out[surfaces, positions] = array
    return out


__all__ = [
    "TokenBatch",
    "TokenDataBuilder",
    "encode_shared_h_geo",
    "surface_center_to_canonical_states",
    "token_rows_to_surface_center_states",
]
