"""Read-only compatibility imports for the V1 implementation.

V2 deliberately lives in a separate directory.  The small adapter here adds
the V1 source directory to ``sys.path`` and imports the tested data, geometry,
normalization, physics, loss, and runtime helpers without editing or copying
the V1 project.
"""

from __future__ import annotations

import sys
from pathlib import Path


V2_ROOT = Path(__file__).resolve().parents[1]
_V1_PROJECT = V2_ROOT.parent / "CoRot-LRDSUN-Constitutive-V1"
_V1_CANDIDATES = (
    _V1_PROJECT / "CoRot-LRDSUN-Constitutive-V1",  # local checkout bundle
    _V1_PROJECT,                                    # cluster project layout
)
V1_ROOT = next((path for path in _V1_CANDIDATES if path.is_dir()), _V1_CANDIDATES[0])

if not V1_ROOT.is_dir():
    raise ImportError(f"V1 source directory not found: {V1_ROOT}")

if str(V1_ROOT) not in sys.path:
    sys.path.insert(0, str(V1_ROOT))

from corot_lrdsun.contracts import (  # noqa: E402
    CONTEXT_DIM,
    EDGE_DIM,
    LE_DIM,
    STATE_DIM,
)
from corot_lrdsun.data import PreparedSample, balanced_centers  # noqa: E402
from corot_lrdsun.geometry import corot_edge_features_for_centers  # noqa: E402
from corot_lrdsun.io import load_manifest, load_json, split_records  # noqa: E402
from corot_lrdsun.losses import compute_losses  # noqa: E402
from corot_lrdsun.model import CoRotLRDSUN as V1CoRotLRDSUN  # noqa: E402
from corot_lrdsun.model import mlp as v1_mlp  # noqa: E402
from corot_lrdsun.normalization import (  # noqa: E402
    delta8_from_norm,
    le_to_norm,
    le_from_norm,
    load_stats,
)
from corot_lrdsun.physics import mises_from_s4_torch  # noqa: E402
from corot_lrdsun.runtime import setup_ddp, barrier, raw_model, save_checkpoint, set_seed  # noqa: E402


__all__ = [
    "V1_ROOT",
    "PreparedSample",
    "balanced_centers",
    "corot_edge_features_for_centers",
    "load_manifest",
    "load_json",
    "split_records",
    "compute_losses",
    "V1CoRotLRDSUN",
    "v1_mlp",
    "load_stats",
    "delta8_from_norm",
    "le_to_norm",
    "le_from_norm",
    "mises_from_s4_torch",
    "setup_ddp",
    "barrier",
    "raw_model",
    "save_checkpoint",
    "set_seed",
    "STATE_DIM",
    "EDGE_DIM",
    "CONTEXT_DIM",
    "LE_DIM",
]
