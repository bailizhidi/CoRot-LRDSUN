"""V2 nonlocal token-based CoRot constitutive research branch."""

from .model_corot_token_transolver import CoRotTokenTransolver
from .physics_attention import PhysicsAttention, PhysicsAttentionBlock, PhysicsAttentionStack
from .token_data import (
    TokenBatch,
    TokenDataBuilder,
    encode_shared_h_geo,
    surface_center_to_canonical_states,
    token_rows_to_surface_center_states,
)
from .token_grouping import (
    PackedTokenGroups,
    TokenGroupKey,
    assert_no_sample_or_frame_mixing,
    build_block_diagonal_mask,
    pack_token_groups,
)

__all__ = [
    "CoRotTokenTransolver",
    "PhysicsAttention",
    "PhysicsAttentionBlock",
    "PhysicsAttentionStack",
    "TokenBatch",
    "TokenDataBuilder",
    "encode_shared_h_geo",
    "surface_center_to_canonical_states",
    "token_rows_to_surface_center_states",
    "PackedTokenGroups",
    "TokenGroupKey",
    "assert_no_sample_or_frame_mixing",
    "build_block_diagonal_mask",
    "pack_token_groups",
]
