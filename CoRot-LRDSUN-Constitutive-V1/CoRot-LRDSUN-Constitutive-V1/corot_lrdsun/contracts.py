from __future__ import annotations

STATE_DIM = 9
DELTA8_DIM = 8
LE_DIM = 4
EDGE_DIM = 12
CONTEXT_DIM = 6

STATE_NAMES = [
    "S_aa", "S_tt", "S_rr", "S_at",
    "PE_aa", "PE_tt", "PE_rr", "PE_at", "PEEQ",
]
DELTA8_NAMES = [
    "dS_aa", "dS_tt", "dS_rr", "dS_at",
    "dPE_aa", "dPE_tt", "dPE_rr", "dPE_at",
]
LE_NAMES = ["LE_aa", "LE_tt", "LE_rr", "LE_at"]
EDGE_NAMES = [
    "r0_a_over_D", "r0_t_over_D", "r0_r_over_D",
    "d_t_a_over_D", "d_t_t_over_D", "d_t_r_over_D",
    "dd_a_over_D", "dd_t_over_D", "dd_r_over_D",
    "r0_norm_over_D", "d_t_norm_over_D", "dd_norm_over_D",
]
CONTEXT_NAMES = [
    "R_over_D", "t_over_D", "surface_z_over_D",
    "angle_progress", "E_modulus", "Poisson_Ratio",
]

SURFACE_OUTER = 0
SURFACE_INNER = 1
SURFACE_NAMES = {0: "outer", 1: "inner"}
N_FRAMES = 181
N_TRANSITIONS = 180

PREPARED_CACHE_FORMAT = "corot_lrdsun_prepared_cache_v1"
STATS_FORMAT = "corot_lrdsun_stats_v1"
MODEL_CONTRACT = "center_9D_state + neighbor_kinematics_only -> center_state_increment; LE auxiliary"
