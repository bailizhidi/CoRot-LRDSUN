# CoRot-LRDSUN-Constitutive-V1: context for code audit

## Goal

This repository implements a constitutive surrogate for tube bending.

Current high-level pipeline:

Geometry / process
→ displacement trajectory U_t
→ CoRot-LRDSUN constitutive model
→ S_t, PE_t, PEEQ_t, LE_t

The immediate research task is NOT to replace the baseline blindly.

We want to investigate a new architecture:

local subgraph
→ CoRot objective local representation
→ local encoder
→ one latent token per material state
→ Transolver / Physics-Attention across tokens from the SAME FE sample and SAME time step
→ center material-state update.

## Current recursive state

z_t = [S4, PE4, PEEQ]

S4 consists of four local stress components.
PE4 consists of four local plastic-strain components.
PEEQ is scalar.

LE is auxiliary/non-recursive.
Mises is computed analytically from the predicted stress tensor components.

## Important current contracts

- local neighborhood is CoRot/objective;
- center material state is recursive;
- neighbors provide kinematics, not their ground-truth material state;
- shell surfaces use paired SPOS/SNEG material states;
- geometric-center sampling is expanded into two material states;
- the current production model supports differentiable multi-step unrolling;
- the current 5-step weights are:
  [1.0, 0.75, 0.5, 0.35, 0.25].

## Important comparison

Baseline 5-step C1024:
- 1024 geometric centers/window
- 2048 paired material states/window
- batch size 2048

Spatial-data experiment C2048:
- 2048 geometric centers/window
- 4096 paired material states/window
- batch size 4096

The intended Transolver extension must NOT mix tokens from different FE samples
inside the same attention sequence.

## First Codex task

Before editing code, identify:

1. exact local CoRot construction;
2. local-neighbor encoder;
3. local attention/pooling;
4. recursive-state encoder;
5. context encoder;
6. fusion point before constitutive trunk;
7. output heads;
8. SPOS/SNEG representation;
9. centers/batch semantics;
10. differentiable 5-step unrolling;
11. tensor shapes;
12. checkpoint loading/saving.

Then propose the smallest safe insertion point for a Transolver-style block.

Do not overwrite the existing production baseline.
