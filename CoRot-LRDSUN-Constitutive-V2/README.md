# CoRot-LRDSUN-Constitutive-V2

This directory is an independent research branch.  The V1 project at
`../CoRot-LRDSUN-Constitutive-V1` is not modified.  V2 reuses V1's tested
cache/geometry/normalization helpers through a read-only compatibility import.

## Current milestone

The first milestone is token representation and Physics-Attention validation.
No training is performed by `test_token_pipeline.py` or by the smoke wrapper.

For one `PreparedSample` and one transition `t`, `TokenDataBuilder` creates:

```text
M geometric centers
    -> M unique/local CoRot geometric representations
    -> 2M material-state rows
       (outer/SPOS + inner/SNEG for every center)
    -> token sequence [2M, d]
```

Each material token is constructed from:

```text
[shared edge-only h_geo, normalized recursive state, normalized context,
 center state/context latent]
```

The sequence carries `center_id`, `surface_id`, `material_state_index`,
`sample_id`, and `time_index`.  It never relies on token row order to identify
SPOS/SNEG.

## Synthetic smoke test

From this directory:

```bash
python test_token_pipeline.py \
  --centers 8 --hidden-dim 32 --token-dim 32 --shuffle
```

The test reports token/output shapes and checks:

- `M -> 2M` expansion;
- explicit surface/center mappings;
- permutation equivariance;
- block-diagonal sample/time isolation.

## Real prepared cache

```bash
python test_token_pipeline.py \
  --cache-dir /path/to/prepared_cache_prod_v1 \
  --sample-id 1 --time 10 --centers 1024 \
  --hidden-dim 192 --token-dim 192 --shuffle
```

Expected formal token shapes are:

```text
C1024: geometric centers=1024, material tokens=2048, token shape=[2048,d]
C2048: geometric centers=2048, material tokens=4096, token shape=[4096,d]
```

## V2 modules

```text
corot_lrdsun_v2/token_data.py
    Builds one isolated sample/time TokenBatch.

corot_lrdsun_v2/token_grouping.py
    Carries (sample_id,time_index) keys and creates block masks.

corot_lrdsun_v2/physics_attention.py
    Minimal mask-aware permutation-equivariant attention/FFN stack.

corot_lrdsun_v2/model_corot_token_transolver.py
    V1 local encoders + material tokens + Physics-Attention + V1 trunk/heads.

train_token_transolver.py
    DDP/AMP/5-step training entry point with an instrumented two-epoch pilot.

evaluate_token_teacher_forcing.py
evaluate_token_rollout.py
    Token-global evaluators; unlike V1 they do not process surfaces separately.
```

## Warm start

`CoRotTokenTransolver.load_v1_checkpoint()` loads matching V1 keys for:

```text
edge_encoder
center_encoder
query/key/value
trunk
delta8/plastic/PEEQ/LE heads
```

The token projection, Physics-Attention stack, and token adapter are new V2
parameters.  The adapter is zero-initialized so the initial trunk path remains
close to the V1 path.

## Slurm wrappers

```bash
sbatch slurm/run_token_transolver_smoke_4gpu.sh
sbatch slurm/run_token_transolver_train_4gpu.sh
CHECKPOINT=/path/to/v2/last.pt sbatch slurm/run_token_transolver_eval.sh
```

The smoke wrapper uses a two-epoch **dry-run contract** when a real cache is
available; it does not call `optimizer.step()`.  This is intentional for the
first milestone.

## Step 1.5: real-data integration smoke

`test_real_token_pipeline.py` is the read-only integration check for the real
V1 prepared cache and `best_val_5step.pt`.  It tests sample 119 at `t=0` for
both C1024 and C2048, loads the V1 encoder/trunk/heads, keeps Physics
Attention newly initialized, evaluates the V1 delta8/PEEQ/gate/LE/Mises loss,
and calls `backward()` without creating an optimizer or calling `step()`.

```bash
python test_real_token_pipeline.py \
  --cache-dir /path/to/prepared_cache_prod_v1 \
  --checkpoint /path/to/best_val_5step.pt \
  --sample-id 119 --time 0 --centers 1024 2048
```

The output is JSON and includes token/output shapes, every loaded/missing/
unexpected checkpoint key, per-loss components, gradient-graph status, and
CUDA allocated/reserved/peak memory for each center count.  The script has no
synthetic fallback: a missing cache or checkpoint is reported as
`BLOCKED_MISSING_REAL_INPUT`.

The formal training handoff, control-variable audit, Slurm submission rules,
and code-only synchronization policy are documented in
[`V2_TRAINING_GUIDE.md`](V2_TRAINING_GUIDE.md) and
[`FORMAL_TRAINING_AUDIT_C1024.md`](FORMAL_TRAINING_AUDIT_C1024.md).  The first
formal C1024 run completed 30 epochs on jobs `1623360` and `1624702`; its
server-side output is isolated under
`outputs/formal_token_transolver_c1024_5step/`.

The real four-rank DDP smoke uses
`test_ddp_token_pipeline.py` through
`slurm/run_token_transolver_smoke_4gpu.sh`.  After its PASS result, the
two-epoch pilot wrapper at
`slurm/run_token_transolver_pilot_4gpu.sh` completed successfully on job
`1623302`.  It used C1024 and is separate from the completed formal V2 run.
