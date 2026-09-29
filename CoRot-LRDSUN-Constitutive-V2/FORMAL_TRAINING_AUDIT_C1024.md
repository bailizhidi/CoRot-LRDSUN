# FORMAL TRAINING AUDIT — V2-Transolver-C1024-5Step

This audit compares the V1 run configuration recorded in
`CoRot-LRDSUN-Constitutive-V1/outputs/finetune_5step/run_20260920_143444_1610485/config.json`
and `slurm/run_5step_finetune_4gpu_4090.sh` with the V2 formal wrapper and
`train_token_transolver.py` before submission.

| Control variable | V1 value | V2 value | Same / different | Reason |
|---|---|---|---|---|
| V1 warm start | `best_val_5step.pt` from run `20260920_143444_1610485` | Same absolute checkpoint | Same | Required direct warm start; never uses the Pilot checkpoint |
| Train/validation split | 120 / 15 records from the prepared manifest | Same cache and split helpers | Same | V2 imports the read-only V1 manifest/split contract |
| Epochs | 30 maximum | 30 maximum | Same | Formal C1024 experiment |
| Early stopping | start 18, patience 8 in metadata; V1 completed all 30 epochs | disabled for formal run; 30 epochs are always completed | Different intentionally | User explicitly required no automatic stop from normal loss fluctuation |
| DDP world size | 4 | 4 | Same | One process per GPU |
| GPU/CPU allocation | 4 GPUs, 32 CPUs | 4 RTX5090, 32 CPUs | Same resource shape | V2 wrapper uses the requested `gpu_5090` partition |
| Geometric centers per rank/window | 1024 | 1024 | Same | C2048 is not used |
| Material states per call | 2048 (`batch_size=2048`) | 2048 full material-state tokens | Same effective count | V2 keeps the complete token sequence for nonlocal attention |
| Train windows/sample/epoch | 15 | 15 | Same | Same cyclic temporal coverage |
| Validation windows/sample | 18 | 18 | Same | V2 now uses the V1 random validation-window contract |
| Validation centers | 1024 | 1024 | Same | Same spatial validation sampling |
| Center sampler | V1 `balanced_centers` | Same V1 helper | Same | Same region-balanced sampling and seed schedule |
| SPOS/SNEG expansion | paired outer/inner states | paired SPOS/SNEG tokens | Same | Required shell material-state contract |
| Token row order | random paired permutation | canonical paired order | Different by representation | V2 has no absolute token-index embedding and passes permutation-equivariance smoke; attention semantics are unchanged |
| Normalization/statistics | prepared-cache `stats.json` | Same `stats.json` and V1 normalization helpers | Same | No new normalization |
| Optimizer | AdamW | AdamW | Same | No optimizer redesign |
| Initial learning rate | `1e-5` | `1e-5` | Same | V1 formal value |
| Final learning rate | `1e-6` | `1e-6` | Same | Cosine scheduler floor |
| Weight decay | `0.01` | `0.01` | Same | V1 formal value |
| Scheduler | `CosineAnnealingLR(T_max=30, eta_min=1e-6)` | Same | Same | Same epoch-level schedule |
| Gradient clipping | norm `1.0` | norm `1.0` | Same | Same clipping contract |
| Five-step weights | `[1.0, 0.75, 0.50, 0.35, 0.25]` | Same | Same | No objective change |
| Recursive state | `z=[S4, PE4, PEEQ]` | Same | Same | LE remains auxiliary/nonrecursive |
| Recursive detach | no detach at steps 2–5 | no detach at steps 2–5 | Same | Differentiable 5-step BPTT |
| Loss components | delta8, PEEQ, gate, LE, analytical Mises | Same V1 helpers and weights | Same | No independent Mises head |
| AMP | CUDA bfloat16 enabled | CUDA bfloat16 enabled | Same | Same default precision path |
| Hidden dimension | 192 | 192 | Same | V1 trunk-compatible dimension |
| Dropout | 0.0 | 0.0 | Same | No regularization change |
| Best checkpoint criterion | minimum validation total | minimum `validation_total` | Same | V2 writes `best_val.pt` in an isolated output directory |
| Output directory | V1 `outputs/finetune_5step/...` | `outputs/formal_token_transolver_c1024_5step/` | Different intentionally | Prevents V1/Pilot overwrite |
| New model modules | none | token projection + Physics-Attention + token adapter | Different intentionally | The sole architectural treatment under study |

The only non-architectural representation difference is material-token row
ordering. V2 uses canonical `[SPOS(center0), SNEG(center0), ...]` order so the
recursive state override remains unambiguous across all five steps. The V2
attention stack has no positional embedding and has passed the permutation
equivariance and sample/time isolation checks, so this does not introduce a
control-variable change in the modeled function.

Formal output is isolated at:

```text
outputs/formal_token_transolver_c1024_5step/
```

The V1 checkpoint is loaded directly from:

```text
/data/home/scxk573/run/lsz/ysy/physicsnemo/my_experiments/CoRot-LRDSUN-Constitutive-V1/outputs/finetune_5step/run_20260920_143444_1610485/best_val_5step.pt
```
