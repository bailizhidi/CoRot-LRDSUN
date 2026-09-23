# Self-tested status

The package was syntax-checked with `python -m compileall` and all Slurm scripts were checked with `bash -n`.

A synthetic 181-frame shell-like dataset was used to exercise the full software path:

```text
raw Production-V1-like NPZ
-> scan
-> parallel prepared-cache build
-> train-only statistics
-> preflight forward/backward
-> teacher-forcing evaluation
-> 180-step true autoregressive rollout
```

The rigid-objectivity self-test produced a maximum absolute co-rotational deformation residual of approximately `3.2e-7` under a finite global rigid rotation + translation, below the built-in `2e-5` acceptance threshold.

GPU DDP execution itself cannot be exercised in the packaging container; the DDP/Slurm structure follows the previously supplied working LRD-SUN 4-GPU pattern and is intended to be validated first with `run_ddp_smoke_4gpu_5090.sh` on the target cluster.
