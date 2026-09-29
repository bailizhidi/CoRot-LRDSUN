# V2 Transolver training guide

This document is the handoff from Step 1.5 integration to the first formal
V2 training run.  It describes the command and the contracts; it does not
start a training job.

## Step 1.5 acceptance record

Slurm job `1622106` completed on one RTX 5090 with:

```text
sample=119, time=0
geometric centers=1024
material tokens=2048
token shape=[2048,192]
status=PASS
forward=true
loss=true
backward=true
optimizer_step_called=false
```

The real V1 checkpoint loaded all 60 V1 model keys.  The V2 token projection,
Physics Attention, and token adapter remained newly initialized.  The V1
encoder, trunk, delta8, PEEQ gate/magnitude, and LE heads all reported zero
missing keys.  The measured smoke peak was 515.951 MB allocated and 576 MB
reserved on the RTX 5090.

The result and logs stay on the server.  They are not part of the V2 code
sync.

## Training contract

`train_token_transolver.py` reuses the V1 differentiable five-step contract:

```text
z_t^GT -> zhat_(t+1) -> zhat_(t+2) -> zhat_(t+3)
       -> zhat_(t+4) -> zhat_(t+5)
```

The step weights are:

```text
[1.0, 0.75, 0.50, 0.35, 0.25]
```

Intermediate predicted states stay attached to the graph.  The normal formal
configuration uses 1024 geometric centers and therefore 2048 material tokens
per sample/time sequence.  The sequence is never spatially chunked because
chunking would change the Physics Attention neighborhood.  Each model call
contains one FE sample and one time frame, so sample or frame mixing is not
introduced by DDP.

The V1 checkpoint is a warm start only.  V2 parameters under
`token_in`, `physics_attention`, and `token_adapter` are initialized by the V2
model.  The output contract remains:

```text
delta8_norm, plastic_logit, peeq_mag_raw, le_norm
```

## Submission

The real four-rank DDP smoke passed on job `1623217`.  It verified NCCL,
four ranks, 2048 tokens per rank, one forward/loss/backward pass, synchronized
gradients, and no unused trainable parameters.  It created no optimizer and
saved no checkpoint.

The two-epoch pilot wrapper is:

```text
slurm/run_token_transolver_pilot_4gpu.sh
```

It was submitted only as the controlled Step 2 pilot (job `1623302`); no
formal 30-epoch run was submitted.

## Step 2 pilot result

Job `1623302` completed with Slurm state `COMPLETED` and exit code `0` on
four RTX 5090 ranks. It used C1024 (`1024` geometric centers and `2048`
material-state tokens per rank), real V1 cache data, the V1
`best_val_5step.pt` warm start, two epochs, and the unchanged five-step loss
weights. `optimizer.step()` was enabled for `120` steps per epoch; no NaN or
Inf was observed.

The pilot summary is server-side at:

```text
outputs/token_transolver_pilot_1623302/pilot_summary.json
```

The measured totals were:

```text
epoch 1: train=0.07075137, validation=0.08345900, time=12.33 s
epoch 2: train=0.08709775, validation=0.07999493, time=11.69 s
```

Physics-Attention received nonzero gradients (`0.00253837` and `0.00186754`
L2 averaged over the two epochs), and its 24 parameters changed by L2
`0.11796819`. The token adapter gradient L2 values were `0.61550021` and
`0.32487536`; the V1 trunk gradient L2 values were `0.94268935` and
`0.63654405`. The peak per-rank memory was `1806.296 MB` allocated and
`1890 MB` reserved. The pilot summary reports
`ready_for_formal_training=true`; formal training still requires an explicit
user decision and is not launched by this repository.

The control-variable audit for the first formal experiment is recorded in
[`FORMAL_TRAINING_AUDIT_C1024.md`](FORMAL_TRAINING_AUDIT_C1024.md). Submit the
dedicated C1024 wrapper from the V2 project directory so `SLURM_SUBMIT_DIR`
identifies the project after Slurm spools the script:

```bash
cd /data/home/scxk573/run/lsz/ysy/physicsnemo/my_experiments/CoRot-LRDSUN-Constitutive-V2

sbatch --gpus=4 -p gpu_5090 \
  slurm/run_token_transolver_formal_c1024_5step_4gpu.sh
```

This wrapper pins the V1 `best_val_5step.pt` path, 30 epochs, 15 train
windows, 18 validation windows, 1024 centers, 2048 tokens, the V1 AdamW and
Cosine settings, and the isolated output directory
`outputs/formal_token_transolver_c1024_5step/`. C2048 is not selected. The
formal wrapper completes all 30 epochs and does not automatically stop on
normal validation fluctuations. The generic `run_token_transolver_train_4gpu.sh`
remains available for later controlled experiments.

To override the pinned checkpoint or run length explicitly for a separately
audited experiment:

```bash
WARM_START=/data/home/scxk573/run/lsz/ysy/physicsnemo/my_experiments/CoRot-LRDSUN-Constitutive-V1/outputs/finetune_5step/run_20260920_143444_1610485/best_val_5step.pt \
EPOCHS=30 \
CENTERS_PER_WINDOW=1024 \
sbatch --gpus=4 -p gpu_5090 slurm/run_token_transolver_train_4gpu.sh
```

Do not add `--mem` on `gpu_4090` or `gpu_5090`; those partitions assign memory
from the GPU count.  The wrapper requests no training action until it has a
valid cache and warm-start checkpoint.

## Formal C1024 result

The first formal run used jobs `1623360` and continuation `1624702` (the
continuation completed epochs 28–30 after the initial run stopped at 27; no
Pilot checkpoint was used). Both jobs completed successfully. The final
server-side summary is:

```text
outputs/formal_token_transolver_c1024_5step/formal_summary.json
```

It contains 30 epochs, best validation total `0.0789974845` at epoch 19,
final validation total `0.0854530161`, finite losses throughout, nonzero
Physics-Attention gradients, and the per-epoch component/step/memory records.
The selected and final checkpoints are `best_val.pt` and `last.pt` in the
same isolated directory. Logs and checkpoints remain on the server and are
not part of the local code synchronization.

## Outputs and monitoring

The formal wrapper creates a server-side run directory under:

```text
CoRot-LRDSUN-Constitutive-V2/outputs/formal_token_transolver_c1024_5step
```

Training writes `last.pt` and stdout history there.  Logs remain under the
server V2 `logs/` directory.  Monitor and cancel jobs with:

```bash
squeue -u scxk573
parajobs
scancel JOBID
```

Evaluation must also be submitted through Slurm.  Set `CHECKPOINT` to the V2
`last.pt` or a later V2 checkpoint and use:

```bash
CHECKPOINT=/path/to/v2/last.pt \
sbatch --gpus=1 -p gpu_4090 slurm/run_token_transolver_eval.sh
```

## Code-only synchronization

The local and server V2 source trees are kept identical by comparing hashes of
Python, shell, and Markdown files.  The following are deliberately excluded
from that comparison and from code synchronization:

```text
logs/
outputs/
prepared_cache_prod_v1/
*.pt
*.npy
*.npz
__pycache__/
```

V1 remains read-only.  The V1 cache and V1 checkpoint are referenced by path
and are never copied into the V2 source tree.
