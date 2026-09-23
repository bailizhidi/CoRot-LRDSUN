# CoRot-LRD-SUN Constitutive V1

## 1. 目标

本工程用于弯管弯曲阶段的局部弹塑性状态恢复。位移场由 FEM 真值或后续 MGN-T 提供，本模型只负责把局部共旋运动学与中心材料历史状态映射为下一步材料状态。

正式递归状态：

```text
z_t = [S4, PE4, PEEQ] = 9D
S4  = [S_aa, S_tt, S_rr, S_at]
PE4 = [PE_aa, PE_tt, PE_rr, PE_at]
```

`LE4=[LE_aa,LE_tt,LE_rr,LE_at]` 是辅助监督，不进入递归状态；`S_mises` 由 `S4` 解析计算，不作为独立预测状态。

核心限制：**邻居节点永远不提供 S/PE/PEEQ，只提供运动学。** 因此 full rollout 不再依赖邻居 FEM 材料真值。

---

## 2. 核心物理思路

### 2.1 固定 2-hop Kabsch 共旋坐标系

Production V1 提供 `X0, U_t, Q0, frame_neighbor_ptr/index`。对中心节点 `i`：

```text
X_t = X0 + U_t
R_i^t = Kabsch({X0_j-X0_i}, {X_t,j-X_t,i})
Q_i^t = R_i^t Q_i^0
```

固定 2-hop stencil 与正式数据提取器一致。

### 2.2 邻居只构造局部运动学

对邻居 `j`：

```text
r0_ij = (Q0_i)^T (X0_j-X0_i)
rt_ij = (Qt_i)^T (Xt_j-Xt_i)
dt_ij = rt_ij-r0_ij
Δd_ij = d(t+1)_ij-dt_ij
```

模型实际使用 12D 邻居特征：

```text
r0/D      3D
d_t/D     3D
Δd/D      3D
|r0|/D    1D
|d_t|/D   1D
|Δd|/D    1D
```

纯刚体平移/旋转时，理论上 `d_t≈0, Δd≈0`。`preflight.py` 包含独立刚体客观性自检。

### 2.3 中心材料历史状态

只有中心材料点携带：

```text
z_i,t = [S4, PE4, PEEQ]
```

outer / inner 作为同一 shell 节点的两个材料表面状态处理。模型上下文显式包含：

```text
R/D
t/D
surface_z/D = ±0.5 t/D
angle_progress
E_modulus
Poisson_Ratio
```

其中 `surface_z/D` 用于区分 SPOS outer 与 SNEG inner；空间 region 不进入模型，只用于训练采样平衡。

### 2.4 局部运动学聚合

邻居 12D 特征先经 MLP 编码，然后由中心状态生成 query，对邻居进行 attention pooling，同时保留 mean/max pooling：

```text
neighbor kinematics -> edge encoder -> attention/mean/max pooling
center state + process context -> center encoder
pooled kinematics + center latent -> constitutive trunk
```

因此该网络是“局部图/邻域集合”的本构更新网络，而不是旧版让所有局部节点携带材料状态的 MeshGraphNet。

### 2.5 增量输出

主输出：

```text
ΔS4, ΔPE4, ΔPEEQ
```

更新：

```text
S4(t+1)   = S4(t)   + ΔS4
PE4(t+1)  = PE4(t)  + ΔPE4
PEEQ(t+1) = PEEQ(t) + ΔPEEQ
```

PEEQ 采用两头设计：

```text
plastic_logit   -> 是否发生塑性增量
peeq_mag_raw    -> 非负塑性增量幅值
```

训练时使用 soft gate，推理/rollout 默认 `gate>=0.5` 才产生非零 `ΔPEEQ`，以减少长期假塑性漂移。

辅助输出：

```text
LE4(t+1)
```

`S_mises` 始终从预测 `S4` 解析计算，避免独立 `S_mises` 与张量分量不一致。

---

## 3. 为什么不直接复用旧 LRD-SUN

旧 CoRot-LRD-SUN 的工程框架很有价值，但其输入把邻居真实 `S/PE/PEEQ` 一并送入网络；旧 rollout 只替换中心预测状态，邻居材料状态仍来自 FEM，因此不是真正独立的本构 rollout。

本 V1 从数据结构上禁止邻居材料状态进入 batch：

```text
center: 9D material state
neighbors: kinematics only
```

这使得：

```text
z_0 -> z_1_hat -> ... -> z_180_hat
```

可以在只给定 `U_0...U_180` 的情况下真实递推。

---

## 4. 工程目录

```text
CoRot-LRDSUN-Constitutive-V1/
├── corot_lrdsun/
│   ├── contracts.py
│   ├── geometry.py
│   ├── io.py
│   ├── data.py
│   ├── normalization.py
│   ├── model.py
│   ├── losses.py
│   ├── metrics.py
│   ├── physics.py
│   └── runtime.py
├── scan_dataset.py
├── prepare_training_cache.py
├── compute_stats.py
├── preflight.py
├── train_ddp.py
├── evaluate_teacher_forcing.py
├── evaluate_rollout.py
├── slurm/
└── VERSION.txt
```

---

## 5. Prepared cache 的目的

canonical Production V1 `.npz` 永远是原始真值。为了训练效率，本工程额外建立 derived prepared cache：

```text
sample_XXXX/
├── X0.npy
├── U.npy
├── Q0.npy
├── ptr.npy
├── idx.npy
├── R.npy
├── region_id.npy
├── angle_progress.npy
├── state_outer.npy
├── state_inner.npy
├── LE_outer.npy
├── LE_inner.npy
├── meta.json
└── PREPARED.ok
```

主要作用：

1. NPZ 拆成 `.npy` 后可 mmap；
2. 每个 `R_i^t` 只执行一次固定 2-hop Kabsch；
3. 不生成数千万 local graph，不再使用旧 MB16 式大规模 graph cache；
4. 训练时仍动态采样 `(sample, transition, center, surface)`。

---

## 6. 推荐服务器安装位置

```bash
cd /data/run01/scxk573/lsz/ysy/physicsnemo/my_experiments/LRD-SUN
```

将本目录放成：

```text
/data/run01/scxk573/lsz/ysy/physicsnemo/my_experiments/LRD-SUN/CoRot-LRDSUN-Constitutive-V1
```

canonical 150 个 Production V1 NPZ 放在同一个数据目录，例如：

```text
/data/run01/scxk573/lsz/ysy/physicsnemo/data/dataset_corot_constitutive_prod_v1
```

如果实际位置不同：

```bash
export RAW_DATA_DIR=/你的/真实/Production_V1_NPZ目录
```

Slurm 默认会读取该环境变量。

---

## 7. 正式运行顺序

### Step 1：准备 cache + train-only stats + preflight

```bash
cd /data/run01/scxk573/lsz/ysy/physicsnemo/my_experiments/LRD-SUN/CoRot-LRDSUN-Constitutive-V1

export RAW_DATA_DIR=/data/run01/scxk573/lsz/ysy/physicsnemo/data/dataset_corot_constitutive_prod_v1

sbatch --gpus=1 -p gpu_5090 ./slurm/run_prepare_cache_1gpu_5090.sh
```

这个 job 依次执行：

```text
scan 150 cases -> expect 120/15/15
prepare mmap cache -> fixed 2-hop Kabsch R_t
compute train-only normalization statistics
rigid-objectivity + contract + forward/backward preflight
```

成功标志：

```text
prepared_cache_prod_v1/CACHE_PREPARED.ok
prepared_cache_prod_v1/stats.json
PREFLIGHT PASSED
```

### Step 2：单独复查 preflight（可选）

```bash
sbatch --gpus=1 -p gpu_5090 ./slurm/run_preflight_1gpu_5090.sh
```

### Step 3：4×5090 DDP smoke

```bash
sbatch --gpus=4 -p gpu_5090 ./slurm/run_ddp_smoke_4gpu_5090.sh
```

该 job 使用 32 CPU，即每 GPU 8 CPU。

Smoke 只检查：

```text
DDP
BF16
forward/backward
checkpoint
validation
```

不用于判断最终精度。

### Step 4：正式 4×5090 训练

```bash
sbatch --gpus=4 -p gpu_5090 ./slurm/run_train_4gpu_5090.sh
```

默认：

```text
epochs                     = 120
hidden_dim                 = 192
batch_size / GPU           = 256
transitions/sample/epoch   = 16
centers/transition         = 512
AdamW lr                   = 2e-4
min_lr                     = 2e-6
BF16                       = on
early-stop start           = 30
early-stop patience        = 20
```

训练输出：

```text
outputs/formal/run_YYYYMMDD_HHMMSS_JOBID/
├── best_val.pt
├── last.pt
├── history.json
└── config.json
```

最新正式 run 路径写入：

```text
formal_latest.txt
```

断点续训：

```bash
export RESUME_CKPT=/path/to/last.pt
sbatch --gpus=4 -p gpu_5090 ./slurm/run_train_4gpu_5090.sh
```

### Step 5：test15 teacher forcing + 真 180-step rollout

```bash
sbatch --gpus=1 -p gpu_5090 ./slurm/run_eval_test_1gpu_5090.sh
```

输出：

```text
test_teacher_forcing.json
test_rollout/rollout_summary.json
test_rollout/sample_XXXX_summary.json
```

核心指标：

```text
S4 component RMSE / MAE / max
S_mises RMSE / MAE / max
PE4 component RMSE / MAE / max
PEEQ RMSE / MAE / max
LE4 component RMSE / MAE / max
plastic gate precision / recall / F1
rollout per-frame drift
final-frame error
```

---

## 8. 直接 Python 调试命令

数据检查：

```bash
python scan_dataset.py \
  --dataset-dir /path/to/dataset_corot_constitutive_prod_v1 \
  --expect-samples 150 --expect-train 120 --expect-val 15 --expect-test 15
```

准备 cache：

```bash
python prepare_training_cache.py \
  --dataset-dir /path/to/dataset_corot_constitutive_prod_v1 \
  --cache-dir ./prepared_cache_prod_v1 \
  --workers 4
```

统计量：

```bash
python compute_stats.py \
  --cache-dir ./prepared_cache_prod_v1 \
  --transitions-per-sample 12 \
  --centers-per-transition 512
```

Preflight：

```bash
python preflight.py --cache-dir ./prepared_cache_prod_v1 --device cuda
```

单 GPU smoke 也可直接：

```bash
python train_ddp.py \
  --cache-dir ./prepared_cache_prod_v1 \
  --output-dir ./outputs/manual_smoke \
  --epochs 2 --smoke
```

---

## 9. 训练采样

每个 epoch 每个 sample 随机选择若干 transition；每个 transition 动态选择 center nodes：

```text
clamp : bending : free = 25% : 50% : 25%
outer : inner          = 50% : 50%
```

region 只用于 sampling，不进入模型特征，避免“节点位置捷径”。

---

## 10. Loss

默认：

```text
L = 1.00 L_delta8
  + 1.00 L_PEEQ
  + 0.20 L_plastic_gate
  + 0.25 L_LE
  + 0.25 L_mises
```

其中：

- `L_delta8`：归一化 `ΔS4 + ΔPE4` MSE；
- `L_PEEQ`：非负 `ΔPEEQ` SmoothL1；
- `L_plastic_gate`：塑性激活 BCE，自动使用 train set 类别不平衡权重；
- `L_LE`：`LE4(t+1)` auxiliary MSE；
- `L_mises`：由预测 `S4(t+1)` 解析得到的 von Mises consistency loss。

所有 normalization statistics 只从 train120 抽样计算，val/test 不参与统计量。

---

## 11. 当前 V1 的正确评价顺序

第一阶段先使用 FEM 真值运动学：

```text
U_FEM -> CoRot kinematics -> CoRot-LRD-SUN -> S/PE/PEEQ/LE
```

先确认 constitutive surrogate 自身可以稳定 180-step rollout。

之后再把运动学替换为 MGN-T：

```text
MGN-T predicted U -> same Kabsch/CoRot -> same trained constitutive model
```

这样可以分别量化：

1. 本构模型误差；
2. MGN-T 位移误差传递到材料场后的额外误差。

不要在第一轮正式训练前就把 MGN-T 预测位移混入训练，否则很难判断误差来源。


## 用户当前服务器路径（2026-09-03）

本包已按以下实际目录设为默认值：

```text
PHYSICSNEMO_ROOT=/data/home/scxk573/run/lsz/ysy/physicsnemo
PROJECT_DIR=/data/home/scxk573/run/lsz/ysy/physicsnemo/my_experiments/CoRot-LRDSUN-Constitutive-V1
RAW_DATA_DIR=/data/home/scxk573/run/lsz/ysy/physicsnemo/data/dataset_corot_constitutive_prod_v1
```

Production NPZ 例如：

```text
corot_constitutive_prod_v1_Base_TC4_sample_0000.npz
```

`scan_dataset.py` 使用 `*.npz` 扫描目录，不依赖固定文件名前缀；真正的数据合同由 NPZ 内的 `sample_id`、`split`、`num_frames` 等字段校验。

推荐解压位置：

```bash
cd /data/home/scxk573/run/lsz/ysy/physicsnemo/my_experiments
tar -xJf CoRot-LRDSUN-Constitutive-V1.tar.xz
cd CoRot-LRDSUN-Constitutive-V1
mkdir -p /data/home/scxk573/run/lsz/ysy/physicsnemo/logs
```

第一阶段提交：

```bash
sbatch --gpus=1 -p gpu_5090 ./slurm/run_prepare_cache_1gpu_5090.sh
```
