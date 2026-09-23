# CoRot-LRDSUN 应力误差空间分区诊断 v1

目的：定量判断 `Outer_SPOS_S_Mises_AbsError_MPa` 是否主要集中在两端、最外侧 ring、直管段，而不是弯曲塑性区。

## 输出
每个 VTU/NPZ：
- `ring_profile.csv`：逐轴向 ring 的 MAE/P95/Max 与 `s/L`
- `region_metrics.csv`：end_left / straight_left / bend / straight_right / end_right
- `exclude_end_rings.csv`：排除 0~5 个端部 ring 后的误差变化
- `ring_error_profile.png`
- `region_mae.png`
- `summary.json`

目录总汇：
- `ALL_region_metrics.csv`
- `ALL_exclude_end_rings.csv`
- `ALL_ring_profiles.csv`
- `ALL_summary.json`

## 原理
1. 用 `X_ref` 做 PCA，得到初始直管轴线；投影得到参考轴向坐标 `s/L`。
2. 沿参考轴向识别网格 ring。
3. 用 `X_def` 的每个 ring 质心形成变形后中心线，并用局部转角自动识别 bend zone。
4. 最外侧 `--end-rings` 个 ring 单独统计。
5. 统计每区 MAE/RMSE/P95/Max，以及该区对全局绝对误差的贡献。
6. 做“排除 0/1/2/3/... 个端部 ring”的敏感性分析。

## 第一步：检查一个文件字段
```bash
python inspect_field_file.py /path/to/one_sample.vtu
```
预期至少存在：
- `X_ref`
- `X_def`（没有时 VTU points 也可作为变形坐标，但当前脚本默认字段名）
- `Outer_SPOS_S_Mises_AbsError_MPa`

## 单文件分析
```bash
python analyze_stress_error_regions.py \
  --input-file /path/to/sample119_final.vtu \
  --out-dir stress_region_diagnosis/sample119 \
  --error-field Outer_SPOS_S_Mises_AbsError_MPa \
  --ref-field X_ref \
  --def-field X_def \
  --end-rings 1
```

## 多文件批量分析
```bash
python analyze_stress_error_regions.py \
  --input-dir /path/to/vtu_folder \
  --glob '*.vtu' \
  --out-dir stress_region_diagnosis/test_samples \
  --error-field Outer_SPOS_S_Mises_AbsError_MPa \
  --ref-field X_ref \
  --def-field X_def \
  --end-rings 1
```

如果目录里包含每个样本很多帧，先用更严格的 `--glob` 只选你想比较的帧，例如：
```bash
--glob '*frame_180*.vtu'
```

## 最关键的判据
看 `ALL_exclude_end_rings.csv`：
- 如果 `exclude=0 -> 1` 时 MAE/P95/Max 大幅下降，而 `1 -> 2 -> 3` 基本稳定：最外一圈是系统性 boundary artifact。
- 如果排除 ring 后仍然两端 straight 区高：继续看 `ALL_region_metrics.csv` 的 `straight_left/right`，说明还有小应变/夹具过渡问题。
- 如果 bend 区反而最高：再考虑本构模型在塑性区的真实失效。

## 依赖
优先使用 `pyvista` 读取 VTU；若无，则尝试 `meshio`。NPZ 只需要 numpy。绘图需要 matplotlib。
