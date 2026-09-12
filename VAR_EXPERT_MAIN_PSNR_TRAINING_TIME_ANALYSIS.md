# VarExpert-INR Main 日志：PSNR–Training Time 分析

## 1. 分析范围

本文分析 VarExpert-INR 在两个多属性数据集上的标准 Main 运行：

- **Ionization**：`runs/var-expert-ionization/20260716_133736_725222/logs/run_20260716_133736_725222.log`
- **Combustion 40NH3 1**：`runs/var-expert-combustion-40NH3-1/20260807_070953_263505/logs/run_20260807_070953_263505.log`

两次运行均训练 600 个主训练 epoch，并在 epoch 100、200、300、400、500、600 记录一次逐属性 PSNR。本文使用每条 `PSNR epoch` 日志中的 `time=...s` 作为横轴时间；该时间从 600-epoch 主训练阶段开始计时，**不包含前置的 5 个 pretrain epoch**，但包含主训练过程中已经发生的日志、数据加载和定期 PSNR 计算等墙钟开销。

配置中的 `psnr_sample_ratio` 为 0.1。代码在训练开始前用固定 seed=42 只构造一次 PSNR 子集，因此六个 checkpoint 使用的是**同一个固定 10% 子集**。曲线适合比较同一运行内的训练进展，但数值不应直接解释为全量数据集 PSNR。

日志中的 `aggregate` 是所有属性 PSNR 的算术平均值。绘图仅画各属性，不额外绘制 `aggregate`，以满足“一条线一个属性”。

## 2. 图

### Ionization

![Ionization per-attribute PSNR versus logged training time](Metric_Fig_Result/Fig/Main/VarExpert-Training-Curves/ionization_psnr_training_time.png)

### Combustion 40NH3 1

![Combustion per-attribute PSNR versus logged training time](Metric_Fig_Result/Fig/Main/VarExpert-Training-Curves/combustion_40NH3_1_psnr_training_time.png)

两张图的下方横轴是日志累计训练时间（小时），上方横轴是对应 epoch；每个数据集一张图，每条线对应一个属性。

## 3. 每个 Epoch、每个属性的 PSNR 原值

下表直接整理自两份训练日志的 `PSNR epoch` 记录，PSNR 单位均为 dB。时间是同一行中记录的累计主训练时间。

### 3.1 Ionization：全部属性

| Epoch | 时间 (h) | GT | H2 | H_plus | He | PD |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | 2.622 | 41.56 | 46.22 | 37.82 | 37.45 | 43.79 |
| 200 | 5.263 | 46.53 | 51.99 | 45.81 | 45.28 | 46.27 |
| 300 | 7.896 | 44.52 | 49.38 | 40.95 | 40.18 | 46.96 |
| 400 | 10.551 | 49.05 | 54.12 | 47.60 | 46.96 | 51.28 |
| 500 | 13.193 | 47.87 | 52.18 | 44.61 | 43.70 | 50.02 |
| 600 | 15.831 | 50.84 | 55.78 | 49.39 | 48.87 | 52.90 |

### 3.2 Combustion：压力、热释放率与温度

| Epoch | 时间 (h) | Absolute_Pressure | Pressure | Chemistry_Heat_Release_Rate | Temperature |
|---:|---:|---:|---:|---:|---:|
| 100 | 3.542 | 44.50 | 44.50 | 41.29 | 46.67 |
| 200 | 7.094 | 45.52 | 45.52 | 42.03 | 47.41 |
| 300 | 10.612 | 45.56 | 45.58 | 42.38 | 47.74 |
| 400 | 14.121 | 45.85 | 45.86 | 42.55 | 47.97 |
| 500 | 17.644 | 46.02 | 46.01 | 42.74 | 48.24 |
| 600 | 21.162 | 46.13 | 46.13 | 42.77 | 48.23 |

### 3.3 Combustion：组分摩尔分数

| Epoch | 时间 (h) | CH4 | CO | CO2 | H2O | NH2 | NH3 | OH |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 3.542 | 47.02 | 39.08 | 43.80 | 46.55 | 40.43 | 47.02 | 38.05 |
| 200 | 7.094 | 47.79 | 39.71 | 44.45 | 47.27 | 41.15 | 47.78 | 38.69 |
| 300 | 10.612 | 48.18 | 39.99 | 44.75 | 47.58 | 41.51 | 48.17 | 39.00 |
| 400 | 14.121 | 48.36 | 40.15 | 44.93 | 47.82 | 41.68 | 48.42 | 39.16 |
| 500 | 17.644 | 48.64 | 40.28 | 45.13 | 48.07 | 41.92 | 48.68 | 39.34 |
| 600 | 21.162 | 48.62 | 40.35 | 45.15 | 48.06 | 41.96 | 48.67 | 39.37 |

其中表头的简写与日志属性名对应如下：CH4=`Mole_Fraction_of_CH4`、CO=`Mole_Fraction_of_CO`、CO2=`Mole_Fraction_of_CO2`、H2O=`Mole_Fraction_of_H2O`、NH2=`Mole_Fraction_of_NH2`、NH3=`Mole_Fraction_of_NH3`、OH=`Mole_Fraction_of_OH`。

### 3.4 Combustion：速度属性

| Epoch | 时间 (h) | Velocity | Velocity_Magnitude |
|---:|---:|---:|---:|
| 100 | 3.542 | 34.06 | 33.21 |
| 200 | 7.094 | 34.60 | 33.82 |
| 300 | 10.612 | 34.88 | 34.11 |
| 400 | 14.121 | 35.04 | 34.29 |
| 500 | 17.644 | 35.17 | 34.45 |
| 600 | 21.162 | 35.25 | 34.54 |

## 4. 时间开销

| Epoch | Ionization 累计时间 (h) | Combustion 累计时间 (h) |
|---:|---:|---:|
| 100 | 2.622 | 3.542 |
| 200 | 5.263 | 7.094 |
| 300 | 7.896 | 10.612 |
| 400 | 10.551 | 14.121 |
| 500 | 13.193 | 17.644 |
| 600 | 15.831 | 21.162 |

- Ionization 相邻 100 epoch 平均耗时约 **2.642 h**；Combustion 约 **3.524 h**。Combustion 每 100 epoch 约慢 **33.4%**。
- 到 epoch 600，日志累计时间分别为 **15.831 h** 和 **21.162 h**，Combustion 多用约 **5.331 h**。
- 每次 Ionization PSNR 计算约 421 s，六次合计约 0.702 h，占其日志累计时间约 **4.4%**；Combustion 每次仅约 5.5–6.3 s，六次合计约 0.010 h，占比低于 **0.1%**。因此 Ionization 的 PSNR 记录频率若提高，会显著增加总用时。
- 完整任务日志中的 `Train total` 分别为 15.907 h（Ionization）和 21.244 h（Combustion）；它比图中最后一个 `time=` 多出的约 0.076 h 和 0.082 h，主要来自主训练计时器之外的 pretrain、启动与收尾开销。

## 5. Aggregate PSNR 随 epoch / 时间的变化

| Epoch | Ionization PSNR (dB) | 相对上次 | Combustion PSNR (dB) | 相对上次 |
|---:|---:|---:|---:|---:|
| 100 | 41.37 | — | 42.01 | — |
| 200 | 47.18 | +5.81 | 42.75 | +0.74 |
| 300 | 44.40 | -2.78 | 43.03 | +0.28 |
| 400 | 49.80 | +5.40 | 43.24 | +0.21 |
| 500 | 47.67 | -2.13 | 43.44 | +0.20 |
| 600 | 51.56 | +3.89 | 43.48 | +0.04 |

### Ionization

Ionization 从 epoch 100 到 600 的 aggregate PSNR 净增 **10.19 dB**，对应 13.209 h 的增量训练时间，端点平均收益约 **0.772 dB/h**。但过程并不单调：epoch 200→300 和 400→500 分别回落 2.78 dB 和 2.13 dB，随后又恢复并刷新最佳值。

由于各 checkpoint 使用相同的固定 PSNR 子集，这两次大幅回落不能归因于“每次抽到了不同样本”。回落在 GT、H2、H_plus、He 上高度同步，说明当时的模型状态确实在固定评估子集上退化；仅凭 100-epoch 间隔的日志，还不能进一步区分是优化振荡、动态多属性权重、路由变化或其他训练机制导致。

epoch 600 是所有 5 个属性各自的已记录最佳点。也就是说，虽然中间波动明显，但当前日志不支持在 epoch 500 提前停止：再训练约 2.638 h 后，aggregate PSNR 提升了 3.89 dB。

| 属性 | Epoch 100 | Epoch 600 | 净增益 | 最佳 PSNR / Epoch | 端点收益 (dB/h) | 回落区间数 |
|---|---:|---:|---:|---:|---:|---:|
| GT | 41.56 | 50.84 | +9.28 | 50.84 / 600 | 0.703 | 2 |
| H2 | 46.22 | 55.78 | +9.56 | 55.78 / 600 | 0.724 | 2 |
| H_plus | 37.82 | 49.39 | +11.57 | 49.39 / 600 | 0.876 | 2 |
| He | 37.45 | 48.87 | +11.42 | 48.87 / 600 | 0.865 | 2 |
| PD | 43.79 | 52.90 | +9.11 | 52.90 / 600 | 0.690 | 1 |

结论上，H2 的最终绝对质量最高（55.78 dB）；H_plus 和 He 的起点最低，但提升最快，分别获得 11.57 dB 和 11.42 dB。最终最弱的两个属性仍是 He（48.87 dB）和 H_plus（49.39 dB），多属性质量差距尚未完全消除。

### Combustion 40NH3 1

Combustion 从 epoch 100 到 600 的 aggregate PSNR 由 42.01 dB 增至 43.48 dB，净增 **1.47 dB**；对应 17.620 h 的增量训练时间，端点平均收益约 **0.083 dB/h**。曲线稳定但边际收益快速衰减：

- epoch 100→200：+0.74 dB，约 0.208 dB/h；
- epoch 200→300：+0.28 dB，约 0.080 dB/h；
- epoch 300→400：+0.21 dB，约 0.060 dB/h；
- epoch 400→500：+0.20 dB，约 0.057 dB/h；
- epoch 500→600：仅 +0.04 dB，约 0.011 dB/h。

因此，若关注 PSNR–training time 性价比，**epoch 500 是明显的实用停止点**：相对 epoch 600 节省约 3.518 h（约占最终累计时间的 16.6%），aggregate PSNR 仅低 0.04 dB。CH4、H2O、NH3 和 Temperature 在 epoch 500 已达到各自的日志最佳值，epoch 600 的 0.01–0.02 dB 回落也基本处在两位小数日志精度附近。

| 属性 | Epoch 100 | Epoch 600 | 净增益 | 最佳 PSNR / Epoch | 端点收益 (dB/h) | 回落区间数 |
|---|---:|---:|---:|---:|---:|---:|
| Absolute_Pressure | 44.50 | 46.13 | +1.63 | 46.13 / 600 | 0.093 | 0 |
| Chemistry_Heat_Release_Rate | 41.29 | 42.77 | +1.48 | 42.77 / 600 | 0.084 | 0 |
| Mole_Fraction_of_CH4 | 47.02 | 48.62 | +1.60 | 48.64 / 500 | 0.091 | 1 |
| Mole_Fraction_of_CO | 39.08 | 40.35 | +1.27 | 40.35 / 600 | 0.072 | 0 |
| Mole_Fraction_of_CO2 | 43.80 | 45.15 | +1.35 | 45.15 / 600 | 0.077 | 0 |
| Mole_Fraction_of_H2O | 46.55 | 48.06 | +1.51 | 48.07 / 500 | 0.086 | 1 |
| Mole_Fraction_of_NH2 | 40.43 | 41.96 | +1.53 | 41.96 / 600 | 0.087 | 0 |
| Mole_Fraction_of_NH3 | 47.02 | 48.67 | +1.65 | 48.68 / 500 | 0.094 | 1 |
| Mole_Fraction_of_OH | 38.05 | 39.37 | +1.32 | 39.37 / 600 | 0.075 | 0 |
| Pressure | 44.50 | 46.13 | +1.63 | 46.13 / 600 | 0.093 | 0 |
| Temperature | 46.67 | 48.23 | +1.56 | 48.24 / 500 | 0.089 | 1 |
| Velocity | 34.06 | 35.25 | +1.19 | 35.25 / 600 | 0.068 | 0 |
| Velocity_Magnitude | 33.21 | 34.54 | +1.33 | 34.54 / 600 | 0.075 | 0 |

最终质量最高的是 Mole_Fraction_of_NH3（48.67 dB）、Mole_Fraction_of_CH4（48.62 dB）、Temperature（48.23 dB）和 Mole_Fraction_of_H2O（48.06 dB）。Velocity_Magnitude（34.54 dB）与 Velocity（35.25 dB）明显低于其他属性，同时 Velocity 的 100→600 增益也是最小的 1.19 dB；若后续目标是缩小属性间差距，这两个速度相关属性应优先检查。

## 6. 结论与建议

1. **Ionization 应保留更密集 checkpoint。** 现有 100-epoch PSNR 间隔只能看到大幅振荡的端点，无法知道区间内部峰值。建议至少每 25–50 epoch 保存一次 checkpoint；若保持 10% PSNR 评估，每次约增加 421 s，需权衡评估频率与训练总时长。
2. **Combustion 的 500→600 性价比很低。** 在当前设置下，epoch 500 可作为默认效率型停止点；只有追求最后约 0.04 dB aggregate 或少数仍缓慢上升的属性时才值得继续到 600。
3. **比较模型时应统一时间口径。** 本文横轴包含周期性 PSNR 评估，且 Ionization 的评估开销显著。若要做严格的模型训练吞吐对比，建议另行使用 timing 日志中的 `training + data + transfer`，或者关闭在线 PSNR 后比较纯训练时间。
4. **不要把本报告数值当作全量评估。** 当前曲线来自固定 10% 子集；最终论文/主表应继续使用独立的全量评估结果。

## 7. 复现绘图

脚本：`scripts/evaluation/plot_var_expert_main_psnr_time.py`

在仓库根目录运行：

```bash
python scripts/evaluation/plot_var_expert_main_psnr_time.py
```

默认输出：

- `Metric_Fig_Result/Fig/Main/VarExpert-Training-Curves/ionization_psnr_training_time.png`
- `Metric_Fig_Result/Fig/Main/VarExpert-Training-Curves/combustion_40NH3_1_psnr_training_time.png`

脚本也支持 `--ionization-log`、`--combustion-log`、`--output-dir` 和 `--dpi`，可直接替换后续新日志。
