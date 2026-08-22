# 正式实验配置审计（12 个模型）

> 审计日期：2026-08-23  
> 审计范围：`configs/main/`、`configs/rd_curve/`、现有 exploration/ablation/sensitivity 配置与汇总日志。  
> 本文检查配置覆盖、向量通道、学习率、batch、每 epoch 采样量及 exploration 依据，不评价最终模型性能。

## 1. 数据集口径

| 数据集 | 类型 | 属性 | 属性数 | 实际输出通道数 |
| --- | --- | --- | ---: | ---: |
| RedSea | 非结构网格节点 | `SALT, TEMP, U, V` | 4 | 4 |
| Katrina | 非结构网格节点 | `fort63, fort64, fort73, speed, v` | 5 | 7（`v` 为三分量） |
| Ionization | 规则时变体数据 | `GT, H_plus, H2, He, PD` | 5 | 5 |
| Combustion `40NH3_1` | 规则时变二维场 | 12 个标量属性和 `Velocity` | 13 | 15（`Velocity` 为三分量） |

覆盖率按实际参与训练的输出属性计算。联合模型的输出宽度必须按通道数计算，不能简单等于属性数。

规则体专用模型可以跳过 RedSea 和 Katrina。当前实现中，fV-SRN、APMGSRN、MINER、ECNR 依赖规则网格、块或体插值；InstantVNR 的正式复现也是 volume-only。它们缺少非结构网格配置不计为覆盖缺失。

## 2. 正式配置覆盖

12 个模型共有 **321** 个正式配置，其中 main **233** 个、RD-curve **88** 个。全仓库 15 个模型共有 main **271** 个、RD-curve **88** 个，合计 **359** 个。

表中“单”表示每个属性独立训练，“联”表示一个联合多输出模型，“manager”表示 NeuralExpert 的配套 manager-pretraining 配置。

| 模型 | 配置数（main + RD） | 非结构网格 | RedSea 4 | Katrina 5 / 7 通道 | Ionization 5 | Combustion 13 / 15 通道 | 判定 |
| --- | ---: | --- | --- | --- | --- | --- | --- |
| **VarExpert (Ours)** | 5 + 4 = 9 | 支持 | 4/4（联） | 5/5，7 通道（联） | 5/5（两个 main；另有 4 RD） | 13/13，15 通道（联） | 完整 |
| **CoordNet** | 27 + 20 = 47 | 支持 | 4/4（单） | 5/5，向量输出 3 通道（单） | 5/5（单；另有 20 RD） | 13/13，向量输出 3 通道（单） | 完整 |
| **SIREN** | 27 + 0 = 27 | 支持 | 4/4（单） | 5/5，向量输出 3 通道（单） | 5/5（单） | 13/13，向量输出 3 通道（单） | 完整 |
| **NeuralExpert** | 54 + 0 = 54 | 支持 | 4 单 + 4 manager | 5 单 + 5 manager；`v` 为 3 通道 | 5 单 + 5 manager | 13 单 + 13 manager；`Velocity` 为 3 通道 | 完整 |
| **MoE-INR** | 27 + 20 = 47 | 支持 | 4/4（单） | 5/5，向量输出 3 通道（单） | 5/5（单；另有 20 RD） | 13/13，向量输出 3 通道（单） | 完整 |
| **fV-SRN** | 17 + 20 = 37 | 不支持 | 跳过 | 跳过 | 5/5（单；另有 20 RD） | 12/13（标量完整） | 兼容范围内完整 |
| **APMGSRN** | 17 + 0 = 17 | 不支持 | 跳过 | 跳过 | 5/5（单） | 12/13（标量完整） | 兼容范围内完整 |
| **InstantVNR** | 17 + 0 = 17 | volume-only | 跳过 | 跳过 | 5/5（单） | 12/13（标量完整） | 当前正式范围内完整 |
| **MINER** | 17 + 20 = 37 | 不支持 | 跳过 | 跳过 | 5/5（单；另有 20 RD） | 12/13（标量完整） | 兼容范围内完整 |
| **ECNR** | 17 + 0 = 17 | 不支持 | 跳过 | 跳过 | 5/5（单） | 12/13（标量完整） | 兼容范围内完整 |
| **STSR-INR** | 4 + 4 = 8 | 支持 | 4/4（联） | 5/5，7 通道（联） | 5/5（main；另有 4 RD） | 13/13，15 通道（联） | 完整 |
| **MVNet** | 4 + 0 = 4 | 支持 | 4/4（联，输出 4） | 5/5（联，输出 7） | 5/5（联，输出 5） | 13/13（联，输出 15） | 完整 |

### 覆盖结论

- 四个支持非结构网格的通用模型族和三个联合模型已覆盖其兼容范围内的全部属性。
- Katrina `v` 和 Combustion `Velocity` 均按三通道训练，不再被误当成标量。
- NeuralExpert 和 MVNet 只改变输出通道宽度；其隐藏层、专家结构、残差块和训练 recipe 没有变化。
- 规则体专用模型仍跳过两个非结构网格数据集及 Combustion 向量，这是实现能力边界，不是漏配置。

## 3. 学习率、batch 与采样量一致性

“每 epoch 采样量”指主训练阶段读取的物理样本数。gradient accumulation 只改变 optimizer update 的有效 batch，不改变物理采样量。

| 模型 | 正式学习率 | batch / 采样单位 | 每 epoch 或等价预算 | 跨数据集一致性 | 判断 |
| --- | --- | --- | --- | --- | --- |
| **VarExpert** | 四数据集均 `5e-5` | `16,000 × 1,500 batches` | 24.0M/epoch；600 epoch = 14.4B | 一致 | 合理；预训练也使用 `5e-5` |
| **CoordNet** | 四数据集及 RD 均 `1e-5` | `16,000 × 1,500` | 24.0M/epoch；600 epoch = 14.4B | 一致 | 已与低学习率稳定性探索对齐 |
| **SIREN** | 四数据集均 `5e-5` | `16,000 × 1,500` | 24.0M/epoch；600 epoch = 14.4B | 一致 | 合理 |
| **NeuralExpert** | 四数据集均 `3e-5` | 每 iteration 采样 16,000 点 | reconstruction 60,000 iterations = 960M；manager 30,000 = 480M | 一致 | 方法专用预算，明显低于 14.4B |
| **MoE-INR** | 四数据集均 `5e-5` | `16,000 × 1,500` | 24.0M/epoch；600 epoch = 14.4B | 一致 | 合理 |
| **fV-SRN** | 两个规则数据集均 `5e-3` | batch 16,000；按 timestep 采样 | 24.000M vs 24.012M/epoch，差 0.05% | 近似一致 | 差异来自 2,001 个时间步不可整除 |
| **APMGSRN** | 两个规则数据集均 `1e-2` | 16,000/iteration；逐 timestep | 总预算 14.400B vs 14.4072B，差 0.05% | 近似一致 | 无传统 epoch，预算可接受 |
| **InstantVNR** | 两个规则数据集均 `1e-3` | `16,000 × 1,500`，累积 4 个 batch | 24.0M/epoch；总计 14.4B | 一致 | 合理，且有直接 exploration 支持 |
| **MINER** | Ionization `1e-3`；Combustion `5e-4` | active-block cap | 实际量随 active blocks 动态变化 | 不一致（有意） | 二维/三维专用 recipe，可解释但缺少直接调参证据 |
| **ECNR** | 两个规则数据集均 `1e-3` | `3,200 × 3,000`/scale-epoch | 9.6M/scale-epoch；三尺度总预算约 14.4B | 一致 | 配置一致，v6 校准实验尚未完成 |
| **STSR-INR** | 四数据集均 `1e-5` | `2,048 × 1,500` | 3.072M/epoch；300 epoch = 921.6M | 一致 | 四数据集已统一，不再混用全量 sampler |
| **MVNet** | 四数据集均 `1e-5` | `2,048 × 1,500` | 3.072M/epoch；300 epoch = 921.6M | 一致 | 方法专用 protocol，预算低于 14.4B |

### 一致性结论

- 除 MINER 的二维/三维专用 recipe 外，各模型的学习率在其支持的数据集之间一致。
- fV-SRN 和 APMGSRN 的 0.05% 采样差异来自 Combustion 的 2,001 个时间步，属于离散预算误差。
- NeuralExpert、STSR-INR、MVNet 的总训练样本量显著低于通用 14.4B protocol；论文比较时应明确这是方法专用训练预算。

## 4. Exploration 与正式设置对应关系

证据等级：“强”表示有直接 optimizer/稳定性 sweep 且正式参数与优胜项一致；“中”表示有结构或短程探索但未完整覆盖正式 recipe；“弱”表示只有 smoke、未完成结果或没有对应探索。

| 模型 | 已有 exploration | 与正式设置的关系 | 证据 | 建议 |
| --- | --- | --- | --- | --- |
| **VarExpert** | architecture、专家数、Top-K、RD smoke | 正式 `experts6/top3` 与部分探索优胜结构不完全一致；lr 未独立 sweep | 中 | 对 `experts6/8/9` 做同预算多 seed 确认 |
| **CoordNet** | architecture、RD、depth/regularization、低 lr sweep | 低 lr 结果支持 `1e-5`；Size652 曾偏好 `5e-6` | 中偏强 | main 已统一为 `1e-5`；大 Size 可保留 `5e-6` 对照 |
| **SIREN** | architecture、历史 RD smoke | volume 的三层结构与 Size163 优胜项一致；node 结构缺跨数据集探索 | 中 | 对 RedSea/Katrina 各做一个短程属性 smoke |
| **NeuralExpert** | architecture、v2 depth、历史 RD smoke | 探索中 depth1 优于 depth2，正式仍采用原方法 depth2；向量输出未单独 sweep | 中 | 对两个三通道目标做短程稳定性检查 |
| **MoE-INR** | architecture、RD smoke | 10 experts 曾优于正式 7 experts；lr/batch 未独立 sweep | 中 | 记录采用 7 experts 的参数预算依据 |
| **fV-SRN** | architecture、v5 optimizer stability | `lr=5e-3, step=20, gamma=0.5` 与 v5 第一名一致 | 强 | 正式 optimizer 可直接采用 |
| **APMGSRN** | architecture、历史 RD smoke | 正式 balanced 结构与结构探索优胜项一致 | 中偏强 | 用一个 Combustion 属性确认逐 timestep 收敛 |
| **InstantVNR** | v5 optimizer stability | 正式 `lr=1e-3 + MSE` 与 v5 第一名一致 | 强 | 保留早期 checkpoint/probe 监测较大正式模型 |
| **MINER** | RD smoke | 没有与当前实现直接对应的完成结果；二维/三维参数不同 | 弱 | 分别对两个数据集做 lr 与 active-block-cap 小矩阵 |
| **ECNR** | v6 optimizer calibration | v6 尚无完整 profile summary，正式 `1e-3` 只能视为内部一致 | 弱 | 完成 v6 后再冻结 optimizer |
| **STSR-INR** | Ionization RD smoke | 没有直接覆盖 `1e-5、batch=2048、1500 batches` 的跨数据集探索 | 弱 | 四数据集各做同 epoch-equivalent smoke，并对比 `5e-5` |
| **MVNet** | 未找到专属 exploration | 正式固定 `1e-5、batch=2048、300 epochs` | 弱 | 四数据集各选代表任务比较 `5e-6/1e-5/5e-5` |

## 5. 最终判定

正式配置矩阵现在满足以下条件：支持非结构网格的模型覆盖 RedSea 与 Katrina；规则体模型覆盖 Ionization 与 Combustion 的兼容属性；NeuralExpert、STSR-INR、MVNet 正确处理三通道向量；CoordNet 正式学习率统一为 `1e-5`；STSR-INR 四数据集训练预算一致。

可以开始正式组实验，但建议优先补充 STSR-INR、MVNet、MINER 和尚未完成的 ECNR optimizer exploration，并在论文表格中显式报告 NeuralExpert、STSR-INR、MVNet 的方法专用采样预算。

## 6. 主要审计依据

- 正式配置生成器：`scripts/main/generate_configs.py`
- 正式 main 配置：`configs/main/`
- 正式 RD 配置：`configs/rd_curve/`
- exploration/ablation/sensitivity：`configs/exploration/`、`configs/ablation/`、`configs/sensitivity/`
- v5 汇总：`batch_logs/exploration_v5/20260822_052849/profile_summary.tsv`
- CoordNet 低学习率汇总：`batch_logs/exploration_CoordNet/20260820_155124/profile_summary.tsv`
- v3 attention：`batch_logs/exploration_v3/20260810_070740/needs_attention.txt`
- v6 状态：`batch_logs/exploration_v6/20260822_083152/status.tsv`
