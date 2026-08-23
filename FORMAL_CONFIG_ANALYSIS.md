# 正式实验配置审计（12 个模型）

> 审计日期：2026-08-23  
> 审计范围：`configs/main/`、`configs/rd_curve/`、现有 exploration/ablation/sensitivity 配置与汇总日志。  
> 本文检查配置覆盖、向量通道、学习率、batch、每 epoch 采样量、模型大小及 exploration 依据，不评价最终模型性能。

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
- NeuralExpert、STSR-INR 和 MVNet 均正确处理三通道向量；本次为匹配 VarExpert 总参数预算调整了数据集级宽度，但保留专家数、层数、残差块数、频率和训练 recipe。
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
| **fV-SRN** | 两个规则数据集均 `1e-2` | batch 16,000；按 timestep 采样 | 24.000M vs 24.012M/epoch，差 0.05% | 近似一致 | 差异来自 2,001 个时间步不可整除 |
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
| **fV-SRN** | APMGSRN 正式 optimizer 对齐 | 采用 APMGSRN 主优化器参数：Adam `lr=1e-2, betas=(0.9,0.99), eps=1e-14`，学习率保持常数 | 用户指定 | 不采用 APMGSRN 专属的坐标变换辅助优化器 |
| **APMGSRN** | architecture、历史 RD smoke | Ionization 保留 balanced；Combustion 改用已有 Size082 小结构，避免改变逐 timestep 方法本身 | 中 | 用一个 Combustion 属性确认小结构逐 timestep 收敛 |
| **InstantVNR** | v5 optimizer stability | 正式 `lr=1e-3 + MSE` 与 v5 第一名一致；Ionization 的 `hash=2^11, hidden=105` 有直接 size-matched 依据 | 强/中 | Combustion 的 `hash=2^10, hidden=88` 需短程确认 |
| **MINER** | RD smoke | 没有与当前实现直接对应的完成结果；二维/三维参数不同 | 弱 | 分别对两个数据集做 lr 与 active-block-cap 小矩阵 |
| **ECNR** | v6 optimizer calibration | v6 尚无完整 profile summary，正式 `1e-3` 只能视为内部一致 | 弱 | 完成 v6 后再冻结 optimizer |
| **STSR-INR** | Ionization RD smoke | 没有直接覆盖 `1e-5、batch=2048、1500 batches` 的跨数据集探索 | 弱 | 四数据集各做同 epoch-equivalent smoke，并对比 `5e-5` |
| **MVNet** | 未找到专属 exploration | 保留 10 个残差块和 `omega_0=30`，仅按数据集调整 hidden width；训练仍为 `1e-5、batch=2048、300 epochs` | 弱 | 四数据集各选代表任务做 size-profile smoke，并比较 `5e-6/1e-5/5e-5` |

## 5. 模型大小审计

### 5.1 统计口径

- 表中统一写作“参数量 / FP16 MiB”，其中 `FP16 MiB = 参数量 × 2 / 1024²`。参数量来自当前正式 YAML 实例化出的 `model.parameters()`；不包含 optimizer、梯度、checkpoint metadata 和非参数 buffer。
- 单属性方法按一个数据集全部正式属性模型求和；Katrina `v` 和 Combustion `Velocity` 各自仍是一个模型，但输出宽度为 3。联合方法按一个多输出模型计数。
- NeuralExpert 的 reconstruction 与 manager-pretrain YAML 架构相同。表中只汇总 reconstruction 模型，且其参数量已经包含 manager；不把配套 manager-pretrain 配置重复相加。
- APMGSRN 每个时间步保存独立子模型，因此同时考虑单 timestep 大小和完整时间序列总量；表中采用当前可复现的 PyTorch fallback 逻辑参数量，tiny-cuda-nn 后端的 packed checkpoint 需按实际文件另报。fV-SRN 此处统计训练态参数，最终 uint8 grid compact artifact 大小需训练完成后从 artifact 读取。
- MINER 的细尺度 active blocks 与 ECNR 的有效残差块、聚类、剪枝和量化均依赖数据及训练结果；两者只能从配置给出静态下界和“所有候选块均保留”的上界，不能用 checkpoint 文件大小替代 FP16 参数口径。

### 5.2 Main：完整数据集表示大小

| 模型 | RedSea | Katrina | Ionization | Combustion | 判定 |
| --- | ---: | ---: | ---: | ---: | --- |
| **VarExpert** | 108,138 / 0.206 | 112,501 / 0.215 | 856,259 / 1.633 | 1,137,357 / 2.169 | 四数据集的正式对齐目标 |
| **CoordNet** | 109,736 / 0.209 | 111,688 / 0.213 | 856,390 / 1.633 | 1,137,828 / 2.170 | 最大偏差 +1.48%；通过 `init_features/num_res` 对齐 |
| **SIREN** | 106,852 / 0.204 | 112,669 / 0.215 | 853,205 / 1.627 | 1,131,808 / 2.159 | 最大绝对偏差 1.19%；保持 node 5 层、volume 3 层 |
| **NeuralExpert** | 108,140 / 0.206 | 112,508 / 0.215 | 856,260 / 1.633 | 1,137,356 / 2.169 | 最大偏差 7 参数；manager-pretrain 不重复计数 |
| **MoE-INR** | 110,936 / 0.212 | 112,756 / 0.215 | 850,150 / 1.622 | 1,159,189 / 2.211 | 最大偏差 +2.59%；保持 7 experts 且 policy width 与 base width 相同 |
| **fV-SRN** | 跳过 | 跳过 | 855,525 / 1.632 | 1,141,068 / 2.176 | 最大偏差 +0.33%；训练态 grid 参数对齐，compact artifact 另计 |
| **APMGSRN** | 跳过 | 跳过 | 848,000 / 1.617 | 21,298,644 / 40.624 | 结构性例外；Ionization 差 −0.96%，Combustion 使用已有 Size082 小结构 |
| **InstantVNR** | 跳过 | 跳过 | 854,860 / 1.631 | 1,133,856 / 2.163 | 最大绝对偏差 0.31%；同时缩放 hash table 与 decoder width |
| **MINER** | 跳过 | 跳过 | 133,610,000–5,629,050,000 / 254.841–10,736.561 | 259,353,612–622,895,292 / 494.678–1,188.078 | 动态 active-block 范围；精确值必须读取训练后汇总 |
| **ECNR** | 跳过 | 跳过 | 1,039,035–48,516,495 / 1.982–92.538 | 5,767,536–110,433,684 / 11.001–210.636 | 范围包含必需 coarse MLP 与 CNN；量化 artifact 需训练后统计 |
| **STSR-INR** | 108,143 / 0.206 | 112,495 / 0.215 | 856,264 / 1.633 | 1,137,355 / 2.169 | 最大偏差 5 参数；保持 5 个 residual blocks 和各属性独立 head |
| **MVNet** | 108,701 / 0.207 | 111,895 / 0.213 | 854,905 / 1.631 | 1,132,875 / 2.161 | 最大绝对偏差 0.54%；保持 10 个 residual blocks，仅调整 hidden width |

MINER 每个 active block 的参数量为 `2h² + (d+4)h + 1`，其中 `h` 为该尺度 hidden width、`d` 为二维或三维坐标维度；最粗尺度使用 `4h`。表中下界保留必训的最粗尺度，上界假定其余尺度所有块均 active。其静态下界已经远高于 VarExpert，若强制对齐必须引入时间共享，会实质改变方法，因此与 APMGSRN 一并作为结构性例外。ECNR 的上下界均为未剪枝参数量；最终 8-bit MLP/CNN、latent 和索引 metadata 应以 `.ecnr` artifact 实际字节数单独报告。

除 APMGSRN、MINER 和训练后大小才确定的 ECNR 外，其余可静态计算的正式 main 模型均已对齐到对应数据集的 VarExpert 参数预算，最大绝对偏差为 2.59%。规则体标量方法在 Combustion 上只覆盖 12 个标量属性，但仍使用完整 VarExpert 联合模型大小作为总预算。

主要 size-matched profile 如下；未列出的层数、专家数、残差块数、频率和训练参数保持不变。

| 模型 | RedSea | Katrina | Ionization | Combustion |
| --- | --- | --- | --- | --- |
| CoordNet | `init=10,res=7` | `init=9,res=7` | `init=29,res=5` | `init=18,res=7` |
| SIREN | `hidden=72,layers=5` | `hidden=66,layers=5` | `hidden=237,layers=3` | `hidden=169,layers=3` |
| NeuralExpert | `decoder=8,enc=76,manager=16/15` | `decoder=7,enc=69,manager=14/17` | `decoder=31,enc=159,manager=62/71` | `decoder=16,enc=126,manager=32/56` |
| MoE-INR | `base=policy=18` | `base=policy=16` | `base=policy=46` | `base=policy=33` |
| fV-SRN | 跳过 | 跳过 | `grid=6³×64` | `grid=5³×60` |
| APMGSRN | 跳过 | 跳过 | `4³, features=14, nodes=16, layers=3` | `7³, features=1, nodes=16, layers=2` |
| InstantVNR | 跳过 | 跳过 | `hash=2¹¹, hidden=105` | `hash=2¹⁰, hidden=88` |
| STSR-INR | `init=12,embedding=655` | `init=12,embedding=36` | `init=33,embedding=635` | `init=28,embedding=120` |
| MVNet | `hidden=73` | `hidden=74` | `hidden=206` | `hidden=237` |

### 5.3 RD：Ionization 五属性汇总与标称档位

括号内为相对标称档位的偏差。CoordNet、MoE-INR 和 fV-SRN 是五个单属性模型之和；VarExpert 与 STSR-INR 是一个联合模型。

| 模型 | Size082（0.82 MiB） | Size163（1.63 MiB） | Size326（3.26 MiB） | Size652（6.52 MiB） | 判定 |
| --- | ---: | ---: | ---: | ---: | --- |
| **VarExpert** | 0.808（−1.4%） | 1.658（+1.7%） | 3.191（−2.1%） | 6.565（+0.7%） | 四档匹配 |
| **CoordNet** | 0.791（−3.5%） | 1.542（−5.4%） | 3.343（+2.5%） | 6.414（−1.6%） | 四档基本匹配 |
| **MoE-INR** | 0.799（−2.6%） | 1.553（−4.7%） | 3.296（+1.1%） | 6.488（−0.5%） | 四档基本匹配 |
| **fV-SRN** | 0.819（−0.1%） | **60.035（+3583.1%）** | 3.270（+0.3%） | 6.509（−0.2%） | Size163 的 `32³ × 16 × 12` feature grids 明显误配；其他三档匹配 |
| **MINER** | 0.711–45.267 | **254.841–10,736.561** | 2.520–130.617 | 5.426–257.166 | 精确值动态；Size163 标称值甚至低于静态下界，确定不匹配 |
| **STSR-INR** | 0.838（+2.2%） | **5.995（+267.8%）** | 3.331（+2.2%） | 6.516（−0.1%） | Size163 明显误配且档位非单调；其他三档匹配 |

SIREN、NeuralExpert、APMGSRN、InstantVNR、ECNR 和 MVNet 当前没有正式 RD 配置，因而不出现在上表。现有 RD 矩阵不能视为全部模型都已 size-matched：至少应重新校准 fV-SRN Size163、STSR-INR Size163 和 MINER Size163；MINER 其他档位也需要以完成训练后的 active-block 参数汇总才能最终验收。

### 5.4 大小结论

- main 已按数据集分别对齐 VarExpert：RedSea 目标 108,138、Katrina 112,501、Ionization 856,259、Combustion 1,137,357 参数。CoordNet、SIREN、NeuralExpert、MoE-INR、fV-SRN、InstantVNR、STSR-INR 和 MVNet 均落在 ±3% 内。
- APMGSRN 不改变逐 timestep 子模型机制。Ionization 保留 1,696 参数/timestep 的现有结构；Combustion 使用已有 Size082 结构，降为 887 参数/timestep，完整标量集合从 40,724,352 降至 21,298,644 参数，但仍作为结构性例外。
- MINER 的 mandatory coarse/active-block 表示无法通过普通 width 调整降至 VarExpert 预算；ECNR 的最终有效模型大小依赖聚类、剪枝和量化。这两者必须报告训练后实际值，不能伪装成静态精确对齐。
- fV-SRN 的 `Size163` 是最明确的静态配置错误：它与 main 共用 `grid_resolution=32, grid_channels=16`，训练态五属性总量约 60.04 MiB。
- STSR-INR `Size163` 复用了 main 结构，实际约 6.00 MiB，并且大于其 `Size326` 的 3.33 MiB，破坏 RD 档位单调性。
- 本次只调整 main 的模型容量；RD、训练学习率、batch、采样预算和训练周期保持各自既有口径。

## 6. 最终判定

正式配置矩阵现在满足以下条件：支持非结构网格的模型覆盖 RedSea 与 Katrina；规则体模型覆盖 Ionization 与 Combustion 的兼容属性；NeuralExpert、STSR-INR、MVNet 正确处理三通道向量；可静态调节的 main 模型已按数据集对齐 VarExpert 参数预算；CoordNet 正式学习率统一为 `1e-5`；STSR-INR 四数据集训练预算一致；fV-SRN 正式 checkpoint 周期统一为 300 epochs。

除 APMGSRN、MINER、ECNR 的结构性/动态例外外，main 已满足“每个数据集的多变量总参数量与 VarExpert 对齐”，可进入短程稳定性 exploration。当前 RD 矩阵仍不能用于严格的等大小比较：fV-SRN、STSR-INR 和 MINER 的 Size163 需要另行校准。论文表格应显式报告 APMGSRN/MINER 的时间子模型汇总，以及 fV-SRN/ECNR 的最终 compact artifact 大小。

## 7. 主要审计依据

- 正式配置生成器：`scripts/main/generate_configs.py`
- 正式 main 配置：`configs/main/`
- 正式 RD 配置：`configs/rd_curve/`
- 统一模型参数统计：`src/var_expert_inr/utils/model_stats.py`
- 方法专用大小统计：NeuralExpert `estimate_model_size_fp16`、MINER timestep/aggregate metrics、fV-SRN 与 ECNR compact artifact summaries
- exploration/ablation/sensitivity：`configs/exploration/`、`configs/ablation/`、`configs/sensitivity/`
- v5 汇总：`batch_logs/exploration_v5/20260822_052849/profile_summary.tsv`
- CoordNet 低学习率汇总：`batch_logs/exploration_CoordNet/20260820_155124/profile_summary.tsv`
- v3 attention：`batch_logs/exploration_v3/20260810_070740/needs_attention.txt`
- v6 状态：`batch_logs/exploration_v6/20260822_083152/status.tsv`
