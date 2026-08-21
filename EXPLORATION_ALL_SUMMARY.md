# Exploration 系列实验总结

生成日期：2026-08-19

## 1. 汇总范围与判定口径

本报告覆盖 `exploration`、`explorationv2`、`explorationv3`、`explorationv4` 共 470 项配置。逐项配置、实验目的、PSNR 轨迹、首/峰/末探针、训练状态、异常原因和数据来源见 `EXPLORATION_ALL_RESULTS.csv`。

| 版本 | 配置数 | 重建项 | 管理器预训练项 | 成功 | 失败 | 重建项需关注 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| exploration | 126 | 111 | 15 | 123 | 3 | 33 |
| explorationv2 | 53 | 38 | 15 | 53 | 0 | 6 |
| explorationv3 | 210 | 185 | 25 | 210 | 0 | 57 |
| explorationv4 | 81 | 81 | 0 | 81 | 0 | 45 |

统一的“需关注”判定为：最终 PSNR 比峰值回落超过 1 dB，或最终 PSNR 相对首个探针提升不足 0.1 dB；训练失败或探针缺失也计入需关注。v3 直接采用官方汇总的严格校验结果；v1/v2 是按同一阈值回溯计算；v4 因尚未生成汇总 TSV，依据 81 份完整日志重建。

NeuralExpert 的 manager-pretraining 配置不产生重建 PSNR，因此不计入“重建项需关注”。原始 v3 `needs_attention.txt` 会因缺探针列出这 25 项，本报告将它们归为“成功完成、重建指标不适用”。

表中的“平均最终 PSNR”是同一 profile 对所覆盖目标的非加权算术平均。它适合在同一方法内部比较结构或 Size，不应把单目标方法的均值与 MC-INR/VarExpert 的多变量 aggregate PSNR 直接横向排名。

## 2. exploration：Size163 结构搜索

### 实验目的

第一轮在 Size163 预算和统一 50 epoch-equivalent 探针下，系统检查网络深度、专家数和网格/解码器预算分配等结构变量。主要目标不是产生正式 RD 曲线，而是快速找出每个方法族的可用结构与明显故障。

### 各实验组结果

| 方法族 | 比较变量 | 最优 profile | 平均最终 PSNR (dB) | 结论 |
| --- | --- | --- | ---: | --- |
| SIREN | depth2 / depth3 / depth5 | depth3 | 37.829 | depth3 略优于 depth5（37.669）和 depth2（36.922），三者均稳定。 |
| CoordNet | res5 / res10 / res15 | res5 | 39.821 | res5 明显最佳且 5/5 稳定；res10、res15 分别有 4/5、3/5 项回落或无增益，深度增加反而恶化。 |
| MoE-INR | 4 / 7 / 10 experts | experts10 | 37.220 | 专家数增加带来小幅均值收益；experts10 无异常，experts4/7 各有 1 项需关注。 |
| VarExpert | 4 / 6 / 8 experts，固定 top-3 | experts8 | 43.936 | experts8 略高于 experts4（43.802），experts6 较低（42.465）；专家数收益并非单调。 |
| MC-INR | depth3_4 / depth5_6 / depth7_8 | 无 | — | 三项全部失败，无法判断结构优劣；随后在 v2 修复目标布局后重跑。 |
| NeuralExpert | depth1 / depth2 / depth3 | depth1 | 37.898 | 三种深度都稳定；depth1 最高，继续加深没有收益。15 个 manager-pretraining 均成功。 |
| APMGSRN | decoder-heavy / balanced / grid-heavy | balanced | 27.156 | balanced 最好，grid-heavy 次之（25.815），decoder-heavy 最低（23.998）。 |
| fV-SRN | decoder-heavy / balanced / grid-heavy | grid-heavy | 27.295 | grid-heavy 明显最好且稳定；balanced 5/5 需关注，decoder-heavy 2/5 需关注。 |
| RMDSRN | decoder-heavy / balanced / grid-heavy | balanced | 28.256 | balanced 均值最高，但 4/5 需关注；另外两组 5/5 需关注，暴露出训练/调度稳定性问题。 |

### v1 结论

- 较浅或较平衡的结构普遍更可靠：CoordNet res5、NeuralExpert depth1、SIREN depth3 分别胜出。
- VarExpert experts8/top-3 是本轮最好的多变量配置，最终 aggregate PSNR 为 43.936 dB。
- MC-INR 的三项失败属于实现问题，而不是可靠的模型能力结论。
- RMDSRN 和 fV-SRN 对训练过程更敏感，只看最终 PSNR 会掩盖中途峰值回落。

## 3. explorationv2：修复验证与定向超参数搜索

### 实验目的

v2 保留 v1，不覆盖旧结果，并围绕三个问题开展复验：修复 MC-INR 目标布局；把 NeuralExpert 提高到 Size326；围绕 v1 的 VarExpert experts8/top-3 对照扫描 9/10 专家和全部 top-k。

### 各实验组结果

| 实验组 | 主要结果 | 结论 |
| --- | --- | --- |
| MC-INR 修复后重跑 | depth3_4=38.166、depth5_6=38.036、depth7_8=31.198 dB | 3/3 成功，证明 v1 的失败主要来自目标布局问题。depth3_4 最终值最高，但有回落标记；depth5_6 略低但更稳定，是更稳妥的候选。 |
| NeuralExpert Size326 | depth1=39.483、depth2=38.908、depth3=39.265 dB | 15 个重建项和 15 个管理器项全部成功；depth1 再次最好，浅层结论跨预算保持一致。 |
| VarExpert experts/top-k | 最佳 experts9_top4=44.329 dB；其次 experts9_top5=44.286、experts10_top7=44.261 | 相对 experts8_top3 对照（43.936），最佳提升约 0.392 dB。top-k 不是越大越好，且若干组合存在明显回落，单次 seed 的最优点应在正式实验中复验。 |

### v2 结论

- v1 的 MC-INR 阻塞问题得到解决，全部配置均已完成。
- VarExpert 的最佳点落在 9 专家/top-4，而不是最大专家数或最稠密路由；路由稀疏度比单纯增加专家更重要。
- NeuralExpert 的浅层优势重复出现，说明 depth1 是后续 Size 矩阵的合理基线。

## 4. explorationv3：全正式 Size 配置烟雾测试

### 实验目的

v3 将正式 RD-curve 清单中的 210 个配置原样复制到隔离目录，只把训练长度统一缩短到 50 epoch-equivalent，并每 5 个等效 epoch 做固定样本探针。目标是覆盖五个 Size 档位和九个方法族，检查正式配置是否能训练、是否随容量改善，以及是否发生塌陷。

### 方法族结果

下表依次列出 Size082 / Size163 / Size326 / Size652 / Size1304 的平均最终 PSNR；括号为该 Size 下“需关注项/重建项”。

| 方法族 | 五个 Size 的平均最终 PSNR (dB) | 最佳 Size | 主要判断 |
| --- | --- | --- | --- |
| APMGSRN | 25.253(0/5), 27.587(0/5), 29.083(0/5), 29.290(0/5), 29.211(0/5) | Size652 | 稳定扩展到 Size652，Size1304 已基本饱和。 |
| CoordNet | 36.447(1/5), 35.787(4/5), 29.203(4/5), 20.780(5/5), 15.436(5/5) | Size082 | 容量越大反而越差，Size652/1304 全面塌陷，是 v3 最明确的系统性故障。 |
| fV-SRN | 27.656(1/5), 27.395(1/5), 28.660(2/5), 28.732(2/5), 30.338(2/5) | Size1304 | 总体随容量改善，但轨迹噪声和峰值回落较多。 |
| MC-INR | 38.648(0/1), 38.166(1/1), 39.918(0/1), 39.587(0/1), 38.612(0/1) | Size326 | Size326 最佳，继续增大没有收益；仅 Size163 被标记回落。 |
| MoE-INR | 35.983(0/5), 36.862(1/5), 37.119(1/5), 38.229(0/5), 37.249(1/5) | Size652 | 到 Size652 基本正向扩展，Size1304 略回落。 |
| NeuralExpert | 35.574(0/5), 37.898(0/5), 39.483(0/5), 41.742(0/5), 43.357(0/5) | Size1304 | 五档单调提升且 25 个重建项全部稳定，是最清晰的容量扩展曲线。 |
| RMDSRN | 28.699(5/5), 28.700(5/5), 29.418(4/5), 30.209(5/5), 29.700(5/5) | Size652 | 24/25 项需关注，旧调度在短程训练中严重不匹配；这直接触发 v4 调度修正。 |
| SIREN | 36.799(0/5), 37.829(0/5), 38.340(0/5), 38.553(0/5), 35.352(2/5) | Size652 | 到 Size652 稳定提升，Size1304 出现退化和两项塌陷。 |
| VarExpert | 41.385(0/1), 42.833(0/1), 43.119(0/1), 42.793(0/1), 42.450(0/1) | Size326 | Size326 达峰，后续容量没有转化为 50-epoch 短程收益，但五档均通过稳定性检查。 |

v3 的 57 个重建异常项中，33 项同时满足“峰值回落”和“无有效增益”，16 项仅峰值回落，8 项仅无有效增益。按方法族计：RMDSRN 24、CoordNet 19、fV-SRN 8、MoE-INR 3、SIREN 2、MC-INR 1；APMGSRN、NeuralExpert、VarExpert 为 0。

### v3 结论

- “更大模型必然更好”不成立。NeuralExpert 能稳定扩展，CoordNet 则随着 Size 增大系统性失稳。
- Size326–Size652 已是多个方法的短程收益甜点：MC-INR、VarExpert 在 Size326 达峰，APMGSRN、MoE-INR、SIREN、RMDSRN 在 Size652 达峰。
- v3 是短程烟雾测试，不是完整训练后的最终 RD 排名。对于慢收敛方法，Size1304 的短程劣势可能包含训练预算不足；但大幅回落仍是必须修复的优化问题。

## 5. explorationv4：CoordNet 稳定性与 RMDSRN 调度/损失消融

### 5.1 CoordNet：等参数深度扫描

实验在 Size326、Size652、Size1304 的 GT/H_plus/He 上，用整数宽度匹配正式 res10 参数量，比较 res2/res3/res5/res7/res10，并保留正式学习率。

| Size | res2 | res3 | res5 | res7 | res10 | res2 相对 res10 的匹配中位提升 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Size326 | 37.748 (0/3) | 36.313 (1/3) | 28.894 (2/3) | 29.496 (2/3) | 23.774 (3/3) | +17.405 dB |
| Size652 | 33.071 (2/3) | 28.742 (2/3) | 21.721 (3/3) | 19.649 (2/3) | 15.106 (3/3) | +22.360 dB |
| Size1304 | 26.090 (2/3) | 21.955 (3/3) | 13.527 (3/3) | 7.467 (3/3) | 7.465 (3/3) | +29.881 dB |

括号仍为“需关注项/重建项”。浅层结构在三个 Size 上都显著优于 res10，且差距随 Size 增大。这说明 v3 的 CoordNet 失败不是参数量不足，而是深网络在当前优化设置下的稳定性问题。

### 5.2 CoordNet：Size1304 因果控制

| profile | 平均最终 PSNR (dB) | 需关注 | 相对 res10_base_lr 的匹配中位提升 | 判断 |
| --- | ---: | ---: | ---: | --- |
| res10_base_lr | 7.465 | 3/3 | 0.000 | 基线全面塌陷。 |
| res10_clip | 7.858 | 3/3 | +1.118 | 单独梯度裁剪基本无效。 |
| res10_scaled_lr | 35.584 | 2/3 | +33.126 | 降低学习率大幅恢复最终质量，但 H_plus/He 仍从更高峰值回落。 |
| res5_scaled_lr_clip | 36.579 | 0/3 | +32.733 | 三个目标全部稳定，是 v4 最可靠的 Size1304 方案。 |

因此，学习率是主要致因，较浅深度和梯度裁剪共同提供额外稳定性。若要更新正式配置，优先候选是 `res5_scaled_lr_clip`，而不是只给 res10 加 clipping。

### 5.3 RMDSRN：900k 调度修正和 lambda 消融

RMDSRN 只训练 75k 步，但 LR 和 lambda 沿正式 900k 步调度的前 75k 步推进。`lambda_max=10` 覆盖五个 Size；`lambda_max=1/0` 仅在 Size082 和 Size1304 做对照。

| Size | lambda=0 | lambda=1 | lambda=10 | 判断 |
| --- | ---: | ---: | ---: | --- |
| Size082 | 33.796 (0/3) | 33.841 (0/3) | 32.164 (0/3) | lambda=1 略优，lambda=10 已有约 1.68 dB 均值损失。 |
| Size163 | — | — | 32.231 (1/3) | lambda=10 有 1 项异常。 |
| Size326 | — | — | 32.763 (1/3) | lambda=10 有 1 项异常。 |
| Size652 | — | — | 34.264 (2/3) | lambda=10 有 2 项异常。 |
| Size1304 | 37.040 (0/3) | 36.648 (0/3) | 33.764 (2/3) | lambda=0 最佳且稳定；lambda=10 比 lambda=0 低 3.276 dB。 |

调度修正后，lambda=0/1 的两个对照 Size 均 6/6 稳定；lambda=10 在五个 Size 中有 6/15 项需关注。结果表明，高权重方差正则显著牺牲重建，且影响在大模型上更明显。若目标首先是重建 PSNR，应优先使用 lambda=0 或 1；若必须保留不确定性建模，需要重新平衡 lambda，而不能直接沿用 10。

### v4 数据限制

v4 的 81 项状态均为成功，但本地批次目录缺少预期的 `exploration_summary.tsv` 和 `profile_summary.tsv`。本报告从每份日志的十个 `Exploration PSNR` 行恢复轨迹，并提取 RMDSRN 最终 step 的 total/member loss、variance KL、lambda 和 LR。设计文档提到的 variance-error Pearson correlation 与 top-1%/top-5% hit rate 没有打印在日志中，因此当前 CSV 不包含这些三类指标；如需分析不确定性质量，需要恢复远端 `runs/exploration_v4` 的 metrics 文件后重新运行 v4 汇总脚本。

## 6. 跨版本综合结论

1. **结构并非越深越好。** CoordNet res5/res2、NeuralExpert depth1、SIREN depth3 都优于更深候选。尤其 CoordNet 的深度惩罚会随 Size 放大。
2. **容量扩展依赖优化配置。** NeuralExpert 呈稳定单调扩展；CoordNet 在原学习率下呈反向扩展。v4 证明降低学习率可恢复大部分质量，浅层加 clipping 才能同时恢复稳定性。
3. **VarExpert 在紧凑预算下表现强，但 top-k 最优点非单调。** v2 的 experts9/top-4 最好，只比 experts8/top-3 高约 0.39 dB，建议用多 seed 确认差异是否稳健。
4. **RMDSRN 的主要问题是调度和不确定性权重。** v3 的旧配置几乎全面触发异常；v4 修正调度后，lambda=0/1 稳定，而 lambda=10 明显损害重建。
5. **短程 exploration 适合筛错和筛结构，不替代正式训练。** 最终正式配置应至少对候选点做多 seed、完整训练预算和统一解码评估；尤其不应依据单次 50-epoch 探针中小于约 0.5 dB 的差异直接定案。

## 7. CSV 字段说明

- `version/group/family/stage/size_label/profile/target`：实验身份和分组。
- `experiment_purpose`：该配置所属实验组的目的。
- `initial_psnr_db/peak_psnr_db/final_psnr_db`：首个、最高和最后一个固定探针结果。
- `gain_from_initial_db/drop_from_peak_db`：短程收益与训练回落。
- `needs_attention/attention_reason/validation_basis`：异常标记、原因和判定来源。
- `trajectory`：5、10、…、50 epoch-equivalent 的完整 PSNR 轨迹。
- `data_source/metrics_or_log_path`：每项结果的本地证据位置。
- `rmdsrn_final_*`：v4 RMDSRN 最后一步日志中的训练损失与调度状态。
- `result_summary`：每项配置的一句话中文结果。

## 8. 主要本地来源

- `scripts/generate_size_exploration_configs.py`、`scripts/summarize_size_exploration.py`
- `EXPLORATION_V2.md`、`scripts/generate_exploration_v2_configs.py`
- `EXPLORATION_V3.md`、`scripts/generate_exploration_v3_configs.py`、`scripts/summarize_exploration_v3.py`
- `EXPLORATION_V4.md`、`scripts/generate_exploration_v4_configs.py`、`scripts/summarize_exploration_v4.py`
- `batch_logs/exploration*/.../status.tsv`、`exploration_summary.tsv`、`needs_attention.txt` 与 v4 的 81 份日志
