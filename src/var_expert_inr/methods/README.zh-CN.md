# 自包含方法

[English](README.md) | [简体中文](README.zh-CN.md)

本目录中的包负责方法专属的部分生命周期，例如配置解析、数据访问、训练、推理 bundle
或预测。因此，即使包中包含 PyTorch 模型类，这些方法也不属于共享的
`var_expert_inr.models` registry。

## 入口与正式实验覆盖范围

下表的数据集表示 `configs/main/` 中当前存在的正式配置，不表示增加新配置后方法可能
支持的全部数据集。

| 方法 | 模块 | 命令 | 正式数据集 | 生命周期说明 |
| --- | --- | --- | --- | --- |
| APMGSRN | `apmgsrn` | `train`、`evaluate` | Combustion、Ionization | 拟合标量时序场，并为运行写入一个推理 bundle。 |
| ECNR | `ecnr` | `train`、`predict`、`evaluate` | Combustion、Ionization | 负责 packed 多尺度训练与紧凑推理 checkpoint。 |
| fV-SRN | `fv_srn` | `train`、`predict`、`evaluate` | Combustion、Ionization | 负责时序特征网格训练与推理。 |
| MC-INR | `mc_inr` | `train`、`predict`、`evaluate` | Combustion、Ionization、Katrina、RedSea | 使用方法专属的配置、数据与 checkpoint 流程。 |
| MINER | `miner` | `train`、`predict`、`evaluate` | Combustion、Ionization | 按时间步训练标量场，并保存完整时序推理 bundle。 |
| NeuralExpert | `neural_expert` | `train`、`evaluate` | Combustion、Ionization、Katrina、RedSea | 支持体数据与网格生命周期，包括 manager-pretrain 配置。 |
| RMDSRN | `rmdsrn` | `train`、`predict`、`evaluate` | Combustion、Ionization | 输出重建均值和集成方差。 |

使用以下形式调用方法：

```bash
python -m var_expert_inr.methods.<module>.cli <command> [arguments]
```

示例：

```bash
python -m var_expert_inr.methods.apmgsrn.cli train \
  --config configs/main/APMGSRN/combustion_40NH3_1__Temperature.yaml

python -m var_expert_inr.methods.rmdsrn.cli predict \
  --config configs/main/RMDSRN/ionization__GT.yaml

python -m var_expert_inr.methods.neural_expert.cli evaluate \
  --run runs/<exp_id>/<timestamp> \
  --metrics psnr,memory
```

使用 `python -m var_expert_inr.methods.<module>.cli --help` 以及具体命令的
`--help` 查看方法专属覆盖参数。

## 统一 CLI 兼容入口

当配置中出现 ECNR 或 MINER 的模型名称时，`python -m var_expert_inr.cli` 会将其
分派给对应方法。这在保留专用运行器的同时兼容通用实验命令：

```bash
python -m var_expert_inr.cli train \
  --config configs/main/MINER/ionization__GT.yaml
```

本目录中的其他包必须使用各自的方法模块进行训练和预测。基于运行的评估是共享的：
每个 `evaluate --run ...` 入口都会转发至 `var_expert_inr.evaluation`，并写入相同的
报告结构。

## Checkpoint 与运行

正式训练会创建新的 `runs/<exp_id>/<timestamp>/`，并在 `checkpoints/` 下保存面向
推理的 checkpoint 或 bundle。各方法的 manifest 和中间指标可能不同，但已完成运行
都会适配到公共的运行评估器。不要假设正式 checkpoint 中包含优化器或训练进度状态。

只有在方法 CLI 明确声明时，才保留旧式的基于配置评估。推荐使用
`evaluate --run <run-dir>`，以获得自包含报告以及一致的目标/时间步选择。

## 归属与许可证

原生 MINER 子系统包含其上游许可证：[`miner/LICENSE.MINER`](miner/LICENSE.MINER)。
该文件不构成 VarExpert-INR 的项目级许可证。
