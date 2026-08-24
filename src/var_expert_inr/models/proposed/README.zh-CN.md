# 提出的方法

[English](README.md) | [简体中文](README.zh-CN.md)

本包包含 VarExpert 及相关多目标 INR 实现。代码文件或包导出存在，并不等同于已经注册
到公开实验 CLI。

## 可用性

| 模型 | Python 导出 | Registry 配置名称 | 统一 CLI |
| --- | --- | --- | --- |
| VarExpert | `VarExpert` | `var_expert` | 是 |
| 共享编码器 INR | `SharedEncINR` | `shared_enc_inr` | 是 |
| 变量无关 MoE | `VariableAgnosticMoE` | 未注册 | 否 |

VarExpert 与 SharedEncINR 使用公共的数据集、训练、checkpoint、预测和评估栈：

```bash
python -m var_expert_inr.cli train \
  --config configs/main/VarExpert/ionization.yaml
```

`VariableAgnosticMoE` 和 `build_variable_agnostic_moe_from_config` 被导出用于直接
Python 调用和开发，但 `var_expert_inr.models.registry` 不接受它们作为
`model.name`。仅向本包添加类，不会自动使其成为可运行的实验模型。

## 多目标契约

已注册的提出模型会为数据集中的每个目标构建一个输出 head，并以目标名称为键返回预测。
目标维度和顺序来自数据集元数据，由公共有效配置和 checkpoint 流程保留。

提出模型共用的可复用层位于 `components.py`。实验架构和训练选项应写入 `configs/`；
相关规则见[配置指南](../../../../configs/README.zh-CN.md)。
