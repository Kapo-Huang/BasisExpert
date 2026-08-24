# 共享引擎基线

[English](README.md) | [简体中文](README.zh-CN.md)

这些模型使用公共的配置、数据集、训练、checkpoint、预测与评估栈。它们的公开配置名称
注册在 `var_expert_inr.models.registry` 中，可以通过
`python -m var_expert_inr.cli` 调用。

## 已注册模型

| 配置名称 | 实现 | 输出模式 |
| --- | --- | --- |
| `siren` | `siren.py` | 单目标 |
| `coordnet` | `coordnet.py` | 单目标 |
| `moe_inr` | `moe_inr.py` | 单目标 |
| `instant_ngp` | `instant_ngp.py` | 单目标 |
| `instant_vnr` | `instant_vnr.py` | 单目标 |
| `mvnet` | `mvnet.py` | 一个模型同时输出多个标量目标 |
| `stsr_inr` | `stsr_inr.py` | 一个模型同时输出多个目标 |

`hash_grid.py` 包含哈希网格基线共用的编码组件，不是独立的已注册模型。

## 共享生命周期

所有已注册基线都从公共数据工厂接收数据集元数据。Registry 会实例化默认值、检查
模型专属字段及输入/输出维度，并返回公共模型适配器。随后训练使用共享的采样器、损失、
日志、checkpoint 与预测代码。

```bash
python -m var_expert_inr.cli train \
  --config configs/main/SIREN/ionization__GT.yaml
```

单目标模型要求配置只选择一个目标。多目标模型会在有效配置和 checkpoint 中保留输出
名称顺序，使预测与评估可以恢复相同映射。

可直接运行的实验设置应放在 `configs/`，而不是本目录。组织规则见
[配置指南](../../../../configs/README.zh-CN.md)，输入契约见
[数据指南](../../data/README.zh-CN.md)。
