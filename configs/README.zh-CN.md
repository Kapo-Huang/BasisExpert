# 实验配置

[English](README.md) | [简体中文](README.zh-CN.md)

本目录存放实验 YAML。配置首先按照实验要回答的问题分类，再按照论文中的方法名称
组织。所有命令均应从仓库根目录执行。

## 目录结构

| 目录 | 用途 |
| --- | --- |
| `main/` | 各方法默认正式预算下的主要对比实验。 |
| `rd_curve/` | 正式率失真层级：`Size041`、`Size082`、`Size163`、`Size326` 和 `Size652`。 |
| `exploration/` | 训练计划、优化器与可行性的短程探测。 |
| `ablation/` | 受控的架构、深度与正则化消融。 |
| `sensitivity/` | VarExpert 专家数量与 Top-K 扫描。 |
| `variable_scaling/` | 不同变量数量下的正式 VarExpert 实验。 |

研究目录使用语义名称，而不是按时间顺序命名的 `vN`。路径中保留 `fV-SRN` 等方法
名称，以便与论文和运行元数据一致。

## 正式实验选择

当前正式矩阵由选择文件定义，不以文档中复制的固定数量为准：

| 选择范围 | 文件 | 运行命令 |
| --- | --- | --- |
| Main 与 RD-Curve | `scripts/main/all_configs.list` | `bash scripts/main/run_all.sh` |
| 仅 Main | `scripts/main/configs.list` | `CONFIG_LIST_FILE=scripts/main/configs.list bash scripts/main/run_all.sh` |
| 仅 RD-Curve | `scripts/rd_curve/configs.list` | `bash scripts/rd_curve/run.sh` |

每个非空且非注释行表示一条仓库相对 YAML 路径。可以复制列表并设置
`CONFIG_LIST_FILE`，在不修改默认列表的情况下运行自定义子集。

## 统一配置结构

共享引擎配置使用以下顶层结构：

```yaml
experiment: descriptive_name
exp_id: stable-run-id
experiment_root: ${RUNS_ROOT}

data:
  kind: volume                 # volume 或 node
  dataset_name: ionization
  targets:
    GT: ${IONIZATION_ROOT}/target_GT.npy
  volume_shape: {X: 600, Y: 248, Z: 248, T: 100}

model:
  name: var_expert
  # 后续为方法专属字段。

training:
  epochs: 600
  batch_size: 16000
  lr: 5.0e-5

evaluation:
  batch_size: 16000
```

`data.kind` 可以是 `node` 或 `volume`。节点配置提供 `coords_path`，以及
`target_path` 或 `targets`；体数据配置提供 `target_path` 或 `targets`，扁平数组还可能
需要 `volume_shape`。单目标方法可以通过 `data.target` 选择 `targets` 中的一个成员。
形状和取值范围要求见[数据指南](../src/var_expert_inr/data/README.zh-CN.md)。

独立方法可能拥有额外的专属字段或 schema。请以该方法目录中的现有配置为模板，并根据
[方法文档](../src/var_expert_inr/methods/README.zh-CN.md)选择正确 CLI。

## 路径占位符

生成的配置使用可移植占位符，并在加载配置时解析：

| 占位符 | 默认值 |
| --- | --- |
| `${REPO_ROOT}` | 从配置或包位置发现的仓库根目录。 |
| `${RUNS_ROOT}` | `original` 为 `<repo>/runs`；`autodl` 为 `/root/autodl-tmp/runs`。 |
| `${DATASETS_ROOT}` | 相邻的 `INR/Datasets` 目录，除非显式覆盖。 |
| `${COMBUSTION_ROOT}` | `<repo>/data/Volume/Combustion` 或 `<AUTODL_DATA_ROOT>/Combustion`。 |
| `${IONIZATION_ROOT}` | `<repo>/data/Volume/Ionization` 或 `<AUTODL_DATA_ROOT>/Ionization`。 |
| `${REDSEA_ROOT}` | `<repo>/data/Mesh/RedSea` 或 `<AUTODL_DATA_ROOT>/RedSea`。 |
| `${KATRINA_ROOT}` | `<repo>/data/Mesh/Katrina` 或 `<AUTODL_DATA_ROOT>/Katrina`。 |

每个同名环境变量都可以覆盖对应数据集根目录。`AUTODL_DATA_ROOT` 用于修改 AutoDL
公共基础目录。

## 生成配置

生成文件必须通过 `scripts/<category>/` 下的对应入口重建。主要矩阵使用：

```bash
python scripts/main/generate_configs.py
```

其他生成器与各研究运行器放在一起，例如：

```bash
python scripts/ablation/generate_architecture.py
python scripts/exploration/generate_optimizer_tuning.py
python scripts/sensitivity/generate_var_expert_num.py
python scripts/sensitivity/generate_var_expert_topk.py
```

不要创建新的顶层 `configs_*` 目录。应在现有分类下添加语义明确的研究目录，并保持
生成的选择列表与 YAML 文件同步。
