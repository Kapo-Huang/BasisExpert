# VarExpert-INR

[English](README.md) | [简体中文](README.zh-CN.md)

VarExpert-INR 是一个基于 PyTorch 的隐式神经表示（INR）框架，用于在时序节点场和
结构化体数据上完成训练、预测与评估。项目为 VarExpert 及多种基线提供统一训练引擎，
同时为具有特殊训练或 checkpoint 生命周期的方法提供自包含复现。

## 环境要求与安装

- Python 3.10 或更高版本
- PyTorch 以及 `pyproject.toml` 中声明的依赖
- 正式实验建议使用支持 CUDA 的环境

从仓库根目录安装项目：

```bash
python -m pip install -e .
```

需要 SSIM、LPIPS 或节点渲染时，安装可选的图像、网格和感知评估依赖：

```bash
python -m pip install -e ".[evaluation]"
```

体渲染还需要相邻目录中的 VolumeVis 项目：

```bash
python -m pip install -e "../Vis[lpips]"
```

## 数据契约

节点目标值和体数据目标值必须已经缩放到 `[-1, 1]`。节点坐标也必须在该范围内，
除非配置提供经过检查的 `coordinate_stats_path` 统计量用于标准化。结构化体数据坐标
由数组索引生成，并由加载器归一化。

Python 包不分发正式实验数据集。请按照[数据指南](src/var_expert_inr/data/README.zh-CN.md)
和[配置指南](configs/README.zh-CN.md)，通过 YAML 与环境变量配置数据路径。

## 快速开始

请从仓库根目录执行命令。统一引擎模型只需提供实验配置即可训练：

```bash
python -m var_expert_inr.cli train \
  --config configs/main/VarExpert/ionization.yaml
```

如果没有显式指定 checkpoint，预测会使用该配置 `exp_id` 对应的最新时间戳运行：

```bash
python -m var_expert_inr.cli predict \
  --config configs/main/VarExpert/ionization.yaml
```

可以按指标、目标和闭区间时间步选择评估现有运行：

```bash
python -m var_expert_inr.cli evaluate \
  --run runs/<exp_id>/<timestamp> \
  --metrics psnr,decode_time,memory \
  --targets GT \
  --timesteps 0,10:30,40:99:10
```

具有特殊生命周期的方法使用各自的模块，例如：

```bash
python -m var_expert_inr.methods.apmgsrn.cli train \
  --config configs/main/APMGSRN/combustion_40NH3_1__Temperature.yaml
```

完整入口与能力矩阵见[方法指南](src/var_expert_inr/methods/README.zh-CN.md)，指标、
渲染、选择与缓存行为见[评估指南](src/var_expert_inr/evaluation/README.zh-CN.md)。

## 正式实验运行器

选择列表是当前实验矩阵的权威来源。默认运行器读取
`scripts/main/all_configs.list`，其中同时包含 Main 与 RD-Curve 实验：

```bash
# 完整正式实验选择
bash scripts/main/run_all.sh

# 仅 Main 实验
CONFIG_LIST_FILE=scripts/main/configs.list bash scripts/main/run_all.sh

# 仅 RD-Curve 实验
bash scripts/rd_curve/run.sh
```

所有 Bash 入口默认使用原始服务器的 `compression` Conda 环境。需要 AutoDL 时应
显式选择对应 profile：

```bash
bash scripts/main/run_all.sh --env autodl
SERVER_ENV=autodl bash scripts/rd_curve/run.sh
```

[自动化指南](scripts/README.zh-CN.md)详细说明服务器 profile、数据集覆盖变量、
并发、dry run、重试行为、批日志和专用运行器。

## 运行目录

训练会在解析后的 `RUNS_ROOT` 下写入一个带时间戳的运行：

```text
runs/<exp_id>/<timestamp>/
├── checkpoints/   # 推理 checkpoint 或 bundle
├── configs/       # 解析后的有效配置
├── logs/
├── metrics/
└── predictions/
```

基于运行的评估会将自包含报告写入 `<run>/evaluations/<timestamp>/`。原始服务器
profile 的 `RUNS_ROOT` 默认为仓库中的 `runs/`，AutoDL 默认为
`/root/autodl-tmp/runs`。

## 文档导航

| 领域 | 文档 |
| --- | --- |
| 实验 YAML 与配置生成 | [configs/README.zh-CN.md](configs/README.zh-CN.md) |
| 批处理运行器与工具 | [scripts/README.zh-CN.md](scripts/README.zh-CN.md) |
| 节点与体数据契约 | [data/README.zh-CN.md](src/var_expert_inr/data/README.zh-CN.md) |
| 评估、指标与渲染 | [evaluation/README.zh-CN.md](src/var_expert_inr/evaluation/README.zh-CN.md) |
| 自包含方法入口 | [methods/README.zh-CN.md](src/var_expert_inr/methods/README.zh-CN.md) |
| 共享引擎基线 | [baselines/README.zh-CN.md](src/var_expert_inr/models/baselines/README.zh-CN.md) |
| 提出的方法 | [proposed/README.zh-CN.md](src/var_expert_inr/models/proposed/README.zh-CN.md) |

## 使用说明

- `data/`、`runs/`、`runs_summary/` 和 `batch_logs/` 是本地产物，不纳入版本控制。
- 质量指标需要可读且形状兼容的 ground truth；性能指标可以在没有 ground truth 的
  情况下评估 checkpoint 解码。
- 体渲染需要 VolumeVis。节点渲染需要受支持的网格，或显式的顶点与单元数据；不提供
  点云回退方案。
- 实验矩阵变化时，生成配置数量也会变化。请查询正在使用的 `.list` 文件，不要依赖
  复制到说明文字中的固定数字。
