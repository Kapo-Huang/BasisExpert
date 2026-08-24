# 实验自动化

[English](README.md) | [简体中文](README.zh-CN.md)

脚本目录与 `configs/` 对应，包含配置生成器、正式与研究运行器、共享 Bash 基础函数，
以及数据集和结果工具。所有命令均应从仓库根目录执行。

## 目录结构

| 目录 | 内容 |
| --- | --- |
| `main/` | 主要配置生成器、完整正式运行器和选定数据集/模型运行器。 |
| `rd_curve/` | RD-Curve 选择列表与包装运行器。 |
| `exploration/` | 优化器、ECNR 和 RD 可行性短程研究。 |
| `ablation/` | 架构、深度和正则化的生成器、运行器与汇总。 |
| `sensitivity/` | VarExpert 专家数量与 Top-K 的生成器、选择列表和运行器。 |
| `variable_scaling/` | 变量数量选择列表与运行器。 |
| `tools/` | 数据检查/导出和结果维护工具。 |
| `lib/` | 共享 shell 函数，不是用户入口。 |

## 正式运行器

```bash
# 使用 scripts/main/all_configs.list 的 Main 与 RD-Curve 选择
bash scripts/main/run_all.sh

# 仅 Main 实验
CONFIG_LIST_FILE=scripts/main/configs.list bash scripts/main/run_all.sh

# 仅 RD-Curve 实验
bash scripts/rd_curve/run.sh

# 任意使用仓库相对路径的自定义选择列表
CONFIG_LIST_FILE=scripts/my_configs.list bash scripts/main/run_all.sh
```

`run_all.sh` 定义模型分组和执行顺序。选择列表仅过滤该矩阵；未被预定义分组识别的路径
会集中到一个附加分组。非空行可以带有行尾 `#` 注释。

专用正式入口包括：

| 入口 | 选择范围 |
| --- | --- |
| `run_neural_expert_non_ionization.sh` | SIREN 与 NeuralExpert 的非 Ionization 阶段及评估。 |
| `run_moe_non_ionization.sh` | MoE-INR 非 Ionization 实验。 |
| `run_selected_datasets.sh` | CoordNet Combustion、MVNet Katrina 和 STSR-INR RedSea。 |
| `run_combustion_fv_apmg_instantvnr.sh` | Combustion fV-SRN、APMGSRN 和 InstantVNR。 |
| `run_combustion_stsr_mvnet.sh` | Combustion STSR-INR 和 MVNet。 |
| `run_combustion_miner_ecnr.sh` | Combustion MINER 和 ECNR。 |

## 服务器 Profile

每个面向用户的 Bash 运行器都接受 `original` 和 `autodl`：

```bash
bash scripts/main/run_all.sh --env autodl
bash scripts/main/run_all.sh --env=autodl
bash scripts/main/run_all.sh env=autodl
SERVER_ENV=autodl bash scripts/main/run_all.sh
```

| 设置 | `original` | `autodl` |
| --- | --- | --- |
| Python 启动 | `conda run -n ${CONDA_ENV} ${PYTHON_BIN}` | 直接使用 `${PYTHON_BIN}` |
| 默认 `CONDA_ENV` | `compression` | 启动器不使用 |
| 默认 `PYTHON_BIN` | `python` | `python` |
| 默认 `RUNS_ROOT` | `<repo>/runs` | `/root/autodl-tmp/runs` |
| 默认数据集根目录 | 仓库 `data/Mesh` 与 `data/Volume` | `/root/autodl-tmp` |

使用 `AUTODL_DATA_ROOT` 替换 AutoDL 数据集公共根目录，或使用
`COMBUSTION_ROOT`、`IONIZATION_ROOT`、`REDSEA_ROOT`、`KATRINA_ROOT` 分别覆盖
单个数据集。

## 批处理控制与恢复

| 变量 | 行为 |
| --- | --- |
| `BATCH_LOG_ROOT` | 复用或指定批处理目录。 |
| `MAX_PARALLEL_JOBS` | 在运行器支持分组时限制并发任务数；对 `run_all.sh` 而言，`0` 表示不限制。 |
| `DRY_RUN=1` | 打印命令，不训练且不写入状态行。 |
| `RUN_TOKEN` | 覆盖新批次使用的时间戳 token。 |
| `VAR_EXPERT_INR_NUM_THREADS` | 传播给常见数值库的默认线程数。 |

除非设置 `BATCH_LOG_ROOT`，每个批次都会在 `batch_logs/<timestamp>/` 下写入
`status.tsv`、`failed.txt` 和逐次尝试日志。同一配置路径的最后一条状态记录具有权威性。
复用批次时会跳过 `ok` 配置；失败、中断、缺失或无效状态会开始全新的训练尝试。恢复身份
由配置路径决定，而不是 YAML 内容哈希。

## 研究运行器

生成器和运行器按照分类放在一起。典型工作流：

```bash
python scripts/ablation/generate_architecture.py
bash scripts/ablation/run_architecture.sh

python scripts/exploration/generate_optimizer_tuning.py
bash scripts/exploration/run_optimizer_tuning.sh

python scripts/sensitivity/generate_var_expert_num.py
bash scripts/sensitivity/run_var_expert_num.sh

python scripts/sensitivity/generate_var_expert_topk.py
bash scripts/sensitivity/run_var_expert_topk.sh

bash scripts/variable_scaling/run_v04.sh
```

研究专属的运行和批处理根目录由各入口定义。不要在未检查对应生成器和运行器的情况下，
跨研究复用 `BATCH_LOG_ROOT`。

## 数据集与结果工具

| 工具 | 用途 |
| --- | --- |
| `scripts/tools/combustion.py` | 检查/渲染 RealPDEBench combustion 轨迹并导出归一化体数据。 |
| `scripts/tools/katrina_wet.py` | 检查和导出 Katrina 动态湿节点样本。 |
| `scripts/tools/evaluate_neural_expert_config.py` | 使用 NeuralExpert 方法生命周期评估配置。 |
| `scripts/tools/organize_runs_summary.py` | 构建本地运行汇总视图和报告。 |

使用 `python <tool> --help` 查看当前子命令与参数。数据集路径应显式提供，示例不应依赖
开发者工作站上的绝对路径。
