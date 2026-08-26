# 评估、指标与渲染

[English](README.md) | [简体中文](README.zh-CN.md)

基于运行的评估器为共享引擎与自包含方法提供统一接口。它可以解码 checkpoint、读取
已保存预测、选择目标与时间步、计算质量或性能指标，并渲染对应帧。

## 基本用法

```bash
python -m var_expert_inr.cli evaluate \
  --run runs/<exp_id>/<timestamp> \
  --metrics psnr,ssim,lpips,decode_time,memory \
  --targets GT,H2 \
  --timesteps 0,10:30,40:99:10
```

统一 CLI 接受 `--run` 或 `--config`。使用 `--config` 时，除非显式 checkpoint 可以
标识一个运行，否则会使用该配置 `exp_id` 对应的最新时间戳运行。独立方法 CLI 通过其
`evaluate` 命令暴露相同的运行评估参数。

## 指标与前置条件

默认指标是 `psnr`。

| 指标或操作 | Ground truth | 渲染 | 主要结果 |
| --- | --- | --- | --- |
| `psnr` | 必需 | 否 | MSE、MAE 和 PSNR 汇总。 |
| `ssim` | 必需 | 必需 | 匹配 GT 与预测图像上的 SSIM。 |
| `lpips` | 必需 | 必需 | 匹配 GT 与预测图像上的 LPIPS。 |
| `decode_time` | 非必需 | 否 | 独立的新鲜解码计时，不包含渲染和指标计算。 |
| `memory` | 非必需 | 否 | 进程 RSS，以及可用时的 CUDA allocated/reserved 峰值。 |
| `--render` | 可选 | 必需 | 选定预测帧；GT 可用时也会渲染 GT 帧。 |

当 ground truth 缺失、不可读或形状不兼容时，PSNR、SSIM 和 LPIPS 会在解码前失败。
基于 checkpoint 的性能评估可以在没有目标的情况下构造坐标；节点评估仍需要坐标数组。

安装可选依赖：

```bash
python -m pip install -e ".[evaluation]"
python -m pip install -e "../Vis[lpips]"  # 仅体渲染需要
```

## 选择语法

- `--targets all` 选择所有已配置目标；否则使用逗号分隔列表。`all` 不能与显式名称组合。
- `--timesteps all` 选择所有时间步。
- 时间步 token 可以是 `N` 或闭区间 `start:end[:step]`。逗号分隔组合保留第一次出现的
  顺序，并拒绝越界索引。

例如，`0,10:14,20:40:10` 选择 `0, 10, 11, 12, 13, 14, 20, 30, 40`。

## 评估来源

`--source` 接受 `auto`、`checkpoint` 或 `prediction`。

- 显式的 `--checkpoint` 或 `--prediction` 路径优先。
- `auto` 优先使用 `checkpoints/` 下的标准/最终 `.pth` checkpoint，然后回退到
  `predictions/` 下已保存的 `.npy` 预测。
- 找不到合适 checkpoint 时，`checkpoint` 会失败。
- 对已保存预测请求性能指标时，测量的是预测文件访问而不是模型解码，报告中会相应标记。

评估器使用运行中保存的有效配置。对于旧式绝对 GT 路径，它还会检查仓库相邻的数据集
位置和本地 `data/` 目录。

## 渲染 Profile

内置 profile 覆盖 Ionization、Combustion、RedSea 和 Katrina。profile 的 `renderer`
可为 `volume`、`image2d` 或 `mesh`；可用 `--eval-config <profile.yaml>` 覆盖内置视角。

体 profile 声明 `kind: volume`、布局、渲染器选项，以及可选的目标到 preset 映射。
体渲染需要相邻的 VolumeVis 包。节点 profile 声明 `kind: node`、点/单元关联、相机和
颜色设置，并提供以下一种网格来源：

- 用于 VTK/VTU 或 ADCIRC `fort.14` 的 `mesh_path` 或 `mesh_path_template`；或
- 顶点与单元两个 NumPy 路径，也可以使用时间步模板。

Combustion 使用 `image2d` 和默认 `viridis` 色图。RedSea 内置视角选择表层坐标，并通过
VTP 的 `wet_mask_surface` 把预测值映射回表层网格。首次渲染前，把本地、Git 忽略的
mesh 准备到固定位置：

```powershell
New-Item -ItemType Directory -Force data/Mesh/RedSea/render
Copy-Item E:/Research/Project/Scientific Compression/INR/Datasets/RedSea_SciVisContest2020/0001/0001/paraview/surface_vtp/surface_0000.vtp `
  data/Mesh/RedSea/render/surface_0000.vtp
```

其他机器将同一 SciVis 导出文件复制到上述目标即可。文件不存在时，评估器会在解码前
报告预期路径。历史 `dataset_name: bathymetry` 运行会自动使用 RedSea profile。

不提供点云回退。仅预测渲染必须设置固定 `clim` 或目标专属 `target_clims`；否则颜色
范围从 ground truth 推导。

## 报告与缓存

每次新评估写入：

```text
EvalResult/Main/<model>/<dataset>/<target>/
├── manifest.json
├── metrics.json
├── metrics.csv
├── logs/evaluate.log
└── renders/<target>/...   # 请求渲染时生成
```

当来源、选择范围、渲染 profile 和 ground-truth 指纹一致时，可以复用质量与渲染结果。
`--overwrite` 绕过该缓存。`decode_time` 和 `memory` 始终执行新测量，不会从质量结果
缓存中返回。
