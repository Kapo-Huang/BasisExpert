# 数据契约

[English](README.md) | [简体中文](README.zh-CN.md)

共享数据层以内存映射方式读取 NumPy 数组，并向训练和评估引擎提供索引批次。它支持
非结构化节点样本与结构化时序体数据。

## 公共要求

- 输入必须是 `numpy.load` 可读取的 NumPy 文件，载荷数组通常存为 `.npy`。
- 所有目标值必须有限且位于 `[-1, 1]` 内（检查容差为 `1e-6`）。加载器不会归一化
  目标值。
- 同一数据集中的所有目标数组必须描述相同样本，并具有兼容形状。
- `target_path` 定义一个名为 `target` 的目标；`targets` 将稳定的目标名称映射到路径。
  单目标方法可以通过 `data.target` 选择其中一个映射项。
- 目标名称、维度和顺序会成为数据集元数据，并保存在有效配置和 checkpoint 中。

## 节点场

节点数据集的坐标形状为 `(N, D)`，目标形状为 `(N,)` 或 `(N, C)`。每个目标的第一维
都必须等于 `N`。

```yaml
data:
  kind: node
  dataset_name: redsea
  coords_path: ${REDSEA_ROOT}/source_XYZT.npy
  targets:
    TEMP: ${REDSEA_ROOT}/target_TEMP.npy
    SALT: ${REDSEA_ROOT}/target_SALT.npy
```

除非提供 `coordinate_stats_path`，坐标必须有限且位于 `[-1, 1]`。坐标统计文件是一个
`.npz`，其中包含长度为 `D` 的一维 `x_mean` 和 `x_std` 数组；所有标准差必须有限且
为正。提供后，批次使用 `(x - x_mean) / x_std` 标准化，而不再对存储坐标应用范围检查。

数据加载器不会主动将节点目标划分为时间步。当运行与渲染器需要时，评估流程根据时间
坐标推导时序帧。

## 结构化体数据

标量场的标准体布局为 `(T, Z, Y, X)`，向量场为 `(T, Z, Y, X, C)`。如果提供
`volume_shape` 且 `N = T * Z * Y * X`，也接受扁平的 `(N,)` 和 `(N, C)` 目标。

```yaml
data:
  kind: volume
  dataset_name: ionization
  volume_shape:
    X: 600
    Y: 248
    Z: 248
    T: 100
  targets:
    GT: ${IONIZATION_ROOT}/target_GT.npy
```

扁平化使用 C 顺序：`x` 变化最快，之后依次为 `y`、`z`、`t`。默认坐标向量是
`(x, y, z, t)`。每个索引轴映射到 `[-1, 1]`，单元素轴映射为零。

`coordinate_axes` 只能省略单元素维度，并且必须保持标准 `x, y, z, t` 顺序。多目标
体数据中的所有目标必须解析为相同 `VolumeShape`。

## 路径解析

配置路径可以是绝对路径、相对 YAML 文件的路径，或使用
[配置指南](../../../configs/README.zh-CN.md)中的可移植占位符。数据集专属环境变量
优先于服务器 profile 的默认值。

正式数据目录是本地产物，不纳入版本控制。只有当所有引用数组都存在，并且形状及归一化
值符合声明时，配置才可运行。

## 检查失败

缺失目标、不受支持的数组维数、样本数量或体形状不一致、无效坐标轴、NaN/Inf，或数值
超出要求范围，都会使数据集构建提前失败。这些错误用于保护实验语义；不要通过在加载器
中重新解释数组形状或裁剪数值来绕过它们。
