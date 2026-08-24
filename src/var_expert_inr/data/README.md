# Data contracts

[English](README.md) | [简体中文](README.zh-CN.md)

The shared data layer memory-maps NumPy arrays and exposes indexed batches to
the training and evaluation engines. It supports unstructured node samples and
structured temporal volumes.

## Common requirements

- Inputs are NumPy files readable by `numpy.load`; payload arrays are normally
  stored as `.npy`.
- Every target value must be finite and within `[-1, 1]` (with a `1e-6`
  validation tolerance). The loader does not normalize target values.
- All target arrays in a dataset must describe the same samples and have
  compatible shapes.
- `target_path` defines one target named `target`. `targets` maps stable target
  names to paths. Single-target methods can select one mapping entry with
  `data.target`.
- Target names, dimensions, and ordering become dataset metadata and are
  preserved in effective configs and checkpoints.

## Node fields

A node dataset uses coordinates with shape `(N, D)` and targets with shape
`(N,)` or `(N, C)`. Every target's first dimension must equal `N`.

```yaml
data:
  kind: node
  dataset_name: redsea
  coords_path: ${REDSEA_ROOT}/source_XYZT.npy
  targets:
    TEMP: ${REDSEA_ROOT}/target_TEMP.npy
    SALT: ${REDSEA_ROOT}/target_SALT.npy
```

Coordinates must be finite and within `[-1, 1]` unless
`coordinate_stats_path` is supplied. A coordinate-statistics file is an `.npz`
containing one-dimensional `x_mean` and `x_std` arrays with length `D`; all
standard deviations must be finite and positive. When present, batches are
standardized as `(x - x_mean) / x_std` instead of applying the range check to
the stored coordinates.

Node targets are not grouped into timesteps by the data loader. Evaluation
derives temporal frames from the time coordinate when the run and renderer
require them.

## Structured volumes

Canonical volume layout is `(T, Z, Y, X)` for scalar fields and
`(T, Z, Y, X, C)` for vector fields. Flat `(N,)` and `(N, C)` targets are also
accepted when `volume_shape` is supplied and `N = T * Z * Y * X`.

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

Flattening uses C order: `x` changes fastest, followed by `y`, `z`, and `t`.
The default coordinate vector is `(x, y, z, t)`. Each index axis is mapped to
`[-1, 1]`; a singleton axis maps to zero.

`coordinate_axes` may omit only singleton dimensions and must preserve the
canonical `x, y, z, t` order. Every target in a multi-target volume must resolve
to the same `VolumeShape`.

## Path resolution

Config paths may be absolute, relative to the YAML file, or use the portable
placeholders documented in the
[configuration guide](../../../configs/README.md). Dataset-specific
environment variables take precedence over the server-profile defaults.

The formal data directories are local artifacts and are excluded from version
control. A configuration is runnable only after all referenced arrays exist
with the declared shapes and normalized values.

## Validation failures

Dataset construction fails early for missing targets, unsupported ranks,
sample-count or volume-shape mismatches, invalid coordinate axes, NaN/Inf
values, or values outside the required range. These errors protect experiment
semantics; do not bypass them by reinterpreting array shapes or clipping values
inside the loader.
