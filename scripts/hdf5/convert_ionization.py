from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from var_expert_inr.utils.hdf5_conversion import DatasetConversionSpec, main_for_dataset

SPEC = DatasetConversionSpec(
    dataset_name="Ionization",
    dataset_kind="volume",
    default_input_dir=REPO_ROOT / "data" / "Volume" / "Ionization",
    default_output_filename="ionization.h5",
    targets={
        "GT": "target_GT_sub_2.npy",
        "H_plus": "target_H+_sub_2.npy",
        "H2": "target_H2_sub_2.npy",
        "He": "target_He_sub_2.npy",
        "PD": "target_PD_sub_2.npy",
    },
    extra_root_attrs={
        "volume_shape_X": 600,
        "volume_shape_Y": 248,
        "volume_shape_Z": 248,
        "volume_shape_T": 2,
    },
)


def main() -> int:
    return main_for_dataset(SPEC)


if __name__ == "__main__":
    raise SystemExit(main())
