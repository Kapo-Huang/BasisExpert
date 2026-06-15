from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from var_expert_inr.utils.hdf5_conversion import DatasetConversionSpec, main_for_dataset

SPEC = DatasetConversionSpec(
    dataset_name="Linkage-P",
    dataset_kind="node",
    default_input_dir=REPO_ROOT / "data" / "Mesh" / "Linkage-P",
    default_output_filename="linkage_p.h5",
    coords_file="source_point_XYZT.npy",
    targets={
        "point_RF": "target_point_RF.npy",
        "point_U": "target_point_U.npy",
    },
)


def main() -> int:
    return main_for_dataset(SPEC)


if __name__ == "__main__":
    raise SystemExit(main())
