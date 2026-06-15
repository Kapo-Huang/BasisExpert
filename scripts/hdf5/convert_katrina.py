from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from var_expert_inr.utils.hdf5_conversion import DatasetConversionSpec, main_for_dataset

SPEC = DatasetConversionSpec(
    dataset_name="Katrina",
    dataset_kind="node",
    default_input_dir=REPO_ROOT / "data" / "Mesh" / "Katrina",
    default_output_filename="katrina.h5",
    coords_file="source_XYZT.npy",
    targets={
        "fort63": "target_fort63.npy",
        "fort64": "target_fort64.npy",
        "fort73": "target_fort73.npy",
        "speed": "target_speed.npy",
        "v": "target_v.npy",
    },
)


def main() -> int:
    return main_for_dataset(SPEC)


if __name__ == "__main__":
    raise SystemExit(main())
