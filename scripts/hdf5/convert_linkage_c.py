from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from var_expert_inr.utils.hdf5_conversion import DatasetConversionSpec, main_for_dataset

SPEC = DatasetConversionSpec(
    dataset_name="Linkage-C",
    dataset_kind="node",
    default_input_dir=REPO_ROOT / "data" / "Mesh" / "Linkage-C",
    default_output_filename="linkage_c.h5",
    coords_file="source_cell_XYZT.npy",
    targets={
        "cell_E": "target_cell_E.npy",
        "cell_E_IntegrationPoints": "target_cell_E_IntegrationPoints.npy",
        "cell_S": "target_cell_S.npy",
        "cell_S_IntegrationPoints": "target_cell_S_IntegrationPoints.npy",
    },
)


def main() -> int:
    return main_for_dataset(SPEC)


if __name__ == "__main__":
    raise SystemExit(main())
