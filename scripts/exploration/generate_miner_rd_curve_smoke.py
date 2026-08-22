from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.exploration.generate_rd_curve_smoke import generate


def main() -> None:
    counts = generate("MINER")
    print(f"Generated {sum(counts.values())} MINER exploration-v3 Size configs: {counts}")


if __name__ == "__main__":
    main()
