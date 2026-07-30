"""Generate or reuse the balanced faithful-Hay soma dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hay_single_compartment import FaithfulSimulationConfig, generate_faithful_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "output",
        nargs="?",
        type=Path,
        default=Path("artifacts/hay_faithful_soma_balanced_v1.h5"),
    )
    parser.add_argument("--force", action="store_true", help="replace an incompatible cache")
    parser.add_argument("--workers", type=int, default=1, help="parallel trajectory workers")
    args = parser.parse_args()
    report = generate_faithful_dataset(
        args.output,
        FaithfulSimulationConfig(),
        progress=True,
        reuse=True,
        force=args.force,
        workers=args.workers,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
