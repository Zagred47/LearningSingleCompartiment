"""Generate the cached four-compartment spike-driven dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hay_single_compartment import MicroDatasetConfig, generate_micro_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    report = generate_micro_dataset(
        args.output,
        MicroDatasetConfig(),
        workers=args.workers,
        force=args.force,
        progress=True,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
