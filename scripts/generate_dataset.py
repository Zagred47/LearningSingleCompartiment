#!/usr/bin/env python
"""Generate a single-compartment dataset from a JSON configuration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hay_single_compartment import SimulationConfig, generate_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/default.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/single_compartment.h5"))
    args = parser.parse_args()
    config = SimulationConfig.from_dict(json.loads(args.config.read_text(encoding="utf-8")))
    print(json.dumps(generate_dataset(args.output, config), indent=2))


if __name__ == "__main__":
    main()
