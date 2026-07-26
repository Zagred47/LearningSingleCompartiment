#!/usr/bin/env python
"""Train MLP, GRU, and LSTM baselines on a generated HDF5 dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hay_single_compartment.training import train_architectures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/models"))
    parser.add_argument("--architectures", nargs="+", default=["mlp", "gru", "lstm"])
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    reports = train_architectures(
        args.dataset,
        args.output_dir,
        args.architectures,
        epochs=args.epochs,
        device=args.device,
    )
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
