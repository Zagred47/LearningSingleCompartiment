# Research runbook

## Before

1. Select one open hypothesis and the knowledge entries the run may change.
2. Copy `research/experiment_card_template.json`; fill decisions, information contract and artifacts before training.
3. Register the run; name one primary factor and fixed controls.
4. Verify dataset hash, trajectory-disjoint split, closed test and Git commit.
5. Save deterministic batch/window manifests; declare parameters, updates, seeds and compute.
6. State support, falsification, inconclusive signatures and guardrails.

## During

Log ETA, elapsed time, epoch/update, learning rate, component losses, primary metric and guardrails. Save best-validation and last checkpoints, RNG state, optimizer/scheduler and environment. Never inspect test. A restart with changed hyperparameters is a new experiment.

## After

1. Verify archive completeness and SHA-256.
2. Apply the preregistered decision rule before interpreting plots.
3. Update registry, knowledge ledger, hypothesis graph and literature decisions in the same commit.
4. Record alternatives and the next orthogonal distinction.
5. Stop after two repeats of the same failure signature.
6. Run `python scripts/validate_research_program.py` and tests.
7. Regenerate `research/research_program.xlsx`.

Every archive contains the experiment card, commit and hashes, configuration, manifests, metrics, figures, logs and decision. Lightweight archives may omit checkpoint binaries but not their hashes and locations.

