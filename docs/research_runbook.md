# Research runbook

## Before

1. Select one open hypothesis and the knowledge entries the run may change.
2. Choose and justify a search mode, taxonomy family, design patterns and atomic capabilities; for a creative screen, archive all sketches rather than only the winner.
3. Copy `research/experiment_card_template.json`; fill decisions, information contract, orthogonal-information test and artifacts before training.
4. Register the run; name one primary factor and fixed controls.
5. Verify dataset hash, trajectory-disjoint split, closed test and Git commit.
6. Save deterministic batch/window manifests; declare parameters, updates, seeds and compute.
7. State support, falsification, inconclusive signatures, qualitative signatures and guardrails.

## During

Log ETA, elapsed time, epoch/update, learning rate, component losses, granular regime metrics and guardrails. Save best-validation and last checkpoints, RNG state, optimizer/scheduler and environment. Capture the mandatory plot suite and internal activation/gradient statistics declared in the card. Never inspect test. A restart with changed hyperparameters is a new experiment.

## After

1. Verify archive completeness and SHA-256.
2. Apply the preregistered decision rule before interpreting plots.
3. Complete structured qualitative observation records: literal description first, explanations second, then measurable signature and falsifying view.
4. Update registry, knowledge ledger, hypothesis graph, design-pattern evidence and literature decisions in the same commit.
5. Record alternatives and the next orthogonal distinction.
6. Stop after two repeats of the same failure signature.
7. Run `python scripts/validate_research_program.py` and tests.
8. Regenerate `research/research_program.xlsx`.

Every archive contains the experiment card, commit and hashes, configuration, manifests, metrics, figures, logs and decision. Lightweight archives may omit checkpoint binaries but not their hashes and locations.
