# Research operating system

This directory is the canonical memory of the micro-Hay research programme.
It separates immutable facts, falsifiable hypotheses, planned experiments and
human-readable interpretation so that a new notebook cannot silently reset the
reasoning history.

## Sources of truth

| File | Purpose |
|---|---|
| `master_plan.json` | finite gated roadmap, budgets and branch stop rules |
| `hypothesis_graph.json` | hypotheses and typed evidence/dependency edges |
| `knowledge_ledger.json` | scoped claims already learned from experiments |
| `architecture_catalog.json` | established architecture families and entry conditions |
| `indicator_catalog.json` | output, activation, gradient and loss-landscape diagnostics |
| `experiment_registry.csv` | one row per preregistered or completed causal comparison |
| `literature_evidence.csv` | primary scientific grounding and transfer limits |
| `experiment_card_template.json` | mandatory pre-run and post-run record |
| `research_program.xlsx` | read-only convenient view of the canonical files |
| `methodology_source_index.json` | identity and incorporation map for the 15 supplied methodology sources |
| `creative_methodology.json` | physics-style, drug-design and compositional creative workflow |
| `inductive_bias_taxonomy.json` | complete architecture library organized by strength of assumption |
| `design_pattern_library.json` | established functional fragments and their diagnostic entry conditions |
| `evaluation_taxonomy.json` | behavioral, internal, optimization, qualitative and capability metrics |
| `qualitative_observation_template.json` | structured conversion of a plot observation into a testable signature |

JSON and CSV files are canonical. The workbook and Markdown documents are
views. They must never become independent stores of unrecorded decisions.

## Mandatory lifecycle

1. Add or activate a hypothesis in `hypothesis_graph.json`.
2. Add a preregistered row to `experiment_registry.csv` and complete an
   experiment card before training.
3. Run only after baseline, factor, primary metric, guardrails, update budget,
   seeds and decision thresholds are frozen.
4. Attach artifact SHA-256 and update the experiment decision.
5. Add or revise at least one entry in `knowledge_ledger.json`.
6. Update affected hypothesis edges and architecture status.
7. Run `python scripts/validate_research_program.py` before commit.

A notebook that does not change the state of a hypothesis or knowledge claim
is exploratory and cannot authorize the next architecture.

## Status vocabulary

- hypothesis: `open`, `supported`, `weakened`, `falsified`, `blocked`;
- experiment: `planned`, `preregistered`, `running`, `completed`, `blocked`;
- registry decision: `pending`, `supported`, `rejected`, `inconclusive`;
- knowledge: `observed`, `supported`, `weakened`, `falsified`, `open`;
- architecture: `baseline`, `historical`, `planned`, `conditional`, `rejected`,
  `promoted`.

All negative results are retained. Rejected branches are not deleted; a new
experiment may reopen one only with new evidence and a new hypothesis ID.
