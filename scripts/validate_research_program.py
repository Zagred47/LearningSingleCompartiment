#!/usr/bin/env python3
"""Validate cross-references in the research operating system."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


VALID_EXPERIMENT_STATUS = {"planned", "preregistered", "ready", "blocked", "running", "completed"}
VALID_DECISIONS = {"pending", "supported", "rejected", "inconclusive"}


def _json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _csv(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _duplicates(values):
    seen, duplicates = set(), set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return sorted(duplicates)


def validate(root: Path) -> list[str]:
    research = root / "research"
    errors: list[str] = []
    graph = _json(research / "hypothesis_graph.json")
    ledger = _json(research / "knowledge_ledger.json")
    plan = _json(research / "master_plan.json")
    architectures = _json(research / "architecture_catalog.json")
    indicators = _json(research / "indicator_catalog.json")
    methodology_sources = _json(research / "methodology_source_index.json")
    methodology = _json(research / "creative_methodology.json")
    bias_taxonomy = _json(research / "inductive_bias_taxonomy.json")
    patterns = _json(research / "design_pattern_library.json")
    evaluation = _json(research / "evaluation_taxonomy.json")
    registry = _csv(research / "experiment_registry.csv")
    literature = _csv(research / "literature_evidence.csv")

    node_ids = [n["id"] for n in graph["nodes"]]
    knowledge_ids = [k["id"] for k in ledger["entries"]]
    experiment_ids = [r["experiment_id"] for r in registry]
    evidence_ids = [r["evidence_id"] for r in literature]
    architecture_ids = [a["id"] for a in architectures["architectures"]]
    indicator_ids = [i["id"] for i in indicators["indicators"]]
    source_ids = [s["id"] for s in methodology_sources["sources"]]
    pattern_ids = [p["id"] for p in patterns["patterns"]]
    method_ids = [a["id"] for a in methodology["axioms"]] + [m["id"] for m in methodology["search_modes"]]
    evaluation_ids = [layer["id"] for layer in evaluation["layers"]]
    evaluation_dimension_ids = [dimension["id"] for layer in evaluation["layers"] for dimension in layer["dimensions"]]

    for label, values in {
        "graph node": node_ids,
        "knowledge": knowledge_ids,
        "experiment": experiment_ids,
        "evidence": evidence_ids,
        "architecture": architecture_ids,
        "indicator": indicator_ids,
        "methodology source": source_ids,
        "design pattern": pattern_ids,
        "method": method_ids,
        "evaluation layer": evaluation_ids,
        "evaluation dimension": evaluation_dimension_ids,
    }.items():
        for duplicate in _duplicates(values):
            errors.append(f"duplicate {label} id: {duplicate}")

    nodes = set(node_ids)
    knowledge = set(knowledge_ids)
    experiments = set(experiment_ids)
    evidence = set(evidence_ids)
    hypotheses = {n["id"] for n in graph["nodes"] if n["type"] == "hypothesis"}

    for edge in graph["edges"]:
        if edge["source"] not in nodes:
            errors.append(f"edge source missing: {edge['source']}")
        if edge["target"] not in nodes:
            errors.append(f"edge target missing: {edge['target']}")
        if edge["type"] not in graph["allowed_edge_types"]:
            errors.append(f"invalid edge type: {edge['type']}")
    for node in graph["nodes"]:
        if node["type"] not in graph["allowed_node_types"]:
            errors.append(f"invalid node type: {node['id']}={node['type']}")
        for kid in node.get("knowledge_ids", []):
            if kid not in knowledge:
                errors.append(f"graph {node['id']} references missing knowledge {kid}")

    for entry in ledger["entries"]:
        for hid in entry.get("updates_hypotheses", []):
            if hid not in hypotheses:
                errors.append(f"knowledge {entry['id']} references missing hypothesis {hid}")

    for row in registry:
        if row["status"] not in VALID_EXPERIMENT_STATUS:
            errors.append(f"invalid experiment status: {row['experiment_id']}={row['status']}")
        if row["decision"] not in VALID_DECISIONS:
            errors.append(f"invalid decision: {row['experiment_id']}={row['decision']}")
        for eid in filter(None, row["evidence_id"].split(";")):
            if eid != "TBD" and eid not in evidence:
                errors.append(f"experiment {row['experiment_id']} references missing evidence {eid}")

    for phase in plan["phases"]:
        for experiment_id in phase["experiment_ids"]:
            if experiment_id not in experiments:
                errors.append(f"phase {phase['phase_id']} references missing experiment {experiment_id}")

    for architecture in architectures["architectures"]:
        for source in architecture.get("source_ids", []):
            if source not in evidence:
                errors.append(f"architecture {architecture['id']} references missing evidence {source}")
        for experiment_id in architecture.get("experiment_ids", []):
            if experiment_id not in experiments:
                errors.append(f"architecture {architecture['id']} references missing experiment {experiment_id}")

    source_set = set(source_ids)
    for pattern in patterns["patterns"]:
        for source in pattern.get("sources", []):
            if source not in source_set:
                errors.append(f"design pattern {pattern['id']} references missing methodology source {source}")
    if bias_taxonomy.get("source_id") not in source_set:
        errors.append(f"bias taxonomy references missing methodology source {bias_taxonomy.get('source_id')}")
    for source in methodology_sources["sources"]:
        canonical = source.get("canonical_source")
        if canonical and canonical not in source_set:
            errors.append(f"methodology source {source['id']} references missing canonical source {canonical}")
    for level in bias_taxonomy["levels"]:
        if not level.get("families"):
            errors.append(f"bias taxonomy level {level.get('level')} has no families")
        for family in level.get("families", []):
            if not family.get("members"):
                errors.append(f"bias taxonomy family {family.get('family')} has no members")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    errors = validate(args.root.resolve())
    if errors:
        print("Research program validation FAILED:")
        for error in errors:
            print(f"- {error}")
        return 1
    print("Research program validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
