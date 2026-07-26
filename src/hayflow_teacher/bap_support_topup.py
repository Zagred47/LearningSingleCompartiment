"""Outcome-blind BAP support supplement for the immutable v1.1 base shard."""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from ..hayflow_data import (
    BAP_SUPPORT_TOPUP_EPISODE_COUNT,
    BAP_SUPPORT_TOPUP_SCHEMA_VERSION,
    ProtocolTrajectory,
    TARGETED_DATASET_SCHEMA_VERSION,
    build_bap_validation_topup_plan,
    select_bap_positive_recipe,
    select_disjoint_topup_seed_start,
    summarize_independent_support,
    validate_composite_support,
    validate_hdf5_store,
    validate_minimum_support,
    write_json,
)
from .audit import sha256_file
from .audit_runtime import PINNED_TEACHER_COMMIT
from .diagnostic_dataset import DiagnosticDatasetSession
from .diagnostic_dataset_v1 import canonical_json_sha256
from .diagnostic_dataset_v1_1 import TargetedDiagnosticDatasetSession


BASE_REQUIRED_ARTIFACTS = (
    "transition_dataset.h5",
    "dataset_manifest.json",
    "validation_report.json",
    "dataset_card.json",
    "planning_budget_report.json",
    "targeted_preflight_report.json",
    "state_schema.json",
    "episodes.parquet",
    "events.parquet",
    "protocols.parquet",
    "transition_index.parquet",
    "release_outcomes.parquet",
    "splits.json",
    "snapshot_bank.json",
    "targeted_pilot/recipe_catalog.json",
    "snapshots/equilibrium_snapshot.neuron.bin",
    "snapshots/equilibrium_snapshot.rng.json",
    "snapshots/equilibrium_snapshot.named_state.npz",
    "snapshots/equilibrium_snapshot.metadata.json",
    "burnin_report.json",
)


def _artifact_records(root: Path) -> Dict[str, Dict[str, Any]]:
    document = json.loads((root / "artifact_index.json").read_text(encoding="utf-8"))
    return {str(row["path"]): dict(row) for row in document["artifacts"]}


def _sha256_with_progress(path: Path, *, label: str) -> str:
    size = path.stat().st_size
    completed = 0
    digest = hashlib.sha256()
    started = time.perf_counter()
    last = started
    print(f"[HayFlow][SHA-256] {label}: {size / 2**30:.3f} GiB", flush=True)
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
            completed += len(block)
            now = time.perf_counter()
            if now - last >= 10 or completed == size:
                rate = completed / max(now - started, 1e-9)
                eta = (size - completed) / max(rate, 1e-9)
                print(
                    f"[HayFlow][SHA-256] {100 * completed / max(size, 1):5.1f}% "
                    f"| {rate / 2**20:.1f} MiB/s | ETA {eta / 60:.1f} min",
                    flush=True,
                )
                last = now
    return digest.hexdigest()


class BapValidationSupportTopupSession(TargetedDiagnosticDatasetSession):
    """Generate and validate a separate eight-episode validation shard."""

    def __init__(self, *args: Any, base_dataset: Path, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.base_dataset = Path(base_dataset).resolve()
        self.base_verification: Dict[str, Any] = {}
        self.topup_plan: Dict[str, Any] = {}

    @staticmethod
    def _normalized_episodes(frame: Any) -> List[Dict[str, Any]]:
        rows = []
        for raw in frame.to_dict("records"):
            row = dict(raw)
            for name in ("event_labels", "hard_negative_for"):
                row[name] = TargetedDiagnosticDatasetSession._parquet_json_list(
                    row.get(name)
                )
            rows.append(row)
        return rows

    def verify_base_dataset(self, *, verify_large_hdf: bool = True) -> Dict[str, Any]:
        """Verify the Quick-Saved shard and its completed replay attestation."""

        root = self.base_dataset
        if root == self.output_dir or self.output_dir.is_relative_to(root):
            raise RuntimeError("top-up output must be separate from the base dataset")
        if not (root / "artifact_index.json").is_file():
            raise RuntimeError("base artifact_index.json is missing")
        records = _artifact_records(root)
        failures = []
        for relative in BASE_REQUIRED_ARTIFACTS:
            path = root / relative
            record = records.get(relative)
            if record is None:
                failures.append({"path": relative, "reason": "not indexed"})
            elif not path.is_file():
                failures.append({"path": relative, "reason": "missing"})
            elif path.stat().st_size != int(record["size_bytes"]):
                failures.append({"path": relative, "reason": "size mismatch"})
            elif relative != "transition_dataset.h5" and sha256_file(path) != str(
                record["sha256"]
            ):
                failures.append({"path": relative, "reason": "SHA-256 mismatch"})
        hdf_record = records.get("transition_dataset.h5", {})
        hdf_sha = str(hdf_record.get("sha256", ""))
        if not failures and verify_large_hdf:
            observed = _sha256_with_progress(
                root / "transition_dataset.h5", label="shard base"
            )
            if observed != hdf_sha:
                failures.append(
                    {"path": "transition_dataset.h5", "reason": "SHA-256 mismatch"}
                )

        validation = json.loads(
            (root / "validation_report.json").read_text(encoding="utf-8")
        )
        replay = dict(validation.get("exhaustive_replay", {}))
        replay_valid = bool(
            replay.get("valid")
            and int(replay.get("replayed_transition_count", -1)) == 29_240
            and int(replay.get("failure_count", -1)) == 0
            and not replay.get("failures")
            and float(replay.get("maximum_error", float("inf"))) <= 1e-5
            and float(replay.get("tolerance", float("inf"))) <= 1e-5
        )
        state_schema = json.loads(
            (root / "state_schema.json").read_text(encoding="utf-8")
        )
        teacher_commit = str(
            validation.get("teacher_commit")
            or json.loads(
                (root / "dataset_manifest.json").read_text(encoding="utf-8")
            ).get("teacher_commit")
        )
        if teacher_commit != PINNED_TEACHER_COMMIT:
            failures.append({"reason": "base teacher commit mismatch"})
        if state_schema.get("schema_version") != TARGETED_DATASET_SCHEMA_VERSION:
            failures.append({"reason": "base schema version mismatch"})
        if not replay_valid:
            failures.append({"reason": "base exhaustive replay proof is invalid"})

        base_episodes = self._normalized_episodes(
            self.pd.read_parquet(root / "episodes.parquet")
        )
        base_support = summarize_independent_support(base_episodes)
        budget = json.loads(
            (root / "planning_budget_report.json").read_text(encoding="utf-8")
        )
        base_minimum = validate_minimum_support(
            base_support,
            positive_targets=budget["minimum_positive_targets"],
            hard_negative_targets=budget["minimum_hard_negative_targets"],
        )
        expected_failure = {
            "event_class": "backpropagating_ap",
            "split": "validation",
            "positive": 3,
            "positive_target": 4,
            "hard_negative": 15,
            "hard_negative_target": 8,
        }
        if base_minimum.get("failures") != [expected_failure]:
            failures.append(
                {
                    "reason": "base support gap is not the expected single BAP shortage",
                    "observed": base_minimum.get("failures"),
                }
            )

        report = {
            "schema_version": BAP_SUPPORT_TOPUP_SCHEMA_VERSION,
            "valid": not failures,
            "failures": failures,
            "root": str(root),
            "transition_store_record": hdf_record,
            "artifact_index_sha256": sha256_file(root / "artifact_index.json"),
            "validation_report_sha256": sha256_file(
                root / "validation_report.json"
            ),
            "exhaustive_replay": replay,
            "minimum_support_validation": base_minimum,
            "teacher_commit": teacher_commit,
            "state_schema_version": state_schema.get("schema_version"),
            "large_hdf_sha256_verified": bool(verify_large_hdf and not failures),
        }
        write_json(self.output_dir / "base_verification_report.json", report)
        if failures:
            raise RuntimeError(f"base dataset verification failed: {failures}")
        self.base_verification = report
        return report

    def import_base_equilibrium(self) -> Dict[str, Any]:
        """Copy only the equilibrated initial state into the independent shard."""

        if not self.base_verification.get("valid"):
            raise RuntimeError("verify_base_dataset() must pass first")
        records = _artifact_records(self.base_dataset)
        copied = []
        for relative in (
            "snapshots/equilibrium_snapshot.neuron.bin",
            "snapshots/equilibrium_snapshot.rng.json",
            "snapshots/equilibrium_snapshot.named_state.npz",
            "snapshots/equilibrium_snapshot.metadata.json",
            "burnin_report.json",
        ):
            source = self.base_dataset / relative
            target = self.output_dir / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            observed = sha256_file(target)
            if observed != str(records[relative]["sha256"]):
                raise RuntimeError(f"copied equilibrium artifact changed: {relative}")
            copied.append({"path": relative, "sha256": observed})
        self.burnin_report = json.loads(
            (self.output_dir / "burnin_report.json").read_text(encoding="utf-8")
        )
        report = {"valid": True, "source": str(self.base_dataset), "copied": copied}
        write_json(self.output_dir / "equilibrium_import_report.json", report)
        return report

    def build_topup_plan(self) -> Tuple[List[ProtocolTrajectory], Dict[str, Any]]:
        """Persist the fixed acquisition plan before any outcome is observed."""

        catalog = json.loads(
            (self.base_dataset / "targeted_pilot/recipe_catalog.json").read_text(
                encoding="utf-8"
            )
        )
        recipe = select_bap_positive_recipe(catalog)
        base_episodes = self._normalized_episodes(
            self.pd.read_parquet(self.base_dataset / "episodes.parquet")
        )
        seed_selection = select_disjoint_topup_seed_start(base_episodes)
        protocols, plan = build_bap_validation_topup_plan(
            recipe, seed_start=seed_selection["seed_start"]
        )
        plan.update(
            {
                "base_transition_store_sha256": self.base_verification[
                    "transition_store_record"
                ]["sha256"],
                "teacher_commit": PINNED_TEACHER_COMMIT,
                "seed_selection": seed_selection,
            }
        )
        plan["topup_contract_sha256"] = canonical_json_sha256({"plan": plan})
        write_json(self.output_dir / "topup_plan.json", plan)
        self.topup_plan = plan
        self._bind_protocol_registry(protocols)
        return protocols, plan

    def generate_topup(
        self, protocols: Sequence[ProtocolTrajectory]
    ) -> Dict[str, Any]:
        """Generate every preregistered episode and no others."""

        protocols = list(protocols)
        if len(protocols) != BAP_SUPPORT_TOPUP_EPISODE_COUNT:
            raise RuntimeError("the top-up must contain exactly eight episodes")
        persisted = json.loads(
            (self.output_dir / "topup_plan.json").read_text(encoding="utf-8")
        )
        if persisted != self.topup_plan:
            raise RuntimeError("the persisted top-up plan changed before generation")
        required_snapshots = {
            str(row.metadata["snapshot_id"]) for row in protocols
        }
        if required_snapshots != set(self.snapshot_bank):
            raise RuntimeError("snapshot bank does not match the preregistered plan")

        zero_targets = {"validation": 0}
        write_json(
            self.output_dir / "planning_budget_report.json",
            {
                "schema_version": BAP_SUPPORT_TOPUP_SCHEMA_VERSION,
                "role": "standalone shard table construction only",
                "effective_positive_targets": zero_targets,
                "effective_hard_negative_targets": zero_targets,
                "minimum_positive_targets": zero_targets,
                "minimum_hard_negative_targets": zero_targets,
            },
        )
        self.targeted_preflight_report = {
            "valid": True,
            "protocol_plan_sha256": self.topup_plan["protocol_plan_sha256"],
            "policy": "fixed_batch_all_episodes_retained",
        }
        write_json(
            self.output_dir / "targeted_preflight_report.json",
            self.targeted_preflight_report,
        )
        self.release_rows = []
        self._bind_protocol_registry(protocols)
        self._collect_release_rows = True
        try:
            manifest = DiagnosticDatasetSession.generate_dataset(self, protocols)
        finally:
            self._collect_release_rows = False
        self.pd.DataFrame(self._parquet_safe_rows(self.release_rows)).to_parquet(
            self.output_dir / "release_outcomes.parquet", index=False
        )
        table_report = self._write_targeted_tables(protocols)
        manifest.update(
            {
                "schema_version": BAP_SUPPORT_TOPUP_SCHEMA_VERSION,
                "dataset_role": "validation_support_topup_shard",
                "compatible_base_schema_version": TARGETED_DATASET_SCHEMA_VERSION,
                "selection_policy": "fixed_batch_all_episodes_retained",
                "topup_plan": "topup_plan.json",
                "base_verification": "base_verification_report.json",
                "indices": {
                    "protocols": "protocols.parquet",
                    "episodes": "episodes.parquet",
                    "transitions": "transition_index.parquet",
                    "events": "events.parquet",
                    "release_outcomes": "release_outcomes.parquet",
                    "splits": "splits.json",
                },
                "table_report": table_report,
            }
        )
        self.dataset_manifest = manifest
        write_json(self.output_dir / "dataset_manifest.json", manifest)
        self._write_artifact_index()
        return manifest

    def validate_topup_and_composite(
        self, protocols: Sequence[ProtocolTrajectory]
    ) -> Dict[str, Any]:
        """Replay only the new shard and certify the logical two-shard union."""

        self._bind_protocol_registry(protocols)
        if not self.snapshot_bank:
            self.snapshot_bank = json.loads(
                (self.output_dir / "snapshot_bank.json").read_text(encoding="utf-8")
            )["snapshots"]
        structural = validate_hdf5_store(self.transition_path)
        topup_sha = _sha256_with_progress(self.transition_path, label="shard BAP")
        replay = self._exhaustive_sequential_replay()
        checkpoint = {
            "schema_version": BAP_SUPPORT_TOPUP_SCHEMA_VERSION,
            "checkpoint_kind": "complete_topup_exhaustive_replay",
            "transition_store_sha256": topup_sha,
            "protocol_plan_sha256": self.topup_plan["protocol_plan_sha256"],
            "topup_contract_sha256": self.topup_plan["topup_contract_sha256"],
            "exhaustive_replay": replay,
        }
        write_json(self.output_dir / "topup_replay_checkpoint.json", checkpoint)

        base_episodes = self._normalized_episodes(
            self.pd.read_parquet(self.base_dataset / "episodes.parquet")
        )
        topup_episodes = self._normalized_episodes(
            self.pd.read_parquet(self.output_dir / "episodes.parquet")
        )
        budget = json.loads(
            (self.base_dataset / "planning_budget_report.json").read_text(
                encoding="utf-8"
            )
        )
        composite_support = validate_composite_support(
            base_episodes,
            topup_episodes,
            minimum_positive_targets=budget["minimum_positive_targets"],
            minimum_hard_negative_targets=budget[
                "minimum_hard_negative_targets"
            ],
        )
        blockers = []
        if not bool(structural.get("valid")):
            blockers.append("top-up HDF5 structural validation failed")
        if not replay.get("valid"):
            blockers.append("top-up exhaustive replay failed")
        if not composite_support.get("valid"):
            blockers.extend(composite_support.get("blockers", ()))
        report = {
            "schema_version": BAP_SUPPORT_TOPUP_SCHEMA_VERSION,
            "valid": not blockers,
            "blockers": blockers,
            "teacher_commit": PINNED_TEACHER_COMMIT,
            "base": self.base_verification,
            "topup": {
                "transition_store_sha256": topup_sha,
                "structural": structural,
                "exhaustive_replay": replay,
                "episode_count": len(topup_episodes),
                "transition_count": sum(int(row.duration_ms) for row in protocols),
            },
            "composite_support": composite_support,
        }
        composite_manifest = {
            "schema_version": BAP_SUPPORT_TOPUP_SCHEMA_VERSION,
            "dataset_kind": "logical_composite_two_shard_dataset",
            "physical_merge_performed": False,
            "base_shard": {
                "external_root": str(self.base_dataset),
                "transition_store_sha256": self.base_verification[
                    "transition_store_record"
                ]["sha256"],
                "transition_count": 29_240,
                "read_only": True,
            },
            "topup_shard": {
                "root": ".",
                "transition_store": "transition_dataset.h5",
                "transition_store_sha256": topup_sha,
                "transition_count": sum(int(row.duration_ms) for row in protocols),
            },
            "topup_plan_sha256": self.topup_plan["protocol_plan_sha256"],
            "topup_contract_sha256": self.topup_plan["topup_contract_sha256"],
            "validation_report": "validation_report.json",
            "valid": not blockers,
        }
        write_json(self.output_dir / "composite_dataset_manifest.json", composite_manifest)
        write_json(self.output_dir / "validation_report.json", report)
        self.dataset_manifest = composite_manifest
        self._write_artifact_index({"transition_dataset.h5": topup_sha})
        if blockers:
            raise RuntimeError(f"BAP support top-up validation failed: {blockers}")
        return report
