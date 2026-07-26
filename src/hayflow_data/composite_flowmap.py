"""Shard-aware loader for the validated HayFlow targeted v1.1 composite.

The module treats ``composite_dataset_manifest.json`` as the sole authority.
It never merges the two HDF5 stores and it never reassigns an episode split.
Only small metadata arrays live in memory; boundary states and microtraces are
read lazily from their original shard.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .flowmap_dataset import (
    DYNAMIC_CATEGORIES,
    EVENT_KINDS,
    FlowmapBundle,
    FlowmapContractError,
    FlowmapLayout,
    _decode,
)


EXPECTED_BASE_HDF_SHA256 = (
    "3fef415544a82b55801461e3cec069ed292faca0075f1f9f431e9dce8f5ea6d8"
)
EXPECTED_TOPUP_HDF_SHA256 = (
    "1274ad51f4aa3244fad4cc2e02f69004fc659f4436d610b65536416020e0fcd2"
)
EXPECTED_PROTOCOL_PLAN_SHA256 = (
    "45c38fae2947ab58c15a3a4308e190cca8e7ed1ff94d6e420a562720314c96eb"
)
EXPECTED_TOPUP_CONTRACT_SHA256 = (
    "bd88d61cecfad861f8d8a1a74ebacf6d34e8f6714a3ba2e7621c89ee05a8515f"
)
EXPECTED_TEACHER_COMMIT = "074c4666300a8ad246601dab179a97a6942f0f29"
EXPECTED_EPISODES = 369
EXPECTED_TRANSITIONS = 29_880
INPUT_VIEWS = ("U_scheduled", "U_rng", "U_realized")

# No stream id, global Random123 index, seed, trajectory id, or snapshot id is
# exposed as a model feature.  U_rng uses the causal draw/probability/sequence
# state, which explains the release decision without providing a memorization
# key for an episode.
INPUT_EVENT_FEATURE_NAMES: Tuple[str, ...] = (
    "offset_ms",
    "weight",
    "is_excitatory",
    "is_inhibitory",
    "is_somatic_current",
    "gmax",
    "release_probability",
    "rng_preview_value",
    "rng_sequence_fraction",
    "release_success",
    "released_quantity",
    "ampa_state_increment",
    "nmda_state_increment",
    "inhibitory_state_increment",
    "current_amplitude_na",
    "current_duration_ms",
)


def _json(path: Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha256(path: Path, progress: Optional[Any] = None) -> str:
    digest = hashlib.sha256()
    total = max(1, path.stat().st_size)
    done = 0
    with path.open("rb") as handle:
        while True:
            block = handle.read(16 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
            done += len(block)
            if progress is not None:
                progress(path.name, done, total)
    return digest.hexdigest()


def _safe_extract(source: Path, destination: Path, wanted: Optional[set] = None) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(source) as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            relative = Path(member.filename)
            if wanted is not None and relative.name not in wanted:
                continue
            target = (destination / relative).resolve()
            if root not in target.parents:
                raise FlowmapContractError(f"unsafe ZIP member {member.filename!r}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst, length=16 * 1024 * 1024)


def _find_unique(root: Path, name: str) -> Path:
    direct = root / name
    if direct.is_file():
        return direct
    matches = list(root.rglob(name))
    if len(matches) != 1:
        raise FlowmapContractError(
            f"expected exactly one {name} below {root}; found {len(matches)}"
        )
    return matches[0]


def _materialize_source(
    source: Path,
    cache_dir: Path,
    *,
    required_names: Sequence[str],
) -> Path:
    source = Path(source).expanduser().resolve()
    if source.is_dir():
        # KaggleHub may mount a single archive.zip inside the dataset folder.
        if list(source.rglob("transition_dataset.h5")):
            return _find_unique(source, "transition_dataset.h5").parent
        archives = list(source.rglob("*.zip"))
        if len(archives) == 1:
            source = archives[0]
        else:
            raise FlowmapContractError(
                f"no transition_dataset.h5 and no unique ZIP below {source}"
            )
    if not source.is_file() or source.suffix.lower() != ".zip":
        raise FlowmapContractError(f"unsupported shard source {source}")
    marker = cache_dir / ".source.json"
    stamp = {"path": str(source), "size": source.stat().st_size, "mtime": source.stat().st_mtime_ns}
    if not marker.is_file() or _json(marker) != stamp:
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
        _safe_extract(source, cache_dir, set(required_names))
        cache_dir.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps(stamp, sort_keys=True), encoding="utf-8")
    return _find_unique(cache_dir, "transition_dataset.h5").parent


@dataclass(frozen=True)
class CompositeShard:
    shard_id: str
    root: Path
    transition_path: Path
    transition_count: int
    transition_sha256: str
    dataset_manifest: Mapping[str, Any]
    validation_report: Mapping[str, Any]
    offset: int


@dataclass(frozen=True)
class CompositeFlowmapBundle:
    manifest_path: Path
    manifest: Mapping[str, Any]
    base: CompositeShard
    topup: CompositeShard
    layout_bundle: FlowmapBundle
    fingerprint: str

    @property
    def shards(self) -> Tuple[CompositeShard, CompositeShard]:
        return self.base, self.topup

    @property
    def transition_count(self) -> int:
        return sum(row.transition_count for row in self.shards)


def _layout_bundle(root: Path, shard: CompositeShard) -> FlowmapBundle:
    state_schema = _json(root / "state_schema.json")
    teacher_path = root / "manifest.json"
    if not teacher_path.is_file():
        teacher_path = _find_unique(root, "teacher_manifest.json")
    return FlowmapBundle(
        root=root,
        transition_path=shard.transition_path,
        manifest=shard.dataset_manifest,
        state_schema=state_schema,
        teacher_manifest=_json(teacher_path),
        validation_report=shard.validation_report,
        artifact_validation={"valid": True, "composite_loader": True},
    )


def prepare_composite_flowmap_bundle(
    composite_manifest: Path,
    *,
    base_source: Optional[Path] = None,
    cache_dir: Optional[Path] = None,
    verify_hashes: bool = True,
    progress: Optional[Any] = None,
) -> CompositeFlowmapBundle:
    """Resolve and validate both shards from the composite authority file."""

    manifest_path = Path(composite_manifest).expanduser().resolve()
    manifest = _json(manifest_path)
    expected = {
        "base": (EXPECTED_BASE_HDF_SHA256, int(manifest["base_shard"]["transition_count"])),
        "topup": (EXPECTED_TOPUP_HDF_SHA256, int(manifest["topup_shard"]["transition_count"])),
    }
    if not manifest.get("valid") or manifest.get("physical_merge_performed"):
        raise FlowmapContractError("composite manifest is not a valid logical two-shard contract")
    if manifest["base_shard"].get("transition_store_sha256") != expected["base"][0]:
        raise FlowmapContractError("base SHA-256 differs from the preregistered contract")
    if manifest["topup_shard"].get("transition_store_sha256") != expected["topup"][0]:
        raise FlowmapContractError("top-up SHA-256 differs from the preregistered contract")
    if manifest.get("topup_plan_sha256") != EXPECTED_PROTOCOL_PLAN_SHA256:
        raise FlowmapContractError("protocol-plan fingerprint mismatch")
    if manifest.get("topup_contract_sha256") != EXPECTED_TOPUP_CONTRACT_SHA256:
        raise FlowmapContractError("top-up contract fingerprint mismatch")

    cache = Path(cache_dir or manifest_path.parent / ".composite_cache").resolve()
    topup_root = manifest_path.parent
    if not (topup_root / "transition_dataset.h5").is_file():
        topup_root = _find_unique(topup_root, "transition_dataset.h5").parent
    if base_source is None:
        base_source = Path(str(manifest["base_shard"]["external_root"]))
    required = (
        "transition_dataset.h5", "dataset_manifest.json", "state_schema.json",
        "manifest.json", "teacher_manifest.json", "validation_report.json",
        "event_definition_config.json", "episodes.parquet", "events.parquet",
        "branching_pairs.parquet", "release_outcomes.parquet", "segments.parquet",
        "synapses.parquet", "dataset_card.json", "splits.json",
    )
    base_root = _materialize_source(Path(base_source), cache / "base", required_names=required)

    def make_shard(shard_id: str, root: Path, offset: int) -> CompositeShard:
        transition = root / "transition_dataset.h5"
        dataset_manifest = _json(root / "dataset_manifest.json")
        validation = _json(root / "validation_report.json")
        sha, count = expected[shard_id]
        if int(dataset_manifest.get("transition_count", -1)) != count:
            raise FlowmapContractError(f"{shard_id} transition count mismatch")
        if str(dataset_manifest.get("teacher_commit")) != EXPECTED_TEACHER_COMMIT:
            raise FlowmapContractError(f"{shard_id} teacher commit mismatch")
        observed = _sha256(transition, progress) if verify_hashes else sha
        if observed != sha:
            raise FlowmapContractError(f"{shard_id} HDF SHA-256 mismatch")
        return CompositeShard(
            shard_id=shard_id, root=root, transition_path=transition,
            transition_count=count, transition_sha256=observed,
            dataset_manifest=dataset_manifest, validation_report=validation,
            offset=offset,
        )

    base = make_shard("base", base_root, 0)
    topup = make_shard("topup", topup_root, base.transition_count)
    base_layout = _layout_bundle(base.root, base)
    topup_layout = _layout_bundle(topup.root, topup)
    base_schema = json.dumps(base_layout.state_schema, sort_keys=True, separators=(",", ":"))
    topup_schema = json.dumps(topup_layout.state_schema, sort_keys=True, separators=(",", ":"))
    if base_schema != topup_schema:
        raise FlowmapContractError("base and top-up state schemas differ")
    if int(base_layout.state_schema.get("core_state_width", -1)) != 17_220:
        raise FlowmapContractError("core state width differs from the audited 17,220")
    if int(base_layout.state_schema.get("privileged_state_width", -1)) != 9_182:
        raise FlowmapContractError("privileged state width differs from the audited 9,182")
    digest = hashlib.sha256()
    digest.update(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode())
    digest.update(base.transition_sha256.encode())
    digest.update(topup.transition_sha256.encode())
    return CompositeFlowmapBundle(
        manifest_path=manifest_path, manifest=manifest, base=base, topup=topup,
        layout_bundle=base_layout, fingerprint=digest.hexdigest(),
    )


def _canonical_release_rows(
    scheduled: Sequence[Mapping[str, Any]],
    releases: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Join all scheduled events to causal outcomes, including failures."""

    by_event = {int(row["event_index"]): dict(row) for row in releases}
    result: List[Dict[str, Any]] = []
    event_index = 0
    for raw in scheduled:
        action = dict(raw)
        if action.get("kind") != "synaptic_event":
            result.append(action)
            continue
        if event_index not in by_event:
            raise FlowmapContractError(f"missing causal outcome for event {event_index}")
        outcome = by_event[event_index]
        if int(outcome["synapse_id"]) != int(action["synapse_id"]):
            raise FlowmapContractError("release outcome is misaligned with scheduled input")
        action.update(
            {
                key: outcome[key]
                for key in (
                    "release_success", "released_quantity", "ampa_state_increment",
                    "nmda_state_increment", "inhibitory_state_increment",
                    "release_probability", "rng_preview_value", "rng_sequence_before",
                )
            }
        )
        result.append(action)
        event_index += 1
    if event_index != len(releases):
        raise FlowmapContractError("release table has unmatched outcomes")
    return result


class CompositeTransitionStore:
    """One logical lazy store backed by two immutable HDF5 shards."""

    def __init__(self, bundle: CompositeFlowmapBundle) -> None:
        try:
            import h5py
        except ImportError as error:  # pragma: no cover - Kaggle supplies it.
            raise RuntimeError("composite loading requires h5py") from error
        self.h5py = h5py
        self.bundle = bundle
        self.layout = ReleaseFlowmapLayout(bundle.layout_bundle)
        self.count = bundle.transition_count
        self._handles: Dict[str, Any] = {}
        metadata: Dict[str, List[np.ndarray]] = {}
        self.shard_id = np.empty(self.count, dtype=object)
        self.local_index = np.empty(self.count, dtype=np.int64)
        for shard in bundle.shards:
            sl = slice(shard.offset, shard.offset + shard.transition_count)
            self.shard_id[sl] = shard.shard_id
            self.local_index[sl] = np.arange(shard.transition_count)
            with h5py.File(shard.transition_path, "r") as handle:
                if int(handle.attrs["transition_count"]) != shard.transition_count:
                    raise FlowmapContractError(f"{shard.shard_id} HDF row count mismatch")
                for name in handle["metadata"]:
                    values = handle[f"metadata/{name}"][...]
                    if values.dtype.kind in {"S", "O", "U"}:
                        values = np.asarray([_decode(value) for value in values], dtype=object)
                    metadata.setdefault(name, []).append(values)
        self.metadata = {name: np.concatenate(parts) for name, parts in metadata.items()}
        self.split_indices = {
            split: np.flatnonzero(self.metadata["split"] == split)
            for split in sorted(set(self.metadata["split"].tolist()))
        }
        test_splits = [name for name in self.split_indices if name not in {"train", "validation"}]
        self.split_indices["test"] = np.sort(
            np.concatenate([self.split_indices[name] for name in test_splits])
        ) if test_splits else np.empty(0, dtype=np.int64)
        self.trajectory_indices: Dict[str, np.ndarray] = {}
        for trajectory in sorted(set(self.metadata["trajectory_id"].tolist())):
            indices = np.flatnonzero(self.metadata["trajectory_id"] == trajectory)
            self.trajectory_indices[str(trajectory)] = indices[np.argsort(self.metadata["step_index"][indices])]
        self.episode_rows = self._load_episode_rows()
        self.episode_by_trajectory = {str(row["trajectory_id"]): row for row in self.episode_rows}
        self._report_cache: Optional[Dict[str, Any]] = None
        self._validate_contract()

    def _handle(self, shard: CompositeShard) -> Any:
        if shard.shard_id not in self._handles:
            self._handles[shard.shard_id] = self.h5py.File(shard.transition_path, "r")
        return self._handles[shard.shard_id]

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()

    def __del__(self) -> None:
        self.close()

    def _load_episode_rows(self) -> List[Dict[str, Any]]:
        import pandas as pd
        rows: List[Dict[str, Any]] = []
        for shard in self.bundle.shards:
            table = pd.read_parquet(shard.root / "episodes.parquet")
            for row in table.to_dict("records"):
                clean = dict(row)
                clean["shard_id"] = shard.shard_id
                rows.append(clean)
        return rows

    @staticmethod
    def _hashable(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, float) and np.isnan(value):
            return ""
        return str(value)

    def _validate_contract(self) -> None:
        blockers: List[str] = []
        if self.count != EXPECTED_TRANSITIONS:
            blockers.append("logical transition count is not 29,880")
        if len(self.episode_rows) != EXPECTED_EPISODES:
            blockers.append("logical episode count is not 369")
        topup = [row for row in self.episode_rows if row["shard_id"] == "topup"]
        if len(topup) != 8 or {str(row["split"]) for row in topup} != {"validation"}:
            blockers.append("the eight top-up episodes are not validation-only")
        memberships: Dict[str, set] = {}
        for row in self.episode_rows:
            memberships.setdefault(str(row["trajectory_id"]), set()).add(str(row["split"]))
        split_overlap = {
            key: sorted(value) for key, value in memberships.items() if len(value) != 1
        }
        if split_overlap:
            blockers.append("episode split overlap detected")
        base = [row for row in self.episode_rows if row["shard_id"] == "base"]
        cross_shard_overlap: Dict[str, List[str]] = {}
        for key in ("seed", "snapshot_id"):
            base_values = {self._hashable(row.get(key)) for row in base}
            topup_values = {self._hashable(row.get(key)) for row in topup}
            base_values.discard("")
            topup_values.discard("")
            overlap = sorted(base_values & topup_values)
            cross_shard_overlap[key] = overlap
            if overlap:
                blockers.append(f"base/top-up {key} overlap detected")
        self.contract_checks = {
            "episode_count_exact": len(self.episode_rows) == EXPECTED_EPISODES,
            "transition_count_exact": self.count == EXPECTED_TRANSITIONS,
            "episode_split_overlap": split_overlap,
            "topup_episode_count": len(topup),
            "topup_splits": sorted({str(row["split"]) for row in topup}),
            "topup_validation_only": len(topup) == 8
            and {str(row["split"]) for row in topup} == {"validation"},
            "cross_shard_seed_overlap": cross_shard_overlap.get("seed", []),
            "cross_shard_snapshot_overlap": cross_shard_overlap.get("snapshot_id", []),
            "blockers": blockers,
        }
        if blockers:
            raise FlowmapContractError(f"composite logical contract failed: {blockers}")

    def report(self) -> Dict[str, Any]:
        if self._report_cache is not None:
            return dict(self._report_cache)
        event_counts: Dict[str, int] = {kind: 0 for kind in EVENT_KINDS}
        for index in range(self.count):
            for row in self.events(index):
                kind = str(row.get("kind"))
                if kind in event_counts:
                    event_counts[kind] += 1
        declared: Dict[str, int] = {kind: 0 for kind in EVENT_KINDS}
        for shard in self.bundle.shards:
            for kind, count in shard.dataset_manifest.get("event_counts", {}).items():
                if kind in declared:
                    declared[kind] += int(count)
        coherent = event_counts == declared
        if not coherent:
            raise FlowmapContractError("HDF event counts differ from shard manifests")
        report = {
            "valid": True,
            "dataset_kind": "logical_composite_two_shard_dataset",
            "physical_merge_performed": False,
            "fingerprint": self.bundle.fingerprint,
            "episode_count": len(self.episode_rows),
            "transition_count": self.count,
            "shards": [
                {
                    "shard_id": row.shard_id, "transition_count": row.transition_count,
                    "transition_store_sha256": row.transition_sha256,
                    "split_counts": {
                        split: int(np.sum(self.metadata["split"][row.offset:row.offset + row.transition_count] == split))
                        for split in sorted(set(self.metadata["split"][row.offset:row.offset + row.transition_count].tolist()))
                    },
                }
                for row in self.bundle.shards
            ],
            "event_counts": event_counts,
            "event_counts_coherent": coherent,
            "topup_validation_only": True,
            "contract_checks": self.contract_checks,
            "state_loading": "lazy_per_shard_no_physical_merge",
        }
        self._report_cache = report
        return dict(report)

    def logical_index(self) -> List[Dict[str, Any]]:
        names = ("trajectory_id", "split", "seed", "step_index", "protocol", "protocol_variant")
        return [
            {
                "logical_index": index,
                "shard_id": str(self.shard_id[index]),
                "local_index": int(self.local_index[index]),
                **{name: self.metadata[name][index].item() if hasattr(self.metadata[name][index], "item") else self.metadata[name][index] for name in names},
            }
            for index in range(self.count)
        ]

    def _shard_for(self, logical_index: int) -> CompositeShard:
        if not 0 <= int(logical_index) < self.count:
            raise IndexError(logical_index)
        return self.bundle.base if int(logical_index) < self.bundle.topup.offset else self.bundle.topup

    def _json_row(self, index: int, path: str) -> Any:
        shard = self._shard_for(index)
        local = int(index) - shard.offset
        return json.loads(_decode(self._handle(shard)[path][local]))

    def actions(self, index: int, view: str = "U_scheduled") -> List[Dict[str, Any]]:
        if view not in INPUT_VIEWS:
            raise ValueError(f"unknown input view {view!r}")
        scheduled = self._json_row(index, "inputs/U_scheduled_json")
        for action in scheduled:
            if action.get("kind") == "synaptic_event":
                synapse_id = int(action["synapse_id"])
                action.update(
                    synapse_type=self.layout.synapse_type[synapse_id],
                    segment_id=self.layout.synapse_to_segment[synapse_id],
                    gmax=self.layout.synapse_gmax[synapse_id],
                )
        if view == "U_scheduled":
            return scheduled
        releases = self._json_row(index, "release_outcomes/records_json")
        realized = _canonical_release_rows(scheduled, releases)
        if view == "U_realized":
            return [
                {
                    key: value
                    for key, value in action.items()
                    if key
                    not in {
                        "release_probability",
                        "rng_preview_value",
                        "rng_sequence_before",
                    }
                }
                for action in realized
            ]
        # U_rng is deliberately causal but strips arbitrary Random123 keys.
        rng_rows: List[Dict[str, Any]] = []
        for action in realized:
            if action.get("kind") != "synaptic_event":
                rng_rows.append(dict(action))
                continue
            keep = {
                key: value
                for key, value in action.items()
                if key
                not in {
                    "release_success",
                    "released_quantity",
                    "ampa_state_increment",
                    "nmda_state_increment",
                    "inhibitory_state_increment",
                }
            }
            rng_rows.append(keep)
        return rng_rows

    def releases(self, index: int) -> List[Dict[str, Any]]:
        return self._json_row(index, "release_outcomes/records_json")

    def events(self, index: int) -> List[Dict[str, Any]]:
        return self._json_row(index, "events/labels_json")

    def read_state(self, indices: Sequence[int], boundary: str, categories: Sequence[str] = DYNAMIC_CATEGORIES) -> np.ndarray:
        if boundary not in {"t", "t_plus_1"}:
            raise ValueError("boundary must be t or t_plus_1")
        ordered = np.asarray(indices, dtype=np.int64)
        if not len(ordered):
            width = sum(self.layout.category_widths[name] for name in categories)
            return np.empty((0, width), dtype=np.float64)
        width = sum(self.layout.category_widths[name] for name in categories)
        output = np.empty((len(ordered), width), dtype=np.float64)
        for shard in self.bundle.shards:
            positions = np.flatnonzero(
                (ordered >= shard.offset)
                & (ordered < shard.offset + shard.transition_count)
            )
            if not len(positions):
                continue
            local = ordered[positions] - shard.offset
            order = np.argsort(local)
            sorted_local = local[order]
            if len(np.unique(sorted_local)) != len(sorted_local):
                values = np.stack(
                    [
                        np.concatenate(
                            [
                                self._handle(shard)[f"states/{category}/{boundary}"][int(value), :]
                                for category in categories
                            ]
                        )
                        for value in local
                    ]
                )
            else:
                sorted_values = np.concatenate(
                    [
                        self._handle(shard)[f"states/{category}/{boundary}"][sorted_local, :]
                        for category in categories
                    ],
                    axis=1,
                )
                inverse = np.empty_like(order)
                inverse[order] = np.arange(len(order))
                values = sorted_values[inverse]
            output[positions] = values
        return output

    def read_privileged(self, indices: Sequence[int]) -> np.ndarray:
        ordered = np.asarray(indices, dtype=np.int64)
        if not len(ordered):
            return np.empty((0, self.layout.privileged_width))
        output = np.empty((len(ordered), self.layout.privileged_width), dtype=np.float64)
        for shard in self.bundle.shards:
            positions = np.flatnonzero(
                (ordered >= shard.offset)
                & (ordered < shard.offset + shard.transition_count)
            )
            if not len(positions):
                continue
            local = ordered[positions] - shard.offset
            order = np.argsort(local)
            sorted_local = local[order]
            dataset = self._handle(shard)["states/currents_conductances/t_plus_1"]
            if len(np.unique(sorted_local)) != len(sorted_local):
                values = np.stack([dataset[int(value), :] for value in local])
            else:
                sorted_values = dataset[sorted_local, :]
                inverse = np.empty_like(order)
                inverse[order] = np.arange(len(order))
                values = sorted_values[inverse]
            output[positions] = values
        return output

    def auxiliary_targets(self, indices: Sequence[int]) -> Tuple[np.ndarray, Dict[str, slice]]:
        """Selective P1 target: privileged currents/conductances only."""

        values = self.read_privileged(indices).astype(np.float32)
        layout = {"currents_conductances_t_plus_1": slice(0, values.shape[1])}
        self.layout.aux_layout = layout
        return values, layout

    def microtrace(self, index: int, name: str = "all_segment_voltage") -> np.ndarray:
        shard = self._shard_for(index)
        return np.asarray(self._handle(shard)[f"microtraces/{name}"][int(index) - shard.offset])

    def encode_inputs(self, indices: Sequence[int], view: str) -> Dict[str, np.ndarray]:
        rows: List[List[List[float]]] = []
        owners: List[List[int]] = []
        max_events = 1
        for index in indices:
            values: List[List[float]] = []
            segments: List[int] = []
            for action in self.actions(int(index), view):
                if action.get("kind") == "somatic_current":
                    values.append([
                        float(action.get("offset_ms", 0.0)), 1.0, 0.0, 0.0, 1.0, 0.0,
                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                        float(action.get("amplitude_na") or 0.0), float(action.get("duration_ms") or 0.0),
                    ])
                    segments.append(0)
                    continue
                synapse_id = int(action["synapse_id"])
                excitatory = self.layout.synapse_type[synapse_id] == "ProbAMPANMDA2"
                sequence = float(action.get("rng_sequence_before", 0.0))
                values.append([
                    float(action.get("offset_ms", 0.0)), float(action.get("weight_multiplier", 1.0)),
                    float(excitatory), float(not excitatory), 0.0,
                    float(self.layout.synapse_gmax[synapse_id]),
                    float(action.get("release_probability", 0.0)) if view == "U_rng" else 0.0,
                    float(action.get("rng_preview_value", 0.0)) if view == "U_rng" else 0.0,
                    np.log1p(max(sequence, 0.0)) / 20.0 if view == "U_rng" else 0.0,
                    float(bool(action.get("release_success", False))) if view == "U_realized" else 0.0,
                    float(action.get("released_quantity", 0.0)) if view == "U_realized" else 0.0,
                    float(action.get("ampa_state_increment", 0.0)) if view == "U_realized" else 0.0,
                    float(action.get("nmda_state_increment", 0.0)) if view == "U_realized" else 0.0,
                    float(action.get("inhibitory_state_increment", 0.0)) if view == "U_realized" else 0.0,
                    0.0, 0.0,
                ])
                segments.append(self.layout.synapse_to_segment[synapse_id])
            rows.append(values)
            owners.append(segments)
            max_events = max(max_events, len(values))
        features = np.zeros((len(rows), max_events, len(INPUT_EVENT_FEATURE_NAMES)), dtype=np.float32)
        segment_ids = np.zeros((len(rows), max_events), dtype=np.int64)
        mask = np.zeros((len(rows), max_events), dtype=bool)
        for row, (values, segments) in enumerate(zip(rows, owners)):
            if values:
                features[row, :len(values)] = np.asarray(values, dtype=np.float32)
                segment_ids[row, :len(values)] = segments
                mask[row, :len(values)] = True
        # The existing B3 event encoder consumes these canonical keys.  Its
        # metadata width is replaced by ``ReleaseFlowmapLayout`` below.
        return {"u2_features": features, "u2_segment_ids": segment_ids, "u2_mask": mask}

    def event_targets(self, indices: Sequence[int]) -> Dict[str, np.ndarray]:
        kind_index = {name: i for i, name in enumerate(EVENT_KINDS)}
        region_index = {name: i for i, name in enumerate(self.layout.region_names)}
        presence = np.zeros((len(indices), len(EVENT_KINDS)), dtype=np.float32)
        timing = np.zeros((len(indices), len(EVENT_KINDS), 4), dtype=np.float32)
        timing_mask = np.zeros_like(timing, dtype=bool)
        region = np.zeros((len(indices), len(EVENT_KINDS)), dtype=np.int64)
        region_mask = np.zeros_like(region, dtype=bool)
        for row, index in enumerate(indices):
            start = float(self.metadata["start_time_ms"][int(index)])
            first: Dict[str, Mapping[str, Any]] = {}
            for event in self.events(int(index)):
                kind = str(event.get("kind"))
                if kind in kind_index and (kind not in first or float(event["onset_ms"]) < float(first[kind]["onset_ms"])):
                    first[kind] = event
            for kind, event in first.items():
                col = kind_index[kind]
                presence[row, col] = 1.0
                timing[row, col] = [float(event[key]) - start if key != "duration_ms" else float(event[key]) for key in ("onset_ms", "peak_ms", "offset_ms", "duration_ms")]
                timing_mask[row, col, :2] = True
                if not event.get("right_censored", False):
                    timing_mask[row, col, 2:] = True
                name = str(event.get("region", ""))
                if name in region_index:
                    region[row, col] = region_index[name]
                    region_mask[row, col] = True
        return {"event_presence": presence, "event_timing": timing, "event_timing_mask": timing_mask, "event_region": region, "event_region_mask": region_mask}

    def episode_indices(self, *, split: Optional[str] = None, regime: Optional[str] = None, event_kind: Optional[str] = None) -> List[np.ndarray]:
        result = []
        for trajectory, indices in self.trajectory_indices.items():
            row = self.episode_by_trajectory.get(trajectory, {})
            if split is not None and str(row.get("split", self.metadata["split"][indices[0]])) != split:
                continue
            labels = str(row.get("event_labels", ""))
            if event_kind is not None and event_kind not in labels:
                continue
            if regime is not None and regime != classify_regime(row, self.metadata["category"][indices[0]]):
                continue
            result.append(indices)
        return result

    def rollout_windows(self, split: str, horizon: int) -> List[np.ndarray]:
        windows: List[np.ndarray] = []
        accepted = set(self.split_indices) - {"train", "validation", "test"} if split == "test" else {split}
        for indices in self.trajectory_indices.values():
            if str(self.metadata["split"][indices[0]]) not in accepted:
                continue
            for start in range(max(0, len(indices) - int(horizon) + 1)):
                candidate = indices[start:start + int(horizon)]
                steps = self.metadata["step_index"][candidate]
                if np.array_equal(steps, np.arange(steps[0], steps[0] + int(horizon))):
                    windows.append(candidate)
        return windows


def classify_regime(episode: Mapping[str, Any], category: Any = "") -> str:
    split = str(episode.get("split", ""))
    labels = str(episode.get("event_labels", ""))
    hard = str(episode.get("hard_negative_for", ""))
    if split == "recovery_test":
        return "recovery"
    if split == "branching_near_test":
        return "branching_near"
    if split == "branching_far_test":
        return "branching_far"
    if hard and hard not in {"[]", "nan"}:
        return "hard_negative"
    for event in ("nmda_plateau", "nmda_spike", "calcium_spike", "backpropagating_ap", "somatic_spike", "axonal_spike"):
        if event in labels:
            return event
    return "rest_subthreshold" if "rest" in str(category) or not labels else "other"


def input_view_schema() -> Dict[str, Any]:
    return {
        "schema_version": "04-input-views-v1",
        "causal_boundary": "before membrane integration over [t,t+1ms]",
        "future_state_used": False,
        "model_event_feature_names": list(INPUT_EVENT_FEATURE_NAMES),
        "views": {
            "U_scheduled": {"contains": ["scheduled event", "offset", "synapse type", "weight", "position"], "release_known": False},
            "U_rng": {"contains": ["U_scheduled", "release probability", "causal RNG variate", "causal sequence phase"], "excluded_memorization_keys": ["random123_seed", "random123_stream_id", "random123_global_index", "episode_id", "trajectory_id", "snapshot_id"]},
            "U_realized": {"contains": ["U_scheduled", "success/failure", "released quantity", "AMPA/NMDA/inhibitory increments"], "construction": "join scheduled actions to exact causal release_outcomes; failures retained"},
        },
    }


class ReleaseFlowmapLayout(FlowmapLayout):
    """Flowmap layout whose event encoder width matches the three v1.1 views."""

    def to_model_metadata(self) -> Dict[str, Any]:
        result = super().to_model_metadata()
        result["u2_event_feature_names"] = list(INPUT_EVENT_FEATURE_NAMES)
        return result
