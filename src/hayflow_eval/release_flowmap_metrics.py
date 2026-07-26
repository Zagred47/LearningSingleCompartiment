"""Decision-grade metrics for release-aware HayFlow experiments."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..hayflow_data.flowmap_dataset import EVENT_KINDS


DENDRITIC_EVENTS = (
    "backpropagating_ap",
    "calcium_spike",
    "nmda_spike",
    "nmda_plateau",
)
SOMATIC_AXONAL_EVENTS = ("axonal_spike", "somatic_spike")


def _average_precision(target: np.ndarray, score: np.ndarray) -> float:
    target = np.asarray(target, dtype=bool)
    score = np.asarray(score, dtype=float)
    positives = int(target.sum())
    if positives == 0:
        return math.nan
    order = np.argsort(-score, kind="mergesort")
    sorted_target = target[order]
    cumulative = np.cumsum(sorted_target)
    precision = cumulative / np.arange(1, len(target) + 1)
    return float(np.sum(precision[sorted_target]) / positives)


def pooled_event_metrics(
    probabilities: np.ndarray,
    targets: np.ndarray,
    episode_ids: Sequence[str],
    *,
    model: str,
    split: str,
    thresholds: Any = 0.5,
    timing_prediction: Optional[np.ndarray] = None,
    timing_target: Optional[np.ndarray] = None,
    timing_mask: Optional[np.ndarray] = None,
    region_prediction: Optional[np.ndarray] = None,
    region_target: Optional[np.ndarray] = None,
    region_mask: Optional[np.ndarray] = None,
) -> List[Dict[str, Any]]:
    """Pool TP/FP/FN before deriving F1; never average transition F1s."""

    probability = np.asarray(probabilities, dtype=float)
    truth = np.asarray(targets, dtype=bool)
    if probability.shape != truth.shape or probability.shape[1] != len(EVENT_KINDS):
        raise ValueError("event probability and target shapes are incompatible")
    threshold = np.asarray(thresholds, dtype=float)
    if threshold.ndim == 0:
        threshold = np.repeat(threshold, len(EVENT_KINDS))
    if threshold.shape != (len(EVENT_KINDS),):
        raise ValueError("one threshold is required per event class")
    prediction = probability >= threshold[None, :]
    episode_ids = np.asarray(episode_ids, dtype=object)
    rows: List[Dict[str, Any]] = []
    for column, kind in enumerate(EVENT_KINDS):
        positive = truth[:, column]
        detected = prediction[:, column]
        tp = int(np.sum(positive & detected))
        fp = int(np.sum(~positive & detected))
        fn = int(np.sum(positive & ~detected))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        row: Dict[str, Any] = {
            "model": model,
            "split": split,
            "event_kind": kind,
            "threshold": float(threshold[column]),
            "support": int(positive.sum()),
            "episode_support": int(len(set(episode_ids[positive].tolist()))),
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "pr_auc": _average_precision(positive, probability[:, column]),
            "positive_prevalence": float(np.mean(positive)),
        }
        if timing_prediction is not None and timing_target is not None and timing_mask is not None:
            absolute = np.abs(np.asarray(timing_prediction)[:, column] - np.asarray(timing_target)[:, column])
            mask = np.asarray(timing_mask)[:, column].astype(bool) & positive[:, None] & detected[:, None]
            for subcolumn, name in enumerate(("onset", "peak", "offset", "duration")):
                selected = mask[:, subcolumn]
                row[f"{name}_mae_ms"] = float(np.mean(absolute[selected, subcolumn])) if selected.any() else math.nan
        if region_prediction is not None and region_target is not None and region_mask is not None:
            selected = np.asarray(region_mask)[:, column].astype(bool) & positive & detected
            row["localization_support"] = int(selected.sum())
            row["localization_error_rate"] = (
                float(np.mean(np.asarray(region_prediction)[selected, column] != np.asarray(region_target)[selected, column]))
                if selected.any() else math.nan
            )
        rows.append(row)
    return rows


def macro_event_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    values = {str(row["event_kind"]): float(row["f1"]) for row in rows}
    def mean(names: Sequence[str]) -> float:
        return float(np.mean([values.get(name, 0.0) for name in names]))
    return {
        "macro_f1_somatic_axonal": mean(SOMATIC_AXONAL_EVENTS),
        "macro_f1_dendritic": mean(DENDRITIC_EVENTS),
        "macro_f1_overall": mean(EVENT_KINDS),
    }


def episode_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    *,
    value_key: str,
    episode_key: str = "episode_id",
    replicates: int = 2000,
    seed: int = 2027,
) -> Dict[str, float]:
    """Bootstrap independent episodes, retaining all rows of sampled episodes."""

    grouped: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        value = float(row[value_key])
        if math.isfinite(value):
            grouped[str(row[episode_key])].append(value)
    episodes = sorted(grouped)
    if not episodes:
        return {"estimate": math.nan, "ci_low": math.nan, "ci_high": math.nan, "episode_count": 0}
    episode_values = np.asarray([np.mean(grouped[key]) for key in episodes])
    rng = np.random.default_rng(int(seed))
    estimates = np.empty(int(replicates), dtype=float)
    for index in range(int(replicates)):
        estimates[index] = float(np.mean(rng.choice(episode_values, size=len(episode_values), replace=True)))
    return {
        "estimate": float(np.mean(episode_values)),
        "ci_low": float(np.percentile(estimates, 2.5)),
        "ci_high": float(np.percentile(estimates, 97.5)),
        "episode_count": len(episodes),
        "bootstrap_replicates": int(replicates),
    }


def episode_bootstrap_event_f1(
    probabilities: np.ndarray,
    targets: np.ndarray,
    episode_ids: Sequence[str],
    *,
    thresholds: Any = 0.5,
    replicates: int = 2000,
    seed: int = 2027,
) -> List[Dict[str, Any]]:
    """Bootstrap pooled event F1 by resampling complete episodes."""

    probability = np.asarray(probabilities, dtype=float)
    truth = np.asarray(targets, dtype=bool)
    episode = np.asarray(episode_ids, dtype=object)
    threshold = np.asarray(thresholds, dtype=float)
    if threshold.ndim == 0:
        threshold = np.repeat(threshold, len(EVENT_KINDS))
    unique = np.asarray(sorted(set(episode.tolist())), dtype=object)
    if not len(unique):
        return []
    indices = {name: np.flatnonzero(episode == name) for name in unique}
    rng = np.random.default_rng(int(seed))
    samples = np.empty((int(replicates), len(EVENT_KINDS)), dtype=float)
    for repeat in range(int(replicates)):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        rows = np.concatenate([indices[name] for name in chosen])
        predicted = probability[rows] >= threshold[None, :]
        selected_truth = truth[rows]
        tp = np.sum(predicted & selected_truth, axis=0)
        fp = np.sum(predicted & ~selected_truth, axis=0)
        fn = np.sum(~predicted & selected_truth, axis=0)
        samples[repeat] = 2.0 * tp / np.maximum(1, 2 * tp + fp + fn)
    point = probability >= threshold[None, :]
    tp = np.sum(point & truth, axis=0)
    fp = np.sum(point & ~truth, axis=0)
    fn = np.sum(~point & truth, axis=0)
    estimate = 2.0 * tp / np.maximum(1, 2 * tp + fp + fn)
    return [
        {
            "event_kind": kind,
            "estimate": float(estimate[column]),
            "ci_low": float(np.percentile(samples[:, column], 2.5)),
            "ci_high": float(np.percentile(samples[:, column], 97.5)),
            "episode_count": len(unique),
            "bootstrap_replicates": int(replicates),
        }
        for column, kind in enumerate(EVENT_KINDS)
    ]


def voltage_fidelity_rows(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    model: str,
    split: str,
    horizon_ms: int,
    segment_regions: Sequence[str],
    regimes: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    prediction = np.asarray(prediction, dtype=float)
    target = np.asarray(target, dtype=float)
    if prediction.shape != target.shape or prediction.shape[-1] != len(segment_regions):
        raise ValueError("voltage arrays must end in the segment dimension")
    error = prediction - target
    regions = np.asarray(segment_regions, dtype=object)
    aliases = {
        "soma": ("soma",), "AIS": ("ais", "axon"), "basal": ("basal",),
        "apical_trunk": ("trunk",), "nexus_hot_zone": ("nexus", "hot"), "tuft": ("tuft",),
    }
    rows: List[Dict[str, Any]] = []
    scopes: List[Tuple[str, np.ndarray]] = [("all", np.ones(len(regions), dtype=bool))]
    lower = np.asarray([str(value).lower() for value in regions])
    for name, tokens in aliases.items():
        mask = np.asarray([any(token in value for token in tokens) for value in lower])
        if mask.any():
            scopes.append((name, mask))
    for region, mask in scopes:
        selected = error[..., mask].reshape(-1)
        teacher = target[..., mask].reshape(-1)
        predicted = prediction[..., mask].reshape(-1)
        absolute = np.abs(selected)
        rows.append({
            "model": model, "split": split, "horizon_ms": int(horizon_ms), "region": region,
            "rmse_mv": float(np.sqrt(np.mean(selected ** 2))), "mae_mv": float(np.mean(absolute)),
            "absolute_error_p50_mv": float(np.percentile(absolute, 50)),
            "absolute_error_p95_mv": float(np.percentile(absolute, 95)),
            "absolute_error_p99_mv": float(np.percentile(absolute, 99)),
            "baseline_drift_mv": float(np.mean(selected)),
            "teacher_peak_mv": float(np.max(teacher)), "predicted_peak_mv": float(np.max(predicted)),
            "peak_attenuation_mv": float(np.max(teacher) - np.max(predicted)),
        })
    if regimes is not None:
        regime_values = np.asarray(regimes, dtype=object)
        if prediction.shape[0] != len(regime_values):
            raise ValueError("one regime is required per leading sample")
        for regime in sorted(set(regime_values.tolist())):
            selected_rows = regime_values == regime
            selected = error[selected_rows].reshape(-1)
            absolute = np.abs(selected)
            rows.append({
                "model": model, "split": split, "horizon_ms": int(horizon_ms), "region": "all",
                "regime": str(regime), "rmse_mv": float(np.sqrt(np.mean(selected ** 2))),
                "mae_mv": float(np.mean(absolute)), "absolute_error_p50_mv": float(np.percentile(absolute, 50)),
                "absolute_error_p95_mv": float(np.percentile(absolute, 95)), "absolute_error_p99_mv": float(np.percentile(absolute, 99)),
                "baseline_drift_mv": float(np.mean(selected)),
            })
    return rows


def branching_metrics(
    pairs: Sequence[Mapping[str, Any]],
    *,
    teacher_distance_floor: float,
    collapse_retention: float = 0.1,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for pair in pairs:
        teacher = np.asarray(pair["teacher_a"], dtype=float) - np.asarray(pair["teacher_b"], dtype=float)
        predicted = np.asarray(pair["prediction_a"], dtype=float) - np.asarray(pair["prediction_b"], dtype=float)
        teacher_distance = float(np.linalg.norm(teacher) / np.sqrt(max(1, teacher.size)))
        predicted_distance = float(np.linalg.norm(predicted) / np.sqrt(max(1, predicted.size)))
        eligible = teacher_distance >= float(teacher_distance_floor)
        retention = predicted_distance / teacher_distance if teacher_distance > 0 else math.nan
        teacher_curve = np.linalg.norm(teacher.reshape(teacher.shape[0], -1), axis=1) if teacher.ndim > 1 else np.abs(teacher)
        predicted_curve = np.linalg.norm(predicted.reshape(predicted.shape[0], -1), axis=1) if predicted.ndim > 1 else np.abs(predicted)
        first_teacher = int(np.argmax(teacher_curve >= float(teacher_distance_floor))) if np.any(teacher_curve >= float(teacher_distance_floor)) else -1
        first_predicted = int(np.argmax(predicted_curve >= float(teacher_distance_floor))) if np.any(predicted_curve >= float(teacher_distance_floor)) else -1
        rows.append({
            "pair_id": str(pair["pair_id"]), "branching_kind": str(pair["branching_kind"]),
            "horizon_ms": int(pair["horizon_ms"]), "teacher_distance": teacher_distance,
            "predicted_distance": predicted_distance, "eligible": bool(eligible),
            "divergence_retention": float(retention), "over_divergence": bool(eligible and retention > 1.0),
            "collapsed": bool(eligible and retention < collapse_retention),
            "teacher_first_divergence_ms": first_teacher, "predicted_first_divergence_ms": first_predicted,
            "first_divergence_error_ms": abs(first_predicted - first_teacher) if min(first_teacher, first_predicted) >= 0 else math.nan,
            "divergent_event_correct": bool(pair.get("divergent_event_correct", False)),
        })
    eligible_values = np.asarray([row["divergence_retention"] for row in rows if row["eligible"]], dtype=float)
    summary = {
        "teacher_distance_floor": float(teacher_distance_floor),
        "eligible_pair_count": int(len(eligible_values)),
        "excluded_low_teacher_distance_count": int(sum(not row["eligible"] for row in rows)),
        "median": float(np.median(eligible_values)) if len(eligible_values) else math.nan,
        "iqr_low": float(np.percentile(eligible_values, 25)) if len(eligible_values) else math.nan,
        "iqr_high": float(np.percentile(eligible_values, 75)) if len(eligible_values) else math.nan,
        "p05": float(np.percentile(eligible_values, 5)) if len(eligible_values) else math.nan,
        "p95": float(np.percentile(eligible_values, 95)) if len(eligible_values) else math.nan,
        "collapse_rate": float(np.mean(eligible_values < collapse_retention)) if len(eligible_values) else math.nan,
        "over_divergence_rate": float(np.mean(eligible_values > 1.0)) if len(eligible_values) else math.nan,
    }
    return rows, summary


def identifiability_summary(pair_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Summarize paired conditional target ambiguity under the three U views."""

    by_regime: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in pair_rows:
        regimes = row.get("release_regimes")
        if regimes is None:
            regimes = [row.get("release_regime", "all")]
        for regime in regimes:
            by_regime[str(regime)].append(row)
        if bool(row.get("near_dendritic_event", False)):
            by_regime["near_dendritic_event"].append(row)
    def summarize(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        scheduled = np.asarray([float(row["scheduled_residual_variance"]) for row in rows])
        rng = np.asarray([float(row["rng_residual_variance"]) for row in rows])
        realized = np.asarray([float(row["realized_residual_variance"]) for row in rows])
        denominator = float(np.mean(scheduled)) if len(scheduled) else math.nan
        return {
            "pair_count": len(rows),
            "scheduled_residual_variance": denominator,
            "rng_residual_variance": float(np.mean(rng)) if len(rng) else math.nan,
            "realized_residual_variance": float(np.mean(realized)) if len(realized) else math.nan,
            "rng_variance_reduction_fraction": 1.0 - float(np.mean(rng)) / denominator if denominator > 0 else math.nan,
            "realized_variance_reduction_fraction": 1.0 - float(np.mean(realized)) / denominator if denominator > 0 else math.nan,
            "mean_target_distance": float(np.mean([float(row["target_distance"]) for row in rows])) if rows else math.nan,
        }
    return {"overall": summarize(pair_rows), "by_release_regime": {key: summarize(value) for key, value in sorted(by_regime.items())}}


def release_flowmap_decision(criteria: Mapping[str, Any]) -> Dict[str, Any]:
    """Conservative preregistered GO/CONDITIONAL_GO/NO_GO decision."""

    b3_better = bool(criteria.get("b3_realized_beats_b1_consistently", False))
    identifiable = bool(criteria.get("realized_reduces_scheduled_ambiguity", False))
    events = bool(criteria.get("dendritic_events_useful", False))
    drift = bool(criteria.get("no_catastrophic_regional_drift", False))
    peaks = bool(criteria.get("peak_attenuation_reduced", False))
    branching = bool(criteria.get("divergence_retention_above_0_30", False))
    recovery = bool(criteria.get("recovery_distinguishable", False))
    stable = bool(criteria.get("stable_across_seeds", False))
    if all((b3_better, identifiable, events, drift, peaks, branching, recovery, stable)):
        decision = "GO"
    elif b3_better and identifiable and stable:
        decision = "CONDITIONAL_GO"
    else:
        decision = "NO_GO"
    blockers = [
        name for name, valid in {
            "B3-U_realized does not consistently beat B1": b3_better,
            "U_realized does not clearly reduce scheduled ambiguity": identifiable,
            "dendritic event fidelity is insufficient": events,
            "regional drift is catastrophic": drift,
            "peak attenuation is not reduced": peaks,
            "branching retention is not above 0.30": branching,
            "recovery is not distinguishable": recovery,
            "results are unstable across seeds": stable,
        }.items() if not valid
    ]
    return {"decision": decision, "criteria": dict(criteria), "blockers": blockers}
