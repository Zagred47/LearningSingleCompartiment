"""Diagnostic atlas for input-only micro-Hay surrogate rollouts.

The atlas is deliberately observational: it never trains or fine-tunes a
model.  It turns complete-state teacher/prediction trajectories into several
orthogonal views (state, compartment, response regime, time horizon, spike
waveform, phase space, recurrence and residual spectrum).  Optional GRU
inspection exposes internal gates without changing the checkpoint.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
import csv
import hashlib
import json
import math

import numpy as np

from .event_aware import MICRO_EVENT_NAMES, classify_micro_events, replay_gru_gates
from .input_only import InputOnlyGRU


@dataclass(frozen=True)
class FailureAtlasConfig:
    spike_threshold_mV: float = -20.0
    spike_tolerance_ms: float = 2.0
    waveform_before_ms: float = 5.0
    waveform_after_ms: float = 12.0
    horizons_ms: tuple[float, ...] = (
        1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 200.0, 500.0, 1000.0,
    )
    normalized_drift_thresholds: tuple[float, ...] = (0.25, 0.5, 1.0)
    recurrence_points: int = 1200
    recurrence_rate: float = 0.05
    recurrence_min_line: int = 2
    recurrence_theiler_steps: int = 2
    takens_dimension: int = 3
    takens_max_delay_ms: float = 25.0
    spectrum_bands_hz: tuple[tuple[float, float], ...] = (
        (0.0, 10.0), (10.0, 50.0), (50.0, 200.0), (200.0, 1000.0),
    )


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def state_family(name: str) -> str:
    if name.endswith(".v_mV"):
        return "voltage"
    if name.endswith(".ca_i_mM"):
        return "calcium"
    if any(token in name for token in (".ampa_", ".nmda_", ".gabaa_")):
        return "synaptic_state"
    if any(token in name for token in (".m_", ".h_", ".z_")):
        return "intrinsic_gate"
    return "other"


def state_compartment(name: str) -> str:
    return name.split(".", 1)[0] if "." in name else "global"


def is_slow_state(name: str) -> bool:
    """Declared slow candidates from the retained Hay mechanisms.

    This is a diagnostic grouping, not an estimated time constant.  Empirical
    timescale estimation remains a separate analysis.
    """

    return (
        name.endswith(".ca_i_mM")
        or ".h_Nap_Et2" in name
        or ".h_K_Pst" in name
        or ".m_Ih" in name
        or ".m_Im" in name
        or ".nmda_decay" in name
    )


def _validate_trajectories(
    truth: np.ndarray,
    prediction: np.ndarray,
    state_names: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if truth.ndim != 3 or prediction.ndim != 3:
        raise ValueError("truth and prediction must have shape [trajectory,time,state]")
    if truth.shape != prediction.shape:
        raise ValueError(f"truth/prediction shape mismatch: {truth.shape} vs {prediction.shape}")
    if truth.shape[-1] != len(state_names):
        raise ValueError("state_names length does not match the state dimension")
    if not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise ValueError("truth and prediction must contain only finite values")
    return truth, prediction


def _safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    if left.size < 2 or left.std() < 1e-12 or right.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def statewise_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    state_names: Sequence[str],
    scale: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    truth, prediction = _validate_trajectories(truth, prediction, state_names)
    error = prediction - truth
    teacher_scale = np.asarray(scale, dtype=np.float64) if scale is not None else truth.std(axis=(0, 1))
    teacher_scale = np.maximum(teacher_scale, 1e-12)
    rows: list[dict[str, Any]] = []
    for index, name in enumerate(state_names):
        residual = error[..., index]
        rmse = float(np.sqrt(np.mean(np.square(residual))))
        rows.append({
            "state": name,
            "index": index,
            "compartment": state_compartment(name),
            "family": state_family(name),
            "slow_candidate": is_slow_state(name),
            "rmse": rmse,
            "mae": float(np.mean(np.abs(residual))),
            "bias": float(np.mean(residual)),
            "normalized_rmse": rmse / float(teacher_scale[index]),
            "teacher_std": float(teacher_scale[index]),
            "correlation": _safe_correlation(truth[..., index], prediction[..., index]),
            "derivative_rmse": float(np.sqrt(np.mean(np.square(np.diff(residual, axis=1)))))
            if truth.shape[1] > 1 else float("nan"),
        })
    return rows


def aggregate_state_metrics(rows: Sequence[Mapping[str, Any]], key: str) -> list[dict[str, Any]]:
    names = sorted({str(row[key]) for row in rows})
    result = []
    for name in names:
        subset = [row for row in rows if str(row[key]) == name]
        result.append({
            key: name,
            "states": len(subset),
            "mean_normalized_rmse": float(np.mean([float(row["normalized_rmse"]) for row in subset])),
            "median_normalized_rmse": float(np.median([float(row["normalized_rmse"]) for row in subset])),
            "mean_correlation": float(np.nanmean([float(row["correlation"]) for row in subset])),
        })
    return result


def _crossings(voltage: np.ndarray, threshold: float) -> np.ndarray:
    voltage = np.asarray(voltage)
    return (voltage[:, :-1] < threshold) & (voltage[:, 1:] >= threshold)


def match_spikes(
    truth_voltage: np.ndarray,
    prediction_voltage: np.ndarray,
    dt_ms: float,
    threshold_mV: float = -20.0,
    tolerance_ms: float = 2.0,
) -> tuple[dict[str, Any], list[tuple[int, int, int]]]:
    truth_crossings = _crossings(truth_voltage, threshold_mV)
    prediction_crossings = _crossings(prediction_voltage, threshold_mV)
    tolerance = max(0, int(round(tolerance_ms / dt_ms)))
    pairs: list[tuple[int, int, int]] = []
    timing_errors: list[float] = []
    for trajectory in range(truth_crossings.shape[0]):
        truth_indices = list(np.flatnonzero(truth_crossings[trajectory]))
        predicted_indices = list(np.flatnonzero(prediction_crossings[trajectory]))
        available = set(range(len(truth_indices)))
        for predicted in predicted_indices:
            candidates = [
                (abs(predicted - truth_indices[index]), index)
                for index in available
                if abs(predicted - truth_indices[index]) <= tolerance
            ]
            if not candidates:
                continue
            _, index = min(candidates)
            available.remove(index)
            actual = truth_indices[index]
            pairs.append((trajectory, actual, predicted))
            timing_errors.append((predicted - actual) * dt_ms)
    truth_count = int(truth_crossings.sum())
    prediction_count = int(prediction_crossings.sum())
    matched = len(pairs)
    precision = matched / max(1, prediction_count)
    recall = matched / max(1, truth_count)
    report = {
        "threshold_mV": threshold_mV,
        "tolerance_ms": tolerance_ms,
        "truth_spikes": truth_count,
        "predicted_spikes": prediction_count,
        "matched_spikes": matched,
        "precision": precision,
        "recall": recall,
        "f1": 2.0 * precision * recall / max(1e-12, precision + recall),
        "timing_bias_ms": float(np.mean(timing_errors)) if timing_errors else float("nan"),
        "timing_mae_ms": float(np.mean(np.abs(timing_errors))) if timing_errors else float("nan"),
        "timing_std_ms": float(np.std(timing_errors)) if timing_errors else float("nan"),
    }
    return report, pairs


def spike_waveform_metrics(
    truth_voltage: np.ndarray,
    prediction_voltage: np.ndarray,
    pairs: Sequence[tuple[int, int, int]],
    dt_ms: float,
    before_ms: float = 5.0,
    after_ms: float = 12.0,
) -> dict[str, Any]:
    before = max(1, int(round(before_ms / dt_ms)))
    after = max(1, int(round(after_ms / dt_ms)))
    truth_windows, prediction_windows = [], []
    peak_errors, peak_time_errors = [], []
    for trajectory, truth_index, predicted_index in pairs:
        if truth_index < before or truth_index + after >= truth_voltage.shape[1]:
            continue
        if predicted_index < before or predicted_index + after >= prediction_voltage.shape[1]:
            continue
        truth_window = truth_voltage[trajectory, truth_index - before : truth_index + after + 1]
        predicted_window = prediction_voltage[trajectory, truth_index - before : truth_index + after + 1]
        truth_windows.append(truth_window)
        prediction_windows.append(predicted_window)
        peak_errors.append(float(predicted_window.max() - truth_window.max()))
        peak_time_errors.append(float((np.argmax(predicted_window) - np.argmax(truth_window)) * dt_ms))
    if not truth_windows:
        return {"waveforms": 0}
    truth_array = np.asarray(truth_windows)
    prediction_array = np.asarray(prediction_windows)
    return {
        "waveforms": len(truth_windows),
        "waveform_rmse_mV": float(np.sqrt(np.mean(np.square(prediction_array - truth_array)))),
        "peak_amplitude_bias_mV": float(np.mean(peak_errors)),
        "peak_amplitude_mae_mV": float(np.mean(np.abs(peak_errors))),
        "peak_timing_bias_ms": float(np.mean(peak_time_errors)),
        "peak_timing_mae_ms": float(np.mean(np.abs(peak_time_errors))),
        "mean_truth_waveform_mV": truth_array.mean(0).tolist(),
        "mean_prediction_waveform_mV": prediction_array.mean(0).tolist(),
        "waveform_time_ms": (np.arange(-before, after + 1) * dt_ms).tolist(),
    }


def teacher_centered_waveform_metrics(
    truth_voltage: np.ndarray,
    prediction_voltage: np.ndarray,
    dt_ms: float,
    threshold_mV: float = -20.0,
    before_ms: float = 5.0,
    after_ms: float = 12.0,
) -> dict[str, Any]:
    """Compare both traces around every teacher spike, including misses.

    Matched-spike metrics become empty when a model predicts no threshold
    crossings.  Teacher-centred windows keep the amplitude and waveform
    failure measurable in exactly that case.
    """

    truth_voltage = np.asarray(truth_voltage, dtype=np.float64)
    prediction_voltage = np.asarray(prediction_voltage, dtype=np.float64)
    if truth_voltage.shape != prediction_voltage.shape or truth_voltage.ndim != 2:
        raise ValueError("voltage traces must have matching [trajectory,time] shapes")
    before = max(1, int(round(before_ms / dt_ms)))
    after = max(1, int(round(after_ms / dt_ms)))
    crossings = _crossings(truth_voltage, threshold_mV)
    truth_windows, prediction_windows = [], []
    for trajectory in range(truth_voltage.shape[0]):
        for crossing in np.flatnonzero(crossings[trajectory]):
            center = int(crossing + 1)
            if center < before or center + after >= truth_voltage.shape[1]:
                continue
            truth_windows.append(truth_voltage[trajectory, center - before : center + after + 1])
            prediction_windows.append(prediction_voltage[trajectory, center - before : center + after + 1])
    if not truth_windows:
        return {"teacher_spike_windows": 0}
    truth_array = np.asarray(truth_windows)
    prediction_array = np.asarray(prediction_windows)
    truth_peaks = truth_array.max(axis=1)
    predicted_peaks = prediction_array.max(axis=1)
    return {
        "teacher_spike_windows": int(len(truth_array)),
        "waveform_rmse_mV": float(np.sqrt(np.mean(np.square(prediction_array - truth_array)))),
        "peak_amplitude_bias_mV": float(np.mean(predicted_peaks - truth_peaks)),
        "peak_amplitude_mae_mV": float(np.mean(np.abs(predicted_peaks - truth_peaks))),
        "mean_truth_peak_mV": float(np.mean(truth_peaks)),
        "mean_prediction_peak_mV": float(np.mean(predicted_peaks)),
        "predicted_peak_above_threshold_fraction": float(np.mean(predicted_peaks >= threshold_mV)),
        "mean_truth_waveform_mV": truth_array.mean(0).tolist(),
        "mean_prediction_waveform_mV": prediction_array.mean(0).tolist(),
        "waveform_time_ms": (np.arange(-before, after + 1) * dt_ms).tolist(),
    }


def masked_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    masks: np.ndarray,
    mask_names: Sequence[str],
    state_names: Sequence[str],
) -> list[dict[str, Any]]:
    truth, prediction = _validate_trajectories(truth, prediction, state_names)
    masks = np.asarray(masks, dtype=bool)
    if masks.shape[:2] != truth.shape[:2] or masks.shape[-1] != len(mask_names):
        raise ValueError("mask shape is incompatible with trajectories")
    error = prediction - truth
    voltage_indices = [i for i, name in enumerate(state_names) if name.endswith(".v_mV")]
    soma_index = list(state_names).index("soma.v_mV")
    scale = np.maximum(truth.std(axis=(0, 1)), 1e-12)
    rows = []
    for index, name in enumerate(mask_names):
        mask = masks[..., index]
        if not mask.any():
            rows.append({"view": name, "samples": 0})
            continue
        residual = error[mask]
        rows.append({
            "view": name,
            "samples": int(mask.sum()),
            "fraction": float(mask.mean()),
            "soma_rmse_mV": float(np.sqrt(np.mean(np.square(error[..., soma_index][mask])))),
            "all_voltage_rmse_mV": float(np.sqrt(np.mean(np.square(residual[:, voltage_indices])))),
            "mean_normalized_rmse": float(np.mean(np.sqrt(np.mean(np.square(residual), axis=0)) / scale)),
        })
    return rows


def horizon_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    state_names: Sequence[str],
    dt_ms: float,
    horizons_ms: Sequence[float],
) -> list[dict[str, Any]]:
    truth, prediction = _validate_trajectories(truth, prediction, state_names)
    error = prediction - truth
    scale = np.maximum(truth.std(axis=(0, 1)), 1e-12)
    soma_index = list(state_names).index("soma.v_mV")
    voltage_indices = [i for i, name in enumerate(state_names) if name.endswith(".v_mV")]
    rows = []
    for horizon in horizons_ms:
        steps = int(round(horizon / dt_ms))
        if steps < 1 or steps > truth.shape[1]:
            continue
        residual = error[:, :steps]
        rows.append({
            "horizon_ms": float(horizon),
            "steps": steps,
            "soma_rmse_mV": float(np.sqrt(np.mean(np.square(residual[..., soma_index])))),
            "all_voltage_rmse_mV": float(np.sqrt(np.mean(np.square(residual[..., voltage_indices])))),
            "mean_normalized_rmse": float(np.mean(np.sqrt(np.mean(np.square(residual), axis=(0, 1))) / scale)),
        })
    return rows


def drift_thresholds(
    truth: np.ndarray,
    prediction: np.ndarray,
    state_names: Sequence[str],
    dt_ms: float,
    thresholds: Sequence[float],
) -> list[dict[str, Any]]:
    truth, prediction = _validate_trajectories(truth, prediction, state_names)
    scale = np.maximum(truth.std(axis=(0, 1)), 1e-12)
    normalized_sq = np.square((prediction - truth) / scale)
    cumulative = np.cumsum(normalized_sq, axis=1)
    denominator = np.arange(1, truth.shape[1] + 1, dtype=np.float64)[None, :, None]
    cumulative_nrmse = np.sqrt(cumulative / denominator).mean(axis=(0, 2))
    rows = []
    for threshold in thresholds:
        crossing = np.flatnonzero(cumulative_nrmse >= threshold)
        rows.append({
            "normalized_rmse_threshold": float(threshold),
            "first_crossing_ms": float((crossing[0] + 1) * dt_ms) if crossing.size else float("nan"),
            "crossed": bool(crossing.size),
        })
    return rows


def _run_lengths(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=bool)
    if not values.size:
        return np.empty(0, dtype=np.int64)
    padded = np.pad(values.astype(np.int8), (1, 1))
    changes = np.flatnonzero(np.diff(padded))
    return (changes[1::2] - changes[::2]).astype(np.int64)


def recurrence_quantification(matrix: np.ndarray, valid: np.ndarray, minimum_line: int = 2) -> dict[str, float]:
    recurrence_points = int((matrix & valid).sum())
    possible = int(valid.sum())
    diagonal_lengths = []
    for offset in range(-matrix.shape[0] + 1, matrix.shape[1]):
        diagonal_lengths.extend(_run_lengths(np.diagonal(matrix & valid, offset=offset)).tolist())
    vertical_lengths = []
    for column in range(matrix.shape[1]):
        vertical_lengths.extend(_run_lengths((matrix & valid)[:, column]).tolist())
    diagonal = np.asarray([value for value in diagonal_lengths if value >= minimum_line], dtype=np.float64)
    vertical = np.asarray([value for value in vertical_lengths if value >= minimum_line], dtype=np.float64)
    return {
        "recurrence_rate": recurrence_points / max(1, possible),
        "determinism": float(diagonal.sum() / max(1, recurrence_points)),
        "mean_diagonal_length": float(diagonal.mean()) if diagonal.size else 0.0,
        "max_diagonal_length": float(diagonal.max()) if diagonal.size else 0.0,
        "laminarity": float(vertical.sum() / max(1, recurrence_points)),
        "trapping_time": float(vertical.mean()) if vertical.size else 0.0,
    }


def recurrence_analysis(
    truth_features: np.ndarray,
    prediction_features: np.ndarray,
    max_points: int = 1200,
    target_rate: float = 0.05,
    minimum_line: int = 2,
    theiler_steps: int = 2,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    truth_features = np.asarray(truth_features, dtype=np.float64)
    prediction_features = np.asarray(prediction_features, dtype=np.float64)
    if truth_features.shape != prediction_features.shape or truth_features.ndim != 2:
        raise ValueError("recurrence features must have matching [time,feature] shapes")
    stride = max(1, int(math.ceil(len(truth_features) / max_points)))
    truth_features = truth_features[::stride]
    prediction_features = prediction_features[::stride]
    mean = truth_features.mean(0)
    std = np.maximum(truth_features.std(0), 1e-12)
    truth_z = (truth_features - mean) / std
    prediction_z = (prediction_features - mean) / std

    def distances(values: np.ndarray) -> np.ndarray:
        squared = np.sum(np.square(values), axis=1, keepdims=True)
        return np.sqrt(np.maximum(squared + squared.T - 2.0 * values @ values.T, 0.0))

    truth_distance = distances(truth_z)
    prediction_distance = distances(prediction_z)
    indices = np.arange(len(truth_z))
    valid = np.abs(indices[:, None] - indices[None, :]) > theiler_steps
    epsilon = float(np.quantile(truth_distance[valid], target_rate))
    truth_recurrence = truth_distance <= epsilon
    prediction_recurrence = prediction_distance <= epsilon
    report = {
        "points": len(truth_z),
        "stride": stride,
        "epsilon_teacher_standardized": epsilon,
        "target_teacher_recurrence_rate": target_rate,
        "teacher": recurrence_quantification(truth_recurrence, valid, minimum_line),
        "prediction": recurrence_quantification(prediction_recurrence, valid, minimum_line),
    }
    return report, truth_recurrence & valid, prediction_recurrence & valid


def phase_histogram_jsd(
    truth_xy: np.ndarray,
    prediction_xy: np.ndarray,
    bins: int = 48,
) -> float:
    truth_xy = np.asarray(truth_xy, dtype=np.float64).reshape(-1, 2)
    prediction_xy = np.asarray(prediction_xy, dtype=np.float64).reshape(-1, 2)
    combined = np.concatenate((truth_xy, prediction_xy), axis=0)
    lower = np.quantile(combined, 0.005, axis=0)
    upper = np.quantile(combined, 0.995, axis=0)
    upper = np.maximum(upper, lower + 1e-9)
    teacher, _, _ = np.histogram2d(truth_xy[:, 0], truth_xy[:, 1], bins=bins, range=list(zip(lower, upper)))
    predicted, _, _ = np.histogram2d(prediction_xy[:, 0], prediction_xy[:, 1], bins=bins, range=list(zip(lower, upper)))
    teacher = (teacher.reshape(-1) + 1e-12)
    predicted = (predicted.reshape(-1) + 1e-12)
    teacher /= teacher.sum()
    predicted /= predicted.sum()
    midpoint = 0.5 * (teacher + predicted)
    return float(0.5 * np.sum(teacher * np.log(teacher / midpoint)) + 0.5 * np.sum(predicted * np.log(predicted / midpoint)))


def takens_delay_embedding(values: np.ndarray, delay_steps: int, dimension: int = 3) -> np.ndarray:
    """Construct a standard delay-coordinate embedding of one scalar trace."""

    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if delay_steps < 1 or dimension < 2:
        raise ValueError("delay_steps must be positive and dimension at least two")
    points = values.size - (dimension - 1) * delay_steps
    if points < 2:
        raise ValueError("trace is too short for the requested delay embedding")
    return np.stack(
        [values[offset * delay_steps : offset * delay_steps + points] for offset in range(dimension)],
        axis=-1,
    )


def estimate_takens_delay(values: np.ndarray, dt_ms: float, max_delay_ms: float = 25.0) -> int:
    """Select the first teacher autocorrelation crossing below 1/e.

    This is a declared heuristic for visualization, not a proof that the
    resulting coordinates satisfy all hypotheses of Takens' theorem.
    """

    values = np.asarray(values, dtype=np.float64).reshape(-1)
    centered = values - values.mean()
    variance = float(np.dot(centered, centered))
    maximum = min(values.size - 2, max(1, int(round(max_delay_ms / dt_ms))))
    if variance < 1e-20 or maximum < 1:
        return 1
    correlations = np.asarray([
        np.dot(centered[:-lag], centered[lag:]) / variance
        for lag in range(1, maximum + 1)
    ])
    crossings = np.flatnonzero(correlations <= math.exp(-1.0))
    return int(crossings[0] + 1) if crossings.size else int(np.argmin(np.abs(correlations)) + 1)


def residual_spectrum(
    truth_voltage: np.ndarray,
    prediction_voltage: np.ndarray,
    dt_ms: float,
    bands_hz: Sequence[tuple[float, float]],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    truth_voltage = np.asarray(truth_voltage, dtype=np.float64)
    prediction_voltage = np.asarray(prediction_voltage, dtype=np.float64)
    points = truth_voltage.shape[1]
    window = np.hanning(points)
    scale = max(np.square(window).sum(), 1e-12)

    def power(values: np.ndarray) -> np.ndarray:
        centered = values - values.mean(axis=1, keepdims=True)
        return np.mean(np.square(np.abs(np.fft.rfft(centered * window, axis=1))) / scale, axis=0)

    frequency = np.fft.rfftfreq(points, d=dt_ms / 1000.0)
    teacher_psd = power(truth_voltage)
    prediction_psd = power(prediction_voltage)
    residual_psd = power(prediction_voltage - truth_voltage)
    rows = []
    for low, high in bands_hz:
        mask = (frequency >= low) & (frequency < high)
        if not mask.any():
            continue
        teacher_power = float(teacher_psd[mask].sum())
        predicted_power = float(prediction_psd[mask].sum())
        rows.append({
            "low_hz": low,
            "high_hz": high,
            "teacher_power": teacher_power,
            "prediction_power": predicted_power,
            "prediction_to_teacher_ratio": predicted_power / max(teacher_power, 1e-20),
            "residual_power": float(residual_psd[mask].sum()),
        })
    log_distance = float(np.sqrt(np.mean(np.square(np.log1p(prediction_psd) - np.log1p(teacher_psd)))))
    return {"log_psd_rmse": log_distance, "bands": rows}, {
        "frequency_hz": frequency,
        "teacher_psd": teacher_psd,
        "prediction_psd": prediction_psd,
        "residual_psd": residual_psd,
    }


def residual_autocorrelation(
    residual: np.ndarray,
    dt_ms: float,
    lags_ms: Sequence[float] = (0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0),
) -> list[dict[str, float]]:
    residual = np.asarray(residual, dtype=np.float64)
    rows = []
    for lag_ms in lags_ms:
        lag = max(1, int(round(lag_ms / dt_ms)))
        if lag >= residual.shape[1]:
            continue
        correlations = [
            _safe_correlation(values[:-lag], values[lag:])
            for values in residual
        ]
        finite = [value for value in correlations if np.isfinite(value)]
        rows.append({
            "lag_ms": float(lag_ms),
            "mean_autocorrelation": float(np.mean(finite)) if finite else float("nan"),
        })
    return rows


def load_prediction_archive(
    path: str | Path,
    model_key: str | None = None,
) -> tuple[np.ndarray, np.ndarray, str, np.ndarray | None]:
    with np.load(path, allow_pickle=False) as archive:
        keys = list(archive.files)
        if "truth" in archive and "prediction" in archive:
            truth, prediction, selected = archive["truth"], archive["prediction"], model_key or "prediction"
        elif "teacher" in archive:
            candidates = [key for key in keys if key not in {"teacher", "events"}]
            selected = model_key or (candidates[0] if len(candidates) == 1 else None)
            if selected is None or selected not in archive:
                raise ValueError(f"choose model_key from {candidates}")
            truth, prediction = archive["teacher"], archive[selected]
        else:
            raise ValueError(f"unsupported prediction archive keys: {keys}")
        events = archive["events"] if "events" in archive else None
    return np.asarray(truth), np.asarray(prediction), str(selected), None if events is None else np.asarray(events)


def _resample_binary(values: np.ndarray, target_steps: int) -> np.ndarray:
    if values.shape[1] == target_steps:
        return values
    if values.shape[1] % target_steps:
        raise ValueError("binary time axis is not an integer multiple of target steps")
    factor = values.shape[1] // target_steps
    return values[:, : target_steps * factor].reshape(values.shape[0], target_steps, factor, *values.shape[2:]).max(axis=2)


def _resample_categorical(values: np.ndarray, target_steps: int) -> np.ndarray:
    if values.shape[1] == target_steps:
        return values
    if values.shape[1] % target_steps:
        raise ValueError("categorical time axis is not an integer multiple of target steps")
    factor = values.shape[1] // target_steps
    bins = values[:, : target_steps * factor].reshape(values.shape[0], target_steps, factor)
    result = np.empty(bins.shape[:2], dtype=values.dtype)
    for trajectory in range(bins.shape[0]):
        for step in range(bins.shape[1]):
            labels, counts = np.unique(bins[trajectory, step], return_counts=True)
            result[trajectory, step] = labels[np.argmax(counts)]
    return result


def load_dataset_context(
    path: str | Path,
    target_trajectories: int,
    target_steps: int,
    split: str = "test",
) -> dict[str, Any]:
    import h5py

    with h5py.File(path, "r") as handle:
        schema_version = handle.attrs.get("schema_version")
        model = handle.attrs.get("model")
        state_names = json.loads(handle.attrs["state_names_json"])
        regime_names = json.loads(handle.attrs.get("regime_names_json", "[]"))
        input_names = json.loads(handle.attrs.get("input_names_json", "[]"))
        config = json.loads(handle.attrs["config_json"])
        raw_steps = handle[f"{split}/inputs"].shape[1]
        raw_dt_ms = float(config["dt_ms"])
        count = min(target_trajectories, handle[f"{split}/inputs"].shape[0])
        raw_spikes = handle[f"{split}/spikes"][:count]
        raw_regimes = handle[f"{split}/regimes"][:count] if f"{split}/regimes" in handle else None
    if count != target_trajectories:
        raise ValueError(f"dataset has {count} trajectories but predictions have {target_trajectories}")
    if raw_steps % target_steps:
        raise ValueError(f"dataset has {raw_steps} steps, incompatible with prediction length {target_steps}")
    factor = raw_steps // target_steps
    return {
        "path": str(path),
        "schema_version": None if schema_version is None else str(schema_version),
        "model": None if model is None else str(model),
        "state_names": state_names,
        "input_names": input_names,
        "regime_names": regime_names,
        "raw_dt_ms": raw_dt_ms,
        "model_dt_ms": raw_dt_ms * factor,
        "temporal_bin": factor,
        "spikes": _resample_binary(raw_spikes, target_steps),
        "regimes": None if raw_regimes is None else _resample_categorical(raw_regimes, target_steps),
    }


def build_failure_atlas(
    truth: np.ndarray,
    prediction: np.ndarray,
    state_names: Sequence[str],
    dt_ms: float,
    *,
    event_masks: np.ndarray | None = None,
    event_names: Sequence[str] = MICRO_EVENT_NAMES,
    regimes: np.ndarray | None = None,
    regime_names: Sequence[str] | None = None,
    teacher_spikes: np.ndarray | None = None,
    state_scale: np.ndarray | None = None,
    config: FailureAtlasConfig | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = config or FailureAtlasConfig()
    truth, prediction = _validate_trajectories(truth, prediction, state_names)
    if dt_ms <= 0:
        raise ValueError("dt_ms must be positive")
    state_rows = statewise_metrics(truth, prediction, state_names, state_scale)
    family_rows = aggregate_state_metrics(state_rows, "family")
    compartment_rows = aggregate_state_metrics(state_rows, "compartment")
    soma_index = list(state_names).index("soma.v_mV")
    voltage_indices = [list(state_names).index(f"{region}.v_mV") for region in ("soma", "basal", "trunk", "tuft")]
    if teacher_spikes is None:
        teacher_spikes = _crossings(truth[..., soma_index], config.spike_threshold_mV)
        teacher_spikes = np.pad(teacher_spikes, ((0, 0), (1, 0)))
    if event_masks is None:
        event_masks = classify_micro_events(truth, teacher_spikes, state_names, dt_ms)
    spike, pairs = match_spikes(
        truth[..., soma_index], prediction[..., soma_index], dt_ms,
        config.spike_threshold_mV, config.spike_tolerance_ms,
    )
    waveform = spike_waveform_metrics(
        truth[..., soma_index], prediction[..., soma_index], pairs, dt_ms,
        config.waveform_before_ms, config.waveform_after_ms,
    )
    teacher_centered_waveform = teacher_centered_waveform_metrics(
        truth[..., soma_index], prediction[..., soma_index], dt_ms,
        config.spike_threshold_mV, config.waveform_before_ms, config.waveform_after_ms,
    )
    event_rows = masked_metrics(truth, prediction, event_masks, event_names, state_names)
    regime_rows: list[dict[str, Any]] = []
    if regimes is not None and regime_names:
        masks = np.stack([regimes == index for index in range(len(regime_names))], axis=-1)
        regime_rows = masked_metrics(truth, prediction, masks, regime_names, state_names)
    horizons = horizon_metrics(truth, prediction, state_names, dt_ms, config.horizons_ms)
    drift = drift_thresholds(truth, prediction, state_names, dt_ms, config.normalized_drift_thresholds)

    phase_pairs = []
    for left, right in (
        ("soma.v_mV", "soma.ca_i_mM"),
        ("soma.v_mV", "soma.h_NaTa_t"),
        ("trunk.v_mV", "tuft.v_mV"),
    ):
        if left in state_names and right in state_names:
            li, ri = list(state_names).index(left), list(state_names).index(right)
            phase_pairs.append({
                "left": left,
                "right": right,
                "histogram_js_divergence": phase_histogram_jsd(
                    truth[..., [li, ri]], prediction[..., [li, ri]],
                ),
            })

    delay_steps = estimate_takens_delay(
        truth[0, :, soma_index], dt_ms, config.takens_max_delay_ms
    )
    takens_teacher = takens_delay_embedding(
        truth[0, :, soma_index], delay_steps, config.takens_dimension
    )
    takens_prediction = takens_delay_embedding(
        prediction[0, :, soma_index], delay_steps, config.takens_dimension
    )
    takens = {
        "dimension": config.takens_dimension,
        "delay_steps": delay_steps,
        "delay_ms": delay_steps * dt_ms,
        "selection": "first teacher autocorrelation crossing below 1/e",
        "histogram_js_divergence_first_two_coordinates": phase_histogram_jsd(
            takens_teacher[:, :2], takens_prediction[:, :2]
        ),
        "interpretation_limit": "diagnostic embedding; Takens theorem assumptions are not asserted",
    }

    # Index time first: NumPy advanced indexing would otherwise transpose this
    # into [feature,time] when the index list is used in the same expression.
    recurrence_features = truth[0][:, voltage_indices]
    recurrence_prediction = prediction[0][:, voltage_indices]
    recurrence, teacher_recurrence, prediction_recurrence = recurrence_analysis(
        recurrence_features,
        recurrence_prediction,
        max_points=config.recurrence_points,
        target_rate=config.recurrence_rate,
        minimum_line=config.recurrence_min_line,
        theiler_steps=config.recurrence_theiler_steps,
    )
    spectrum, spectrum_arrays = residual_spectrum(
        truth[..., soma_index], prediction[..., soma_index], dt_ms, config.spectrum_bands_hz,
    )
    autocorrelation = residual_autocorrelation(prediction[..., soma_index] - truth[..., soma_index], dt_ms)
    report = {
        "config": asdict(config),
        "shape": list(truth.shape),
        "dt_ms": dt_ms,
        "summary": {
            "mean_state_normalized_rmse": float(np.mean([row["normalized_rmse"] for row in state_rows])),
            "median_state_normalized_rmse": float(np.median([row["normalized_rmse"] for row in state_rows])),
            "soma_rmse_mV": next(row["rmse"] for row in state_rows if row["state"] == "soma.v_mV"),
            "all_voltage_rmse_mV": float(np.sqrt(np.mean(np.square((prediction - truth)[..., voltage_indices])))),
            "slow_state_mean_normalized_rmse": float(np.mean([
                row["normalized_rmse"] for row in state_rows if row["slow_candidate"]
            ])),
        },
        "spikes": spike,
        "waveform": waveform,
        "teacher_centered_waveform": teacher_centered_waveform,
        "phase_space": phase_pairs,
        "takens": takens,
        "recurrence": recurrence,
        "spectrum": spectrum,
        "residual_autocorrelation": autocorrelation,
        "drift": drift,
    }
    tables = {
        "statewise": state_rows,
        "families": family_rows,
        "compartments": compartment_rows,
        "events": event_rows,
        "regimes": regime_rows,
        "horizons": horizons,
        "recurrence_teacher": teacher_recurrence,
        "recurrence_prediction": prediction_recurrence,
        "spectrum_arrays": spectrum_arrays,
        "takens_teacher": takens_teacher,
        "takens_prediction": takens_prediction,
    }
    return report, tables


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row.keys()))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_failure_atlas(
    output_dir: str | Path,
    report: Mapping[str, Any],
    tables: Mapping[str, Any],
    truth: np.ndarray,
    prediction: np.ndarray,
    state_names: Sequence[str],
    model_name: str,
) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for name in ("statewise", "families", "compartments", "events", "regimes", "horizons"):
        _write_csv(output / f"{name}_metrics.csv", tables[name])
    (output / "atlas_report.json").write_text(
        json.dumps(_json_safe(dict(report)), indent=2), encoding="utf-8",
    )
    np.savez_compressed(
        output / "dynamics_arrays.npz",
        **tables["spectrum_arrays"],
        takens_teacher=tables["takens_teacher"],
        takens_prediction=tables["takens_prediction"],
    )

    state_rows = sorted(tables["statewise"], key=lambda row: row["normalized_rmse"], reverse=True)
    figure, axis = plt.subplots(figsize=(13, 7))
    shown = state_rows[:30]
    axis.barh([row["state"] for row in reversed(shown)], [row["normalized_rmse"] for row in reversed(shown)])
    axis.set_xlabel("normalized RMSE (teacher test standard deviation)")
    axis.set_title(f"{model_name}: 30 stati più difficili")
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output / "statewise_nrmse.png", dpi=170)
    plt.close(figure)

    if tables["horizons"]:
        figure, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        horizon = np.asarray([row["horizon_ms"] for row in tables["horizons"]])
        axes[0].plot(horizon, [row["soma_rmse_mV"] for row in tables["horizons"]], marker="o")
        axes[0].set_ylabel("soma RMSE (mV)")
        axes[1].plot(horizon, [row["mean_normalized_rmse"] for row in tables["horizons"]], marker="o")
        axes[1].set_ylabel("mean normalized RMSE")
        for axis in axes:
            axis.set_xscale("log")
            axis.set_xlabel("cumulative horizon (ms)")
            axis.grid(alpha=0.25)
        figure.tight_layout()
        figure.savefig(output / "horizon_drift.png", dpi=170)
        plt.close(figure)

    event_rows = [row for row in tables["events"] if row.get("samples", 0)]
    if event_rows:
        figure, axis = plt.subplots(figsize=(11, 5))
        axis.bar([row["view"] for row in event_rows], [row["soma_rmse_mV"] for row in event_rows])
        axis.tick_params(axis="x", rotation=35)
        axis.set_ylabel("soma RMSE (mV)")
        axis.set_title(f"{model_name}: errore condizionato sul regime di risposta")
        axis.grid(axis="y", alpha=0.25)
        figure.tight_layout()
        figure.savefig(output / "event_soma_rmse.png", dpi=170)
        plt.close(figure)

    teacher_waveform = report.get("teacher_centered_waveform", {})
    if teacher_waveform.get("teacher_spike_windows", 0):
        figure, axis = plt.subplots(figsize=(10, 5))
        axis.plot(
            teacher_waveform["waveform_time_ms"],
            teacher_waveform["mean_truth_waveform_mV"],
            label="teacher",
        )
        axis.plot(
            teacher_waveform["waveform_time_ms"],
            teacher_waveform["mean_prediction_waveform_mV"],
            label=model_name,
        )
        axis.axhline(report["spikes"]["threshold_mV"], color="black", linestyle="--", alpha=0.5)
        axis.set_xlabel("time from teacher spike (ms)")
        axis.set_ylabel("soma V (mV)")
        axis.set_title("Teacher-centred spike waveform (includes missed spikes)")
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        figure.savefig(output / "teacher_centered_spike_waveform.png", dpi=170)
        plt.close(figure)

    soma = list(state_names).index("soma.v_mV")
    calcium = list(state_names).index("soma.ca_i_mM")
    stride = max(1, truth.shape[1] // 5000)
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(truth[0, ::stride, soma], truth[0, ::stride, calcium], s=3, alpha=0.35, label="teacher")
    axes[0].scatter(prediction[0, ::stride, soma], prediction[0, ::stride, calcium], s=3, alpha=0.35, label=model_name)
    axes[0].set_xlabel("soma V (mV)")
    axes[0].set_ylabel("soma Ca (mM)")
    axes[0].legend()
    axes[1].imshow(
        np.concatenate((tables["recurrence_teacher"], tables["recurrence_prediction"]), axis=1),
        cmap="binary", origin="lower", aspect="auto", interpolation="nearest",
    )
    axes[1].set_title("recurrence: teacher | prediction")
    axes[1].set_xlabel("time index")
    axes[1].set_ylabel("time index")
    figure.tight_layout()
    figure.savefig(output / "phase_and_recurrence.png", dpi=170)
    plt.close(figure)

    takens_teacher = tables["takens_teacher"]
    takens_prediction = tables["takens_prediction"]
    stride = max(1, len(takens_teacher) // 5000)
    figure = plt.figure(figsize=(11, 8))
    axis = figure.add_subplot(111, projection="3d")
    axis.scatter(*takens_teacher[::stride, :3].T, s=3, alpha=0.30, label="teacher")
    axis.scatter(*takens_prediction[::stride, :3].T, s=3, alpha=0.30, label=model_name)
    delay = report["takens"]["delay_ms"]
    axis.set_xlabel("V(t)")
    axis.set_ylabel(f"V(t+{delay:g} ms)")
    axis.set_zlabel(f"V(t+{2 * delay:g} ms)")
    axis.set_title("Takens-style delay embedding (diagnostic)")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "takens_embedding.png", dpi=170)
    plt.close(figure)

    spectrum = tables["spectrum_arrays"]
    positive = spectrum["frequency_hz"] > 0
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.loglog(spectrum["frequency_hz"][positive], spectrum["teacher_psd"][positive] + 1e-20, label="teacher")
    axis.loglog(spectrum["frequency_hz"][positive], spectrum["prediction_psd"][positive] + 1e-20, label=model_name)
    axis.loglog(spectrum["frequency_hz"][positive], spectrum["residual_psd"][positive] + 1e-20, label="residual", alpha=0.75)
    axis.set_xlabel("frequency (Hz)")
    axis.set_ylabel("power")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "soma_spectrum.png", dpi=170)
    plt.close(figure)

    steps = min(truth.shape[1], int(round(500.0 / float(report["dt_ms"]))))
    time = np.arange(steps) * float(report["dt_ms"])
    figure, axes = plt.subplots(4, 1, figsize=(15, 10), sharex=True)
    for axis, region in zip(axes, ("soma", "basal", "trunk", "tuft")):
        index = list(state_names).index(f"{region}.v_mV")
        axis.plot(time, truth[0, :steps, index], label="teacher", linewidth=1.1)
        axis.plot(time, prediction[0, :steps, index], label=model_name, linewidth=1.0)
        axis.set_ylabel(f"{region} V")
        axis.grid(alpha=0.2)
    axes[0].legend()
    axes[-1].set_xlabel("time (ms)")
    figure.tight_layout()
    figure.savefig(output / "rollout_example_500ms.png", dpi=170)
    plt.close(figure)

    summary = report["summary"]
    spikes = report["spikes"]
    markdown = f"""# Failure Atlas — {model_name}

- Shape: `{tuple(report['shape'])}` at `{report['dt_ms']} ms`
- Soma RMSE: `{summary['soma_rmse_mV']:.4f} mV`
- Mean state normalized RMSE: `{summary['mean_state_normalized_rmse']:.4f}`
- Slow-state mean normalized RMSE: `{summary['slow_state_mean_normalized_rmse']:.4f}`
- Spike truth/predicted/matched: `{spikes['truth_spikes']}/{spikes['predicted_spikes']}/{spikes['matched_spikes']}`
- Spike precision/recall/F1: `{spikes['precision']:.4f}/{spikes['recall']:.4f}/{spikes['f1']:.4f}`

Questo report è diagnostico. Le associazioni tra metriche e meccanismi devono
essere confermate con ablation controllate prima di diventare affermazioni sul
sistema.
"""
    (output / "README.md").write_text(markdown, encoding="utf-8")
    return output


def checkpoint_gru_spec(checkpoint: Mapping[str, Any]) -> tuple[dict[str, Any], Mapping[str, Any]]:
    weights = checkpoint.get("model_state_dict", checkpoint.get("model"))
    if not isinstance(weights, Mapping):
        raise ValueError("checkpoint does not contain model weights")
    if any(key.startswith("baseline.") for key in weights):
        weights = {key.removeprefix("baseline."): value for key, value in weights.items() if key.startswith("baseline.")}
    if "recurrent.weight_hh_l0" not in weights or "recurrent.weight_hh_l1" in weights:
        raise ValueError("checkpoint is not a supported one-layer InputOnlyGRU")
    hidden_dim = int(weights["recurrent.weight_hh_l0"].shape[1])
    input_dim = int(weights["input_encoder.0.weight"].shape[1])
    state_dim = int(weights["decoder.network.3.weight"].shape[0])
    decoder_dim = int(weights["decoder.network.1.weight"].shape[0])
    return {
        "input_dim": input_dim,
        "state_dim": state_dim,
        "hidden_dim": hidden_dim,
        "decoder_dim": decoder_dim,
    }, weights


def inspect_gru_checkpoint(
    checkpoint_path: str | Path,
    dataset_path: str | Path,
    *,
    split: str = "test",
    max_trajectories: int | None = None,
    chunk_steps: int = 512,
    activation_timepoints: int = 4096,
    device: str | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str], float, dict[str, Any], list[dict[str, Any]]]:
    """Run a checkpoint read-only and expose event-conditioned GRU statistics."""

    import h5py
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    spec, weights = checkpoint_gru_spec(checkpoint)
    if activation_timepoints < 1:
        raise ValueError("activation_timepoints must be positive")
    state_mean = np.asarray(checkpoint["state_mean"], dtype=np.float32)
    state_std = np.asarray(checkpoint["state_std"], dtype=np.float32)
    with h5py.File(dataset_path, "r") as handle:
        state_names = list(checkpoint.get("state_names", json.loads(handle.attrs["state_names_json"])))
        raw_input_dim = int(handle[f"{split}/inputs"].shape[-1])
        if spec["input_dim"] % raw_input_dim:
            raise ValueError("checkpoint input width is incompatible with dataset spike width")
        temporal_bin = spec["input_dim"] // raw_input_dim
        raw_dt = float(json.loads(handle.attrs["config_json"])["dt_ms"])
        count = handle[f"{split}/inputs"].shape[0]
        if max_trajectories is not None:
            count = min(count, max_trajectories)
        if f"{split}/burnin_inputs" not in handle:
            raise ValueError("dataset schema with spike-only burn-in is required for causal inspection")
        raw_burnin = handle[f"{split}/burnin_inputs"][:count]
        raw_inputs = handle[f"{split}/inputs"][:count]
        raw_states = handle[f"{split}/states"][:count]
        raw_spikes = handle[f"{split}/spikes"][:count]

    def pack(values: np.ndarray) -> np.ndarray:
        usable = values.shape[1] // temporal_bin * temporal_bin
        return values[:, :usable].reshape(values.shape[0], usable // temporal_bin, temporal_bin * values.shape[2]).astype(np.float32)

    burnin = pack(raw_burnin)
    inputs = pack(raw_inputs)
    truth = raw_states[:, ::temporal_bin].astype(np.float32)[:, 1 : inputs.shape[1] + 1]
    spikes = _resample_binary(raw_spikes, inputs.shape[1])
    event_masks = classify_micro_events(truth, spikes, state_names, raw_dt * temporal_bin)
    torch_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = InputOnlyGRU(**spec).to(torch_device)
    model.load_state_dict(weights)
    model.eval()

    activation_names = ("encoded", "reset", "update", "candidate", "hidden", "decoder")
    predictions = []
    activation_records = {name: [] for name in activation_names}
    sampled_event_masks = []
    sample_stride = max(1, int(math.ceil(count * inputs.shape[1] / activation_timepoints)))
    global_step = 0
    max_replay_error = 0.0
    with torch.no_grad():
        for trajectory in range(count):
            hidden = None
            burn = torch.as_tensor(burnin[trajectory : trajectory + 1], device=torch_device)
            for start in range(0, burn.shape[1], chunk_steps):
                _, hidden = model(burn[:, start : start + chunk_steps], hidden)
            sequence = torch.as_tensor(inputs[trajectory : trajectory + 1], device=torch_device)
            pieces = []
            for start in range(0, sequence.shape[1], chunk_steps):
                chunk = sequence[:, start : start + chunk_steps]
                encoded = model.input_encoder(chunk)
                output, next_hidden = model(chunk, hidden)
                replay = replay_gru_gates(model, chunk, hidden)
                recurrent, _ = model.recurrent(encoded, hidden)
                max_replay_error = max(max_replay_error, float((recurrent - replay["hidden"]).abs().max()))
                decoder = model.decoder.network[2](model.decoder.network[1](model.decoder.network[0](recurrent)))
                pieces.append(output)
                for name, value in (("encoded", encoded), ("decoder", decoder), *replay.items()):
                    positions = np.arange(global_step, global_step + chunk.shape[1])
                    selected = np.flatnonzero(positions % sample_stride == 0)
                    if selected.size:
                        activation_records[name].append(value.squeeze(0)[selected].cpu().numpy())
                if selected.size:
                    sampled_event_masks.append(
                        event_masks[trajectory, start : start + chunk.shape[1]][selected]
                    )
                global_step += chunk.shape[1]
                hidden = next_hidden
            normalized = torch.cat(pieces, dim=1).squeeze(0).cpu().numpy()
            predictions.append(normalized * state_std + state_mean)
    prediction = np.stack(predictions)
    records = {name: np.concatenate(values, axis=0) for name, values in activation_records.items()}
    sampled_masks = np.concatenate(sampled_event_masks, axis=0).astype(bool)
    rows = []
    for event_index, event_name in enumerate(MICRO_EVENT_NAMES):
        mask = sampled_masks[:, event_index]
        inverse = ~mask
        if not mask.any():
            continue
        for name, record in records.items():
            values = record[mask]
            background = record[inverse]
            contrast = np.linalg.norm(values.mean(0) - background.mean(0)) / max(
                1e-12, np.sqrt(np.mean(np.square(background.std(0))))
            )
            rows.append({
                "event": event_name,
                "activation": name,
                "samples": int(values.shape[0]),
                "mean": float(values.mean()),
                "std": float(values.std()),
                "q01": float(np.quantile(values, 0.01)),
                "q50": float(np.quantile(values, 0.50)),
                "q99": float(np.quantile(values, 0.99)),
                "near_zero_fraction": float((np.abs(values) < 0.05).mean()),
                "low_saturation_fraction": float((values < 0.05).mean()) if name in {"reset", "update"} else float("nan"),
                "high_saturation_fraction": float((values > 0.95).mean()) if name in {"reset", "update"} else float("nan"),
                "event_background_centroid_distance": float(contrast),
            })
    hidden_flat = records["hidden"]
    singular = np.linalg.svd(hidden_flat - hidden_flat.mean(0), compute_uv=False)
    explained = np.square(singular)
    cumulative = np.cumsum(explained) / max(explained.sum(), 1e-12)
    diagnostics = {
        "checkpoint": str(checkpoint_path),
        "dataset": str(dataset_path),
        "device": str(torch_device),
        "temporal_bin": temporal_bin,
        "model_dt_ms": raw_dt * temporal_bin,
        "trajectories": count,
        "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "hidden_dim": spec["hidden_dim"],
        "gate_replay_max_abs_error": max_replay_error,
        "activation_timepoints": int(len(sampled_masks)),
        "activation_sampling_stride": sample_stride,
        "hidden_effective_rank_90_percent": int(np.searchsorted(cumulative, 0.90) + 1),
        "hidden_effective_rank_99_percent": int(np.searchsorted(cumulative, 0.99) + 1),
        "singular_values": singular.tolist(),
    }
    return truth, prediction, state_names, raw_dt * temporal_bin, diagnostics, rows


def write_activation_diagnostics(
    output_dir: str | Path,
    diagnostics: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "activation_by_event.csv", rows)
    (output / "activation_report.json").write_text(
        json.dumps(_json_safe(dict(diagnostics)), indent=2), encoding="utf-8",
    )
    gate_rows = [row for row in rows if row["activation"] in {"reset", "update"}]
    if gate_rows:
        events = list(dict.fromkeys(row["event"] for row in gate_rows))
        figure, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
        for axis, gate in zip(axes, ("reset", "update")):
            subset = {row["event"]: row for row in gate_rows if row["activation"] == gate}
            x = np.arange(len(events))
            axis.bar(x - 0.18, [subset[event]["low_saturation_fraction"] for event in events], 0.36, label="<0.05")
            axis.bar(x + 0.18, [subset[event]["high_saturation_fraction"] for event in events], 0.36, label=">0.95")
            axis.set_xticks(x, events, rotation=35, ha="right")
            axis.set_title(f"{gate} gate saturation")
            axis.grid(axis="y", alpha=0.2)
            axis.legend()
        figure.tight_layout()
        figure.savefig(output / "gate_saturation_by_event.png", dpi=170)
        plt.close(figure)
