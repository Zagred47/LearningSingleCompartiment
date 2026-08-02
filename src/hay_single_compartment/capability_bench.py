"""Controlled synthetic capability tasks and established sequence-model adapters.

The task is deliberately smaller than the micro-Hay teacher.  It isolates the
combination that the current surrogate misses: slow causal evidence, a
state-dependent trigger, a narrow fast excursion, and recovery.  Architectures
are standard published families; this module does not introduce a new cell.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class FastSlowDifficulty:
    name: str
    slow_decay: float
    cue_probability: float
    trigger_probability: float
    threshold: float
    refractory_steps: int
    waveform: tuple[float, ...]


FAST_SLOW_DIFFICULTIES = {
    "easy": FastSlowDifficulty("easy", 0.94, 0.08, 0.040, 0.15, 8, (1.0, 0.30, -0.40, -0.20)),
    "medium": FastSlowDifficulty("medium", 0.98, 0.05, 0.020, 0.25, 12, (1.0, 0.20, -0.40)),
    "hard": FastSlowDifficulty("hard", 0.995, 0.03, 0.008, 0.35, 16, (1.0, -0.35)),
}


def generate_fast_slow_sequences(
    count: int,
    steps: int,
    difficulty: str | FastSlowDifficulty,
    seed: int,
) -> dict[str, np.ndarray | dict[str, Any]]:
    """Generate a deterministic causal threshold-and-recovery benchmark.

    Inputs are positive cue, negative cue, and trigger event.  The hidden slow
    evidence is a leaky integral of signed cues.  A fast waveform is emitted
    only when a trigger arrives while the evidence is above threshold and the
    process is outside its refractory interval.
    """

    spec = FAST_SLOW_DIFFICULTIES[difficulty] if isinstance(difficulty, str) else difficulty
    rng = np.random.default_rng(seed)
    inputs = np.zeros((count, steps, 3), dtype=np.float32)
    targets = np.zeros((count, steps, 3), dtype=np.float32)
    triggers = np.zeros((count, steps), dtype=np.uint8)
    spikes = np.zeros((count, steps), dtype=np.uint8)
    eligible = np.zeros((count, steps), dtype=np.uint8)
    slow = np.zeros(count, dtype=np.float64)
    recovery = np.zeros(count, dtype=np.float64)
    refractory = np.zeros(count, dtype=np.int64)
    phase = np.zeros(count, dtype=np.int64)

    waveform = np.asarray(spec.waveform, dtype=np.float64)
    width = len(waveform)
    for step in range(steps):
        positive = rng.random(count) < spec.cue_probability
        negative = rng.random(count) < spec.cue_probability
        trigger = rng.random(count) < spec.trigger_probability
        inputs[:, step, 0] = positive
        inputs[:, step, 1] = negative
        inputs[:, step, 2] = trigger
        triggers[:, step] = trigger

        slow = np.clip(
            spec.slow_decay * slow
            + 0.12 * (positive.astype(np.float64) - negative.astype(np.float64)),
            -1.0,
            1.0,
        )
        can_fire = trigger & (slow > spec.threshold) & (refractory == 0) & (phase == 0)
        eligible[:, step] = trigger & (slow > spec.threshold)
        spikes[:, step] = can_fire
        phase[can_fire] = width
        refractory[can_fire] = spec.refractory_steps
        recovery *= 0.90
        recovery[can_fire] = 1.0

        fast = np.zeros(count, dtype=np.float64)
        active = phase > 0
        fast[active] = waveform[width - phase[active]]
        voltage = 0.20 * slow + fast - 0.12 * recovery
        targets[:, step, 0] = slow
        targets[:, step, 1] = voltage
        targets[:, step, 2] = recovery
        phase = np.maximum(0, phase - 1)
        refractory = np.maximum(0, refractory - 1)

    return {
        "inputs": inputs,
        "targets": targets,
        "triggers": triggers,
        "spikes": spikes,
        "eligible": eligible,
        "difficulty": asdict(spec),
    }


def _event_window(spikes: np.ndarray, radius: int) -> np.ndarray:
    mask = np.zeros_like(spikes, dtype=bool)
    for trajectory, step in zip(*np.where(spikes)):
        mask[trajectory, max(0, step - radius): min(spikes.shape[1], step + radius + 1)] = True
    return mask


def _predicted_spikes(voltage: np.ndarray, threshold: float = 0.50) -> np.ndarray:
    high = voltage >= threshold
    previous = np.pad(high[:, :-1], ((0, 0), (1, 0)), constant_values=False)
    return high & ~previous


def _match_events(truth: np.ndarray, prediction: np.ndarray, tolerance: int = 2) -> tuple[int, int, int, list[int]]:
    truth_count = int(truth.sum())
    prediction_count = int(prediction.sum())
    matched = 0
    offsets: list[int] = []
    for trajectory in range(truth.shape[0]):
        available = list(np.flatnonzero(prediction[trajectory]))
        for step in np.flatnonzero(truth[trajectory]):
            candidates = [value for value in available if abs(value - step) <= tolerance]
            if not candidates:
                continue
            selected = min(candidates, key=lambda value: abs(value - step))
            available.remove(selected)
            matched += 1
            offsets.append(int(selected - step))
    return truth_count, prediction_count, matched, offsets


def fast_slow_metrics(
    prediction: np.ndarray,
    reference: dict[str, np.ndarray | dict[str, Any]],
    target_std: np.ndarray,
) -> dict[str, float | int]:
    target = np.asarray(reference["targets"], dtype=np.float32)
    truth_spikes = np.asarray(reference["spikes"], dtype=bool)
    triggers = np.asarray(reference["triggers"], dtype=bool)
    predicted_spikes = _predicted_spikes(prediction[..., 1])
    truth_count, prediction_count, matched, offsets = _match_events(truth_spikes, predicted_spikes)
    precision = matched / max(1, prediction_count)
    recall = matched / max(1, truth_count)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    error = prediction - target
    normalized = error / np.maximum(np.asarray(target_std, dtype=np.float32), 1e-6)
    event = _event_window(truth_spikes, radius=4)
    non_event = ~event

    peak_errors = []
    for trajectory, step in zip(*np.where(truth_spikes)):
        low, high = max(0, step - 2), min(target.shape[1], step + 4)
        peak_errors.append(abs(float(prediction[trajectory, low:high, 1].max()) - float(target[trajectory, low:high, 1].max())))
    false_spikes = max(0, prediction_count - matched)
    return {
        "mean_state_nrmse": float(np.sqrt(np.mean(np.square(normalized), axis=(0, 1))).mean()),
        "slow_nrmse": float(np.sqrt(np.mean(np.square(normalized[..., 0])))),
        "voltage_nrmse": float(np.sqrt(np.mean(np.square(normalized[..., 1])))),
        "recovery_nrmse": float(np.sqrt(np.mean(np.square(normalized[..., 2])))),
        "event_voltage_rmse": float(np.sqrt(np.mean(np.square(error[..., 1][event])))) if event.any() else float("nan"),
        "subthreshold_voltage_rmse": float(np.sqrt(np.mean(np.square(error[..., 1][non_event])))) if non_event.any() else float("nan"),
        "peak_amplitude_mae": float(np.mean(peak_errors)) if peak_errors else float("nan"),
        "truth_spikes": truth_count,
        "predicted_spikes": prediction_count,
        "matched_spikes": matched,
        "spike_precision": float(precision),
        "spike_recall": float(recall),
        "spike_f1": float(f1),
        "timing_mae_steps": float(np.mean(np.abs(offsets))) if offsets else float("nan"),
        "false_spikes_per_1000_steps": float(1000 * false_spikes / prediction[..., 1].size),
        "trigger_accuracy": float(np.mean((prediction[..., 1][triggers] >= 0.50) == truth_spikes[triggers])) if triggers.any() else float("nan"),
    }


class _PointwiseMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, width: int) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Linear(input_dim, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU(), nn.Linear(width, output_dim))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class _Recurrent(nn.Module):
    def __init__(self, kind: str, input_dim: int, output_dim: int, width: int) -> None:
        super().__init__()
        recurrent = {"rnn": nn.RNN, "gru": nn.GRU, "lstm": nn.LSTM}[kind]
        self.input = nn.Sequential(nn.Linear(input_dim, width), nn.SiLU())
        self.recurrent = recurrent(width, width, batch_first=True)
        self.output = nn.Linear(width, output_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        sequence, _ = self.recurrent(self.input(inputs))
        return self.output(sequence)


class _CausalBlock(nn.Module):
    def __init__(self, width: int, dilation: int) -> None:
        super().__init__()
        self.dilation = dilation
        self.first = nn.Conv1d(width, width, 3, dilation=dilation)
        self.second = nn.Conv1d(width, width, 3, dilation=dilation)
        self.norm = nn.LayerNorm(width)

    def _convolve(self, layer: nn.Conv1d, values: torch.Tensor) -> torch.Tensor:
        return layer(F.pad(values, (2 * self.dilation, 0)))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = values
        values = F.silu(self._convolve(self.first, values))
        values = self._convolve(self.second, values)
        values = (values + residual).transpose(1, 2)
        return F.silu(self.norm(values)).transpose(1, 2)


class _TCN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, width: int) -> None:
        super().__init__()
        self.input = nn.Conv1d(input_dim, width, 1)
        self.blocks = nn.Sequential(*(_CausalBlock(width, dilation) for dilation in (1, 2, 4, 8)))
        self.output = nn.Conv1d(width, output_dim, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        values = self.output(self.blocks(self.input(inputs.transpose(1, 2))))
        return values.transpose(1, 2)


class _CausalTransformer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, width: int) -> None:
        super().__init__()
        heads = 4 if width % 4 == 0 else 2
        self.input = nn.Linear(input_dim, width)
        layer = nn.TransformerEncoderLayer(width, heads, 2 * width, dropout=0.0, batch_first=True, norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, 2)
        self.output = nn.Linear(width, output_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        steps = inputs.shape[1]
        mask = torch.triu(torch.ones(steps, steps, dtype=torch.bool, device=inputs.device), diagonal=1)
        position = torch.arange(steps, device=inputs.device, dtype=inputs.dtype)[None, :, None]
        scale = torch.exp(torch.arange(0, self.input.out_features, 2, device=inputs.device, dtype=inputs.dtype) * (-np.log(10000.0) / self.input.out_features))
        encoding = torch.zeros(1, steps, self.input.out_features, device=inputs.device, dtype=inputs.dtype)
        encoding[..., 0::2] = torch.sin(position * scale)
        encoding[..., 1::2] = torch.cos(position * scale[: encoding[..., 1::2].shape[-1]])
        return self.output(self.encoder(self.input(inputs) + encoding, mask=mask, is_causal=True))


class _Liquid(nn.Module):
    def __init__(self, kind: str, input_dim: int, output_dim: int, width: int) -> None:
        super().__init__()
        if kind == "cfc":
            from ncps.torch import CfC
            self.sequence = CfC(input_dim, width, return_sequences=True, batch_first=True, backbone_units=width, backbone_layers=1)
        else:
            from ncps.torch import LTC
            self.sequence = LTC(input_dim, width, return_sequences=True, batch_first=True, ode_unfolds=6)
        self.output = nn.Linear(width, output_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        values, _ = self.sequence(inputs)
        return self.output(values)


class _ConvRecurrent(nn.Module):
    def __init__(self, kind: str, input_dim: int, output_dim: int, width: int) -> None:
        super().__init__()
        channels = max(4, width // 2)
        self.first = nn.Conv1d(input_dim, channels, 3)
        self.second = nn.Conv1d(channels, channels, 3, dilation=2)
        recurrent = nn.GRU if kind == "conv_gru" else nn.LSTM
        self.recurrent = recurrent(channels, width, batch_first=True)
        self.output = nn.Linear(width, output_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        values = F.silu(self.first(F.pad(inputs.transpose(1, 2), (2, 0))))
        values = F.silu(self.second(F.pad(values, (4, 0)))).transpose(1, 2)
        values, _ = self.recurrent(values)
        return self.output(values)


CAPABILITY_ARCHITECTURES = (
    "mlp", "rnn", "gru", "lstm", "tcn", "transformer", "cfc", "ltc", "conv_gru", "conv_lstm",
)


def build_capability_model(name: str, input_dim: int, output_dim: int, width: int) -> nn.Module:
    if name == "mlp":
        return _PointwiseMLP(input_dim, output_dim, width)
    if name in {"rnn", "gru", "lstm"}:
        return _Recurrent(name, input_dim, output_dim, width)
    if name == "tcn":
        return _TCN(input_dim, output_dim, width)
    if name == "transformer":
        return _CausalTransformer(input_dim, output_dim, width)
    if name in {"cfc", "ltc"}:
        return _Liquid(name, input_dim, output_dim, width)
    if name in {"conv_gru", "conv_lstm"}:
        return _ConvRecurrent(name, input_dim, output_dim, width)
    raise ValueError(f"unknown capability architecture: {name}")


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def width_for_budget(name: str, budget: int, input_dim: int = 3, output_dim: int = 3) -> tuple[int, int]:
    def evaluate(widths: tuple[int, ...]) -> list[tuple[int, int, int]]:
        values = []
        for width in widths:
            if name == "transformer" and width % 4:
                continue
            model = build_capability_model(name, input_dim, output_dim, width)
            parameters = count_parameters(model)
            values.append((abs(parameters - budget), width, parameters))
        return values

    coarse = tuple(range(8, 257, 16)) + (256,)
    choices = evaluate(coarse)
    _, center, _ = min(choices)
    local = tuple(range(max(8, center - 16), min(256, center + 16) + 1, 4))
    choices.extend(evaluate(local))
    _, width, parameters = min(choices)
    return width, parameters
