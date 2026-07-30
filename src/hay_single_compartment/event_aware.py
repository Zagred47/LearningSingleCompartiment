"""Event-aware sampling, losses and diagnostics for the micro-Hay surrogate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


MICRO_EVENT_NAMES = (
    "subthreshold",
    "near_threshold",
    "isolated_spike",
    "burst_spike",
    "rapid_fire",
    "tuft_plateau",
    "spike_with_tuft_plateau",
    "post_spike_recovery",
)


@dataclass(frozen=True)
class MicroEventConfig:
    spike_threshold_mV: float = -20.0
    near_threshold_low_mV: float = -40.0
    tuft_plateau_threshold_mV: float = -35.0
    burst_isi_ms: float = 20.0
    rapid_window_ms: float = 50.0
    rapid_min_spikes: int = 3
    event_radius_ms: float = 5.0
    recovery_ms: float = 20.0
    minimum_plateau_ms: float = 10.0


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.copy()
    kernel = np.ones(2 * radius + 1, dtype=np.int32)
    return np.convolve(mask.astype(np.int32), kernel, mode="same") > 0


def _minimum_run_mask(mask: np.ndarray, minimum: int) -> np.ndarray:
    result = np.zeros_like(mask, dtype=bool)
    padded = np.pad(mask.astype(np.int8), (1, 1))
    changes = np.flatnonzero(np.diff(padded))
    for start, stop in changes.reshape(-1, 2):
        if stop - start >= minimum:
            result[start:stop] = True
    return result


def classify_micro_events(
    states: np.ndarray,
    spikes: np.ndarray,
    state_names: Sequence[str],
    dt_ms: float,
    config: MicroEventConfig | None = None,
) -> np.ndarray:
    """Return multi-label event masks with shape ``[trajectory, time, event]``.

    Labels describe the teacher response and may overlap.  This is deliberate:
    for example, a somatic spike can also occur during a tuft plateau.
    """

    config = config or MicroEventConfig()
    states = np.asarray(states)
    spikes = np.asarray(spikes, dtype=bool)
    if states.ndim != 3 or spikes.shape != states.shape[:2]:
        raise ValueError("states and spikes must have shapes [N,T,S] and [N,T]")
    if dt_ms <= 0:
        raise ValueError("dt_ms must be positive")
    names = list(state_names)
    soma_index, tuft_index = names.index("soma.v_mV"), names.index("tuft.v_mV")
    soma, tuft = states[..., soma_index], states[..., tuft_index]
    labels = np.zeros(spikes.shape + (len(MICRO_EVENT_NAMES),), dtype=np.uint8)
    event_radius = max(1, int(round(config.event_radius_ms / dt_ms)))
    burst_isi = max(1, int(round(config.burst_isi_ms / dt_ms)))
    rapid_window = max(1, int(round(config.rapid_window_ms / dt_ms)))
    recovery = max(1, int(round(config.recovery_ms / dt_ms)))
    plateau_minimum = max(1, int(round(config.minimum_plateau_ms / dt_ms)))

    for trajectory in range(states.shape[0]):
        spike_indices = np.flatnonzero(spikes[trajectory])
        burst = np.zeros(spikes.shape[1], dtype=bool)
        rapid = np.zeros_like(burst)
        recovery_mask = np.zeros_like(burst)
        if spike_indices.size:
            isi = np.diff(spike_indices)
            close = np.flatnonzero(isi <= burst_isi)
            if close.size:
                burst[spike_indices[close]] = True
                burst[spike_indices[close + 1]] = True
            left = 0
            for right in range(spike_indices.size):
                while spike_indices[right] - spike_indices[left] > rapid_window:
                    left += 1
                if right - left + 1 >= config.rapid_min_spikes:
                    rapid[spike_indices[left] : spike_indices[right] + 1] = True
            for index in spike_indices:
                recovery_mask[index + 1 : min(len(recovery_mask), index + recovery + 1)] = True
        isolated = spikes[trajectory] & ~burst
        plateau = _minimum_run_mask(
            tuft[trajectory, : spikes.shape[1]] >= config.tuft_plateau_threshold_mV,
            plateau_minimum,
        )
        spike_window = _dilate(spikes[trajectory], event_radius)
        labels[trajectory, :, 1] = (
            (soma[trajectory, : spikes.shape[1]] >= config.near_threshold_low_mV)
            & ~spike_window
        )
        labels[trajectory, :, 2] = _dilate(isolated, event_radius)
        labels[trajectory, :, 3] = _dilate(burst, event_radius)
        labels[trajectory, :, 4] = _dilate(rapid, event_radius)
        labels[trajectory, :, 5] = plateau
        labels[trajectory, :, 6] = plateau & spike_window
        labels[trajectory, :, 7] = recovery_mask
        labels[trajectory, :, 0] = ~labels[trajectory, :, 1:].any(axis=-1)
    return labels


def event_catalog(labels: np.ndarray) -> dict[str, int]:
    labels = np.asarray(labels)
    if labels.ndim != 3 or labels.shape[-1] != len(MICRO_EVENT_NAMES):
        raise ValueError("labels have an incompatible shape")
    return {name: int(labels[..., index].sum()) for index, name in enumerate(MICRO_EVENT_NAMES)}


class StratifiedWindowSampler:
    """Sample full contiguous windows from a declared event mixture."""

    def __init__(
        self,
        event_labels: np.ndarray,
        window_steps: int,
        mixture: Mapping[str, float] | None = None,
        seed: int = 0,
    ) -> None:
        labels = np.asarray(event_labels)
        if labels.ndim != 3 or labels.shape[-1] != len(MICRO_EVENT_NAMES):
            raise ValueError("event_labels have an incompatible shape")
        if window_steps < 1 or window_steps > labels.shape[1]:
            raise ValueError("window_steps are outside the trajectory length")
        self.labels, self.window_steps = labels, window_steps
        self.rng = np.random.default_rng(seed)
        mixture = mixture or {
            "subthreshold": 0.50,
            "near_threshold": 0.05,
            "isolated_spike": 0.15,
            "burst_spike": 0.10,
            "rapid_fire": 0.05,
            "tuft_plateau": 0.10,
            "spike_with_tuft_plateau": 0.05,
        }
        unknown = set(mixture) - set(MICRO_EVENT_NAMES)
        if unknown or any(value < 0 for value in mixture.values()) or sum(mixture.values()) <= 0:
            raise ValueError(f"invalid event mixture; unknown={sorted(unknown)}")
        self.names = tuple(mixture)
        probabilities = np.asarray([mixture[name] for name in self.names], dtype=float)
        self.probabilities = probabilities / probabilities.sum()
        self.locations = {
            name: np.argwhere(labels[..., MICRO_EVENT_NAMES.index(name)] > 0)
            for name in self.names
        }

    def sample(self, count: int) -> np.ndarray:
        """Return integer ``[count,2]`` rows of trajectory and start index."""

        if count < 1:
            raise ValueError("count must be positive")
        result = np.empty((count, 2), dtype=np.int64)
        chosen = self.rng.choice(len(self.names), size=count, p=self.probabilities)
        maximum_start = self.labels.shape[1] - self.window_steps
        for row, event_index in enumerate(chosen):
            locations = self.locations[self.names[event_index]]
            if not len(locations):
                trajectory = int(self.rng.integers(self.labels.shape[0]))
                start = int(self.rng.integers(maximum_start + 1))
            else:
                trajectory, center = locations[int(self.rng.integers(len(locations)))]
                jitter = int(self.rng.integers(-self.window_steps // 4, self.window_steps // 4 + 1))
                start = int(np.clip(center - self.window_steps // 2 + jitter, 0, maximum_start))
            result[row] = trajectory, start
        return result


class EventAwareStateLoss(nn.Module):
    """State reconstruction plus differentiable spike/shape emphasis.

    Predictions and targets are normalized states.  Voltage-specific terms are
    evaluated in mV after applying the stored normalization.
    """

    def __init__(
        self,
        state_names: Sequence[str],
        state_mean: np.ndarray | torch.Tensor,
        state_std: np.ndarray | torch.Tensor,
        *,
        global_weight: float = 1.0,
        event_voltage_weight: float = 0.25,
        derivative_weight: float = 0.10,
        rapid_gate_weight: float = 0.10,
        soft_spike_weight: float = 0.10,
        spike_threshold_mV: float = -20.0,
        event_radius_steps: int = 10,
        soft_temperature_mV: float = 2.0,
    ) -> None:
        super().__init__()
        names = list(state_names)
        self.soma_index = names.index("soma.v_mV")
        self.rapid_gate_indices = tuple(
            i for i, name in enumerate(names)
            if name in {
                "soma.m_NaTa_t", "soma.h_NaTa_t", "soma.m_K_Tst",
                "soma.h_K_Tst", "soma.m_SKv3_1", "soma.m_Ca_HVA",
            }
        )
        self.register_buffer("state_mean", torch.as_tensor(state_mean, dtype=torch.float32))
        self.register_buffer("state_std", torch.as_tensor(state_std, dtype=torch.float32))
        self.global_weight = global_weight
        self.event_voltage_weight = event_voltage_weight
        self.derivative_weight = derivative_weight
        self.rapid_gate_weight = rapid_gate_weight
        self.soft_spike_weight = soft_spike_weight
        self.spike_threshold_mV = spike_threshold_mV
        self.event_radius_steps = event_radius_steps
        self.soft_temperature_mV = soft_temperature_mV

    def _physical(self, values: torch.Tensor) -> torch.Tensor:
        return values * self.state_std + self.state_mean

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if prediction.shape != target.shape:
            raise ValueError("prediction and target shapes differ")
        global_loss = F.mse_loss(prediction, target)
        prediction_physical, target_physical = self._physical(prediction), self._physical(target)
        pv = prediction_physical[..., self.soma_index]
        tv = target_physical[..., self.soma_index]
        crossing = torch.zeros_like(tv)
        crossing[..., 1:] = (
            (tv[..., :-1] < self.spike_threshold_mV)
            & (tv[..., 1:] >= self.spike_threshold_mV)
        ).to(tv.dtype)
        radius = self.event_radius_steps
        event_mask = F.max_pool1d(crossing.unsqueeze(1), 2 * radius + 1, stride=1, padding=radius).squeeze(1)
        event_denominator = event_mask.sum().clamp_min(1.0)
        event_voltage = (((pv - tv) / 20.0).square() * event_mask).sum() / event_denominator
        derivative = F.mse_loss((pv[..., 1:] - pv[..., :-1]) / 20.0, (tv[..., 1:] - tv[..., :-1]) / 20.0)
        if self.rapid_gate_indices:
            indices = torch.as_tensor(self.rapid_gate_indices, device=prediction.device)
            gate_error = (prediction.index_select(-1, indices) - target.index_select(-1, indices)).square().mean(-1)
            rapid_gate = (gate_error * event_mask).sum() / event_denominator
        else:
            rapid_gate = global_loss.new_zeros(())
        soft_logits = (pv - self.spike_threshold_mV) / self.soft_temperature_mV
        soft_spike = F.binary_cross_entropy_with_logits(
            soft_logits,
            (tv >= self.spike_threshold_mV).to(tv.dtype),
        )
        terms = {
            "global": global_loss,
            "event_voltage": event_voltage,
            "derivative": derivative,
            "rapid_gate": rapid_gate,
            "soft_spike": soft_spike,
        }
        total = (
            self.global_weight * global_loss
            + self.event_voltage_weight * event_voltage
            + self.derivative_weight * derivative
            + self.rapid_gate_weight * rapid_gate
            + self.soft_spike_weight * soft_spike
        )
        return total, terms


@torch.no_grad()
def replay_gru_gates(model: nn.Module, inputs: torch.Tensor, hidden: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    """Expose standard PyTorch GRU gates and verifyable hidden trajectories."""

    encoded = model.input_encoder(inputs)
    gru = model.recurrent
    if gru.num_layers != 1 or gru.bidirectional:
        raise ValueError("diagnostic replay currently supports one unidirectional GRU layer")
    batch, steps, _ = encoded.shape
    h = encoded.new_zeros(batch, gru.hidden_size) if hidden is None else hidden[-1].clone()
    w_ir, w_iz, w_in = gru.weight_ih_l0.chunk(3)
    w_hr, w_hz, w_hn = gru.weight_hh_l0.chunk(3)
    b_ir, b_iz, b_in = gru.bias_ih_l0.chunk(3)
    b_hr, b_hz, b_hn = gru.bias_hh_l0.chunk(3)
    records = {name: [] for name in ("reset", "update", "candidate", "hidden")}
    for step in range(steps):
        x = encoded[:, step]
        reset = torch.sigmoid(F.linear(x, w_ir, b_ir) + F.linear(h, w_hr, b_hr))
        update = torch.sigmoid(F.linear(x, w_iz, b_iz) + F.linear(h, w_hz, b_hz))
        candidate = torch.tanh(F.linear(x, w_in, b_in) + reset * F.linear(h, w_hn, b_hn))
        h = (1.0 - update) * candidate + update * h
        for name, value in (("reset", reset), ("update", update), ("candidate", candidate), ("hidden", h)):
            records[name].append(value)
    return {name: torch.stack(values, dim=1) for name, values in records.items()}
