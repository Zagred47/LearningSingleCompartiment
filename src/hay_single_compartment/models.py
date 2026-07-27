"""Sequence baselines for learning the compartment state transition."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .ontology import ONTOLOGY_GROUPS


class CausalConv1d(nn.Conv1d):
    """One-dimensional convolution that never reads future time steps."""

    def forward(self, inputs):
        left_padding = self.dilation[0] * (self.kernel_size[0] - 1)
        return super().forward(F.pad(inputs, (left_padding, 0)))


class RecurrentSurrogate(nn.Module):
    """Residual LSTM/GRU/RNN state predictor in normalized coordinates."""

    def __init__(
        self,
        input_dim: int,
        state_dim: int,
        hidden_dim: int = 128,
        layers: int = 2,
        architecture: str = "lstm",
        dropout: float = 0.1,
        **_: object,
    ) -> None:
        super().__init__()
        architecture = architecture.lower()
        recurrent = {"lstm": nn.LSTM, "gru": nn.GRU, "rnn": nn.RNN}.get(architecture)
        if recurrent is None:
            raise ValueError("architecture must be lstm, gru, or rnn")
        self.state_dim = state_dim
        self.architecture = architecture
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.SiLU())
        self.recurrent = recurrent(
            hidden_dim,
            hidden_dim,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, state_dim)
        )

    def forward(self, features, hidden=None, return_hidden: bool = False):
        encoded = self.encoder(features)
        memory, hidden = self.recurrent(encoded, hidden)
        prediction = features[..., : self.state_dim] + self.head(memory)
        return (prediction, hidden) if return_hidden else prediction


class MLPSurrogate(nn.Module):
    """Memoryless residual baseline applied independently at every time step."""

    def __init__(self, input_dim: int, state_dim: int, hidden_dim: int = 128, **_: object) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.architecture = "mlp"
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(self, features, hidden=None, return_hidden: bool = False):
        prediction = features[..., : self.state_dim] + self.network(features)
        return (prediction, None) if return_hidden else prediction


class ConvLSTMSurrogate(nn.Module):
    """Wide causal temporal convolution front-end followed by an LSTM.

    ``hidden_dim`` is intentionally expanded twofold inside this architecture.
    With the Kaggle setting (96, two layers) the model has roughly one million
    parameters.  Raw feature history is carried during step-wise rollout so
    causal convolutions behave exactly as they do on complete sequences.
    """

    def __init__(
        self,
        input_dim: int,
        state_dim: int,
        hidden_dim: int = 128,
        layers: int = 2,
        dropout: float = 0.1,
        width_multiplier: int = 2,
        **_: object,
    ) -> None:
        super().__init__()
        if width_multiplier < 1:
            raise ValueError("width_multiplier must be positive")
        width = hidden_dim * width_multiplier
        self.state_dim = state_dim
        self.architecture = "conv_lstm"
        self.context_steps = 20
        self.conv_in = CausalConv1d(input_dim, width, kernel_size=5, dilation=1)
        self.conv_mid = CausalConv1d(width, width, kernel_size=5, dilation=2)
        self.conv_out = CausalConv1d(width, width, kernel_size=3, dilation=4)
        self.norm_in = nn.LayerNorm(width)
        self.norm_mid = nn.LayerNorm(width)
        self.norm_out = nn.LayerNorm(width)
        self.recurrent = nn.LSTM(
            width,
            width,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(width, width), nn.SiLU(), nn.Linear(width, state_dim)
        )

    @staticmethod
    def _time_norm(values, norm):
        return norm(values.transpose(1, 2)).transpose(1, 2)

    def _encode(self, features):
        values = features.transpose(1, 2)
        values = F.silu(self._time_norm(self.conv_in(values), self.norm_in))
        residual = values
        values = F.silu(self._time_norm(self.conv_mid(values), self.norm_mid))
        values = values + residual
        residual = values
        values = F.silu(self._time_norm(self.conv_out(values), self.norm_out))
        return (values + residual).transpose(1, 2)

    def forward(self, features, hidden=None, return_hidden: bool = False):
        recurrent_hidden = None
        history = None
        if isinstance(hidden, dict):
            recurrent_hidden = hidden.get("recurrent")
            history = hidden.get("conv_history")
        conv_features = (
            torch.cat([history, features], dim=1) if history is not None else features
        )
        encoded = self._encode(conv_features)[:, -features.shape[1] :]
        memory, recurrent_hidden = self.recurrent(encoded, recurrent_hidden)
        prediction = features[..., : self.state_dim] + self.head(memory)
        if not return_hidden:
            return prediction
        next_hidden = {
            "recurrent": recurrent_hidden,
            "conv_history": conv_features[:, -self.context_steps :].detach(),
        }
        return prediction, next_hidden


class OntologyGRUMosaic(nn.Module):
    """Causally factorized flow map built only from standard GRU layers.

    Every ontology group owns an independent ``torch.nn.GRU`` and residual
    output head.  The novelty tested here is only sparse causal factorization;
    the recurrent primitive itself is unchanged and well established.
    """

    def __init__(
        self,
        input_dim: int,
        state_dim: int,
        hidden_dim: int = 48,
        layers: int = 2,
        dropout: float = 0.1,
        **_: object,
    ) -> None:
        super().__init__()
        expected_input = max(index for group in ONTOLOGY_GROUPS for index in group.feature_indices) + 1
        expected_state = max(index for group in ONTOLOGY_GROUPS for index in group.output_indices) + 1
        if input_dim != expected_input or state_dim != expected_state:
            raise ValueError(
                f"ontology_gru requires input_dim={expected_input} and state_dim={expected_state}"
            )
        self.state_dim = state_dim
        self.architecture = "ontology_gru"
        self.groups = ONTOLOGY_GROUPS
        self.encoders = nn.ModuleDict()
        self.recurrents = nn.ModuleDict()
        self.heads = nn.ModuleDict()
        for group in self.groups:
            self.encoders[group.name] = nn.Sequential(
                nn.Linear(len(group.feature_indices), hidden_dim), nn.SiLU()
            )
            self.recurrents[group.name] = nn.GRU(
                hidden_dim,
                hidden_dim,
                num_layers=layers,
                batch_first=True,
                dropout=dropout if layers > 1 else 0.0,
            )
            self.heads[group.name] = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, len(group.output_indices)),
            )

    def forward(self, features, hidden=None, return_hidden: bool = False):
        hidden = hidden if isinstance(hidden, dict) else {}
        next_hidden = {}
        prediction = features[..., : self.state_dim].clone()
        for group in self.groups:
            local_features = features[..., list(group.feature_indices)]
            encoded = self.encoders[group.name](local_features)
            memory, group_hidden = self.recurrents[group.name](
                encoded, hidden.get(group.name)
            )
            output_indices = list(group.output_indices)
            prediction[..., output_indices] = (
                features[..., output_indices] + self.heads[group.name](memory)
            )
            next_hidden[group.name] = group_hidden
        return (prediction, next_hidden) if return_hidden else prediction


def build_model(architecture: str, input_dim: int, state_dim: int, **kwargs):
    architecture = architecture.lower()
    if architecture == "mlp":
        return MLPSurrogate(input_dim, state_dim, **kwargs)
    if architecture in {"conv_lstm", "convlstm", "cnn_lstm"}:
        return ConvLSTMSurrogate(input_dim, state_dim, **kwargs)
    if architecture in {"ontology_gru", "causal_gru", "gru_mosaic"}:
        return OntologyGRUMosaic(input_dim, state_dim, **kwargs)
    return RecurrentSurrogate(input_dim, state_dim, architecture=architecture, **kwargs)
