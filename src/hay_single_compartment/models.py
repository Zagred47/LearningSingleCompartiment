"""Sequence baselines for learning the compartment state transition."""

from __future__ import annotations

import torch
from torch import nn


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


def build_model(architecture: str, input_dim: int, state_dim: int, **kwargs):
    if architecture.lower() == "mlp":
        return MLPSurrogate(input_dim, state_dim, **kwargs)
    return RecurrentSurrogate(input_dim, state_dim, architecture=architecture, **kwargs)
