"""Input-only recurrent surrogates for the four-compartment teacher.

These models never consume teacher state.  A separate spike-only burn-in is
used to construct the recurrent state before supervised prediction begins.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class StateDecoder(nn.Module):
    def __init__(self, hidden_dim: int, state_dim: int, decoder_dim: int | None = None) -> None:
        super().__init__()
        width = decoder_dim or hidden_dim
        self.network = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, width),
            nn.SiLU(),
            nn.Linear(width, state_dim),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.network(hidden)


class InputOnlyGRU(nn.Module):
    """Standard GRU baseline with a state decoder and no state feedback."""

    architecture = "gru_input_only"

    def __init__(
        self,
        input_dim: int,
        state_dim: int,
        hidden_dim: int = 256,
        layers: int = 1,
        dropout: float = 0.0,
        decoder_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.input_encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.SiLU())
        self.recurrent = nn.GRU(
            hidden_dim,
            hidden_dim,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.decoder = StateDecoder(hidden_dim, state_dim, decoder_dim)

    def forward(self, inputs: torch.Tensor, hidden: torch.Tensor | None = None, timespans: torch.Tensor | None = None):
        del timespans
        sequence, hidden = self.recurrent(self.input_encoder(inputs), hidden)
        return self.decoder(sequence), hidden

    def decode_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.decoder(hidden[-1])

    @staticmethod
    def detach_hidden(hidden: torch.Tensor | None) -> torch.Tensor | None:
        return None if hidden is None else hidden.detach()


class InputOnlyCfC(nn.Module):
    """Adapter around the published ``ncps.torch.CfC`` layer.

    Import is lazy so dataset generation and the GRU control do not require
    the optional ncps package.
    """

    architecture = "cfc_input_only"

    def __init__(
        self,
        input_dim: int,
        state_dim: int,
        hidden_dim: int = 192,
        input_embedding_dim: int = 128,
        backbone_units: int = 192,
        backbone_layers: int = 2,
        backbone_dropout: float = 0.0,
        decoder_dim: int | None = None,
        mode: str = "default",
    ) -> None:
        super().__init__()
        try:
            from ncps.torch import CfC
        except ImportError as error:
            raise ImportError("InputOnlyCfC requires `pip install ncps==1.0.1`") from error
        self.hidden_dim = hidden_dim
        self.input_encoder = nn.Sequential(
            nn.Linear(input_dim, input_embedding_dim), nn.SiLU()
        )
        self.recurrent = CfC(
            input_size=input_embedding_dim,
            units=hidden_dim,
            return_sequences=True,
            batch_first=True,
            mixed_memory=False,
            mode=mode,
            backbone_units=backbone_units,
            backbone_layers=backbone_layers,
            backbone_dropout=backbone_dropout,
        )
        self.decoder = StateDecoder(hidden_dim, state_dim, decoder_dim)

    def forward(self, inputs: torch.Tensor, hidden: torch.Tensor | None = None, timespans: torch.Tensor | None = None):
        # ncps 1.0.1 squeezes a batched timespan vector to [B], which does not
        # broadcast against [B,H].  For constant-rate data, one CfC time unit
        # is therefore defined as one caller step.  Reject irregular sampling
        # rather than silently applying the package's broken batched path.
        if timespans is not None:
            reference = timespans.reshape(-1)[0]
            if not torch.allclose(timespans, reference.expand_as(timespans)):
                raise ValueError("ncps 1.0.1 CfC adapter supports constant timespans only")
        sequence, hidden = self.recurrent(self.input_encoder(inputs), hx=hidden)
        return self.decoder(sequence), hidden

    def decode_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.decoder(hidden)

    @staticmethod
    def detach_hidden(hidden: torch.Tensor | None) -> torch.Tensor | None:
        return None if hidden is None else hidden.detach()


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def move_hidden(hidden: Any, device: torch.device | str) -> Any:
    if hidden is None:
        return None
    if isinstance(hidden, tuple):
        return tuple(move_hidden(item, device) for item in hidden)
    return hidden.to(device)
