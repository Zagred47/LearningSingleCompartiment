"""Input-only recurrent surrogates for the four-compartment teacher.

These models never consume teacher state.  A separate spike-only burn-in is
used to construct the recurrent state before supervised prediction begins.
"""

from __future__ import annotations

from typing import Any

import math

import torch
from torch import nn
from torch.nn import functional as F


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


class _CausalConvRecurrent(nn.Module):
    """Causal temporal Conv1d front-end followed by a standard GRU or LSTM."""

    def __init__(
        self,
        input_dim: int,
        state_dim: int,
        hidden_dim: int,
        conv_channels: int,
        dilations: tuple[int, ...],
        recurrent_kind: str,
        kernel_size: int = 3,
        decoder_dim: int | None = None,
    ) -> None:
        super().__init__()
        if not dilations or kernel_size < 2:
            raise ValueError("dilations must be nonempty and kernel_size >= 2")
        if recurrent_kind not in {"gru", "lstm"}:
            raise ValueError("recurrent_kind must be gru or lstm")
        self.hidden_dim, self.recurrent_kind = hidden_dim, recurrent_kind
        self.dilations, self.kernel_size = tuple(dilations), kernel_size
        self.receptive_field = 1 + (kernel_size - 1) * sum(dilations)
        convolutions = []
        channels = input_dim
        for dilation in dilations:
            convolutions.extend((
                nn.Conv1d(channels, conv_channels, kernel_size, dilation=dilation),
                nn.SiLU(),
            ))
            channels = conv_channels
        self.frontend = nn.Sequential(*convolutions)
        recurrent_class = nn.GRU if recurrent_kind == "gru" else nn.LSTM
        self.recurrent = recurrent_class(conv_channels, hidden_dim, batch_first=True)
        self.decoder = StateDecoder(hidden_dim, state_dim, decoder_dim)

    def forward(self, inputs: torch.Tensor, hidden: Any = None, timespans: torch.Tensor | None = None):
        del timespans
        recurrent_hidden, cache = (None, None) if hidden is None else hidden
        keep = self.receptive_field - 1
        if cache is None:
            cache = inputs.new_zeros(inputs.shape[0], keep, inputs.shape[-1])
        combined = torch.cat((cache, inputs), dim=1)
        features = self.frontend(combined.transpose(1, 2)).transpose(1, 2)
        sequence, recurrent_hidden = self.recurrent(features, recurrent_hidden)
        next_cache = combined[:, -keep:] if keep else combined[:, :0]
        return self.decoder(sequence), (recurrent_hidden, next_cache)

    def decode_hidden(self, hidden: Any) -> torch.Tensor:
        recurrent_hidden = hidden[0]
        if self.recurrent_kind == "lstm":
            recurrent_hidden = recurrent_hidden[0]
        return self.decoder(recurrent_hidden[-1])

    @staticmethod
    def detach_hidden(hidden: Any) -> Any:
        if hidden is None:
            return None
        recurrent_hidden, cache = hidden
        if isinstance(recurrent_hidden, tuple):
            recurrent_hidden = tuple(value.detach() for value in recurrent_hidden)
        else:
            recurrent_hidden = recurrent_hidden.detach()
        return recurrent_hidden, cache.detach()


class InputOnlyConvGRU(_CausalConvRecurrent):
    architecture = "causal_conv_gru_input_only"

    def __init__(self, input_dim: int, state_dim: int, hidden_dim: int = 192, conv_channels: int = 128,
                 dilations: tuple[int, ...] = (1, 2, 4), kernel_size: int = 3,
                 decoder_dim: int | None = None) -> None:
        super().__init__(input_dim, state_dim, hidden_dim, conv_channels, dilations, "gru", kernel_size, decoder_dim)


class InputOnlyConvLSTM(_CausalConvRecurrent):
    architecture = "causal_conv_lstm_input_only"

    def __init__(self, input_dim: int, state_dim: int, hidden_dim: int = 192, conv_channels: int = 128,
                 dilations: tuple[int, ...] = (1, 2, 4), kernel_size: int = 3,
                 decoder_dim: int | None = None) -> None:
        super().__init__(input_dim, state_dim, hidden_dim, conv_channels, dilations, "lstm", kernel_size, decoder_dim)


class InputOnlyBranchELM(nn.Module):
    """Branch ELM port using the published GIADA recurrence and pure spike input."""

    architecture = "branch_elm_input_only"

    def __init__(
        self,
        input_dim: int,
        state_dim: int,
        num_branch: int = 24,
        num_memory: int = 128,
        mlp_hidden_dim: int | None = None,
        branch_tau_ms: float = 5.0,
        memory_tau_min_ms: float = 1.0,
        memory_tau_max_ms: float = 1000.0,
        model_dt_ms: float = 0.5,
        lambda_value: float = 5.0,
    ) -> None:
        super().__init__()
        if input_dim % num_branch:
            raise ValueError("input_dim must be divisible by num_branch")
        self.input_dim, self.num_branch, self.num_memory = input_dim, num_branch, num_memory
        self.inputs_per_branch = input_dim // num_branch
        self.lambda_value = lambda_value
        width = mlp_hidden_dim or 2 * num_memory
        self._proto_input_weight = nn.Parameter(torch.full((input_dim,), 0.5))
        self.update = nn.Sequential(
            nn.Linear(num_branch + num_memory, width), nn.ReLU(),
            nn.Linear(width, num_memory),
        )
        self.decoder = StateDecoder(num_memory, state_dim, num_memory)
        tau_memory = torch.logspace(math.log10(memory_tau_min_ms), math.log10(memory_tau_max_ms), num_memory)
        self.register_buffer("branch_decay", torch.tensor(math.exp(-model_dt_ms / branch_tau_ms)))
        self.register_buffer("memory_decay", torch.exp(-model_dt_ms / tau_memory))

    @property
    def input_weight(self) -> torch.Tensor:
        return F.relu(self._proto_input_weight)

    def _route(self, values: torch.Tensor) -> torch.Tensor:
        # Packed inputs are ordered [microbin, synapse]; transpose to branch traces.
        shape = values.shape[:-1]
        routed = values.reshape(*shape, self.inputs_per_branch, self.num_branch)
        weights = self.input_weight.reshape(self.inputs_per_branch, self.num_branch)
        return (routed * weights).sum(dim=-2)

    def forward(self, inputs: torch.Tensor, hidden: Any = None, timespans: torch.Tensor | None = None):
        del timespans
        batch = inputs.shape[0]
        if hidden is None:
            branch = inputs.new_zeros(batch, self.num_branch)
            memory = inputs.new_zeros(batch, self.num_memory)
        else:
            branch, memory = hidden
        outputs = []
        routed = self._route(inputs)
        for step in range(inputs.shape[1]):
            branch = self.branch_decay * branch + routed[:, step]
            proposal = torch.tanh(self.update(torch.cat((branch, self.memory_decay * memory), dim=-1)))
            memory = self.memory_decay * memory + self.lambda_value * (1.0 - self.memory_decay) * proposal
            outputs.append(self.decoder(memory))
        return torch.stack(outputs, dim=1), (branch, memory)

    def decode_hidden(self, hidden: Any) -> torch.Tensor:
        return self.decoder(hidden[1])

    @staticmethod
    def detach_hidden(hidden: Any) -> Any:
        return None if hidden is None else tuple(value.detach() for value in hidden)


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
