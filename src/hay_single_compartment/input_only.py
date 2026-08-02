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


class StateContextGRU(nn.Module):
    """Standard GRUCell with a controlled normalized physical-state channel.

    ``mode`` is the only experimental factor:

    - ``none`` always supplies zeros on the state channel;
    - ``initial_only`` supplies the caller's state only on the first step;
    - ``predicted_feedback`` supplies the first state once, then recursively
      supplies the model's own previous normalized prediction.

    No mode consumes a teacher state after trajectory initialization.  All
    modes instantiate exactly the same trainable modules and parameter count.
    """

    architecture = "standard_gru_controlled_state_context"
    valid_modes = ("none", "initial_only", "predicted_feedback")

    def __init__(
        self,
        input_dim: int,
        state_dim: int,
        hidden_dim: int = 200,
        decoder_dim: int | None = None,
        mode: str = "none",
    ) -> None:
        super().__init__()
        if mode not in self.valid_modes:
            raise ValueError(f"mode must be one of {self.valid_modes}, got {mode!r}")
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.mode = mode
        self.input_encoder = nn.Sequential(
            nn.Linear(input_dim + state_dim, hidden_dim), nn.SiLU()
        )
        self.recurrent = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.decoder = StateDecoder(hidden_dim, state_dim, decoder_dim)

    def _gru_step(self, encoded: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        """Execute the exact one-layer PyTorch GRU equations for feedback mode."""
        input_terms = F.linear(
            encoded, self.recurrent.weight_ih_l0, self.recurrent.bias_ih_l0
        )
        hidden_terms = F.linear(
            hidden, self.recurrent.weight_hh_l0, self.recurrent.bias_hh_l0
        )
        input_reset, input_update, input_candidate = input_terms.chunk(3, dim=-1)
        hidden_reset, hidden_update, hidden_candidate = hidden_terms.chunk(3, dim=-1)
        reset = torch.sigmoid(input_reset + hidden_reset)
        update = torch.sigmoid(input_update + hidden_update)
        candidate = torch.tanh(input_candidate + reset * hidden_candidate)
        return (1.0 - update) * candidate + update * hidden

    def forward(
        self,
        inputs: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
        initial_state: torch.Tensor | None = None,
        timespans: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        del timespans
        if inputs.ndim != 3 or inputs.shape[-1] != self.input_dim:
            raise ValueError(
                f"inputs must have shape [batch,time,{self.input_dim}], got {tuple(inputs.shape)}"
            )
        batch = inputs.shape[0]
        first_call = hidden is None
        if first_call:
            recurrent_hidden = inputs.new_zeros(batch, self.hidden_dim)
            if initial_state is None:
                state_context = inputs.new_zeros(batch, self.state_dim)
            else:
                if initial_state.shape != (batch, self.state_dim):
                    raise ValueError(
                        "initial_state must have shape "
                        f"{(batch, self.state_dim)}, got {tuple(initial_state.shape)}"
                    )
                state_context = initial_state
        else:
            if initial_state is not None:
                raise ValueError("initial_state may only be supplied when hidden is None")
            recurrent_hidden, state_context = hidden

        zero_context = inputs.new_zeros(batch, self.state_dim)
        if self.mode != "predicted_feedback":
            contexts = inputs.new_zeros(batch, inputs.shape[1], self.state_dim)
            if self.mode == "initial_only" and first_call and inputs.shape[1]:
                contexts[:, 0] = state_context
            encoded = self.input_encoder(torch.cat((inputs, contexts), dim=-1))
            sequence, recurrent_sequence_hidden = self.recurrent(
                encoded, recurrent_hidden.unsqueeze(0)
            )
            prediction = self.decoder(sequence)
            return prediction, (recurrent_sequence_hidden[0], zero_context)

        outputs = []
        for step in range(inputs.shape[1]):
            encoded = self.input_encoder(
                torch.cat((inputs[:, step], state_context), dim=-1)
            )
            recurrent_hidden = self._gru_step(encoded, recurrent_hidden)
            prediction = self.decoder(recurrent_hidden)
            outputs.append(prediction)
            state_context = prediction
        if outputs:
            sequence = torch.stack(outputs, dim=1)
        else:
            sequence = inputs.new_empty(batch, 0, self.state_dim)
        return sequence, (recurrent_hidden, state_context)

    def decode_hidden(self, hidden: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        return self.decoder(hidden[0])

    @staticmethod
    def detach_hidden(
        hidden: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if hidden is None:
            return None
        return tuple(value.detach() for value in hidden)


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


class CausalResidualTCNAdapter(nn.Module):
    """Standard causal dilated Conv1d stack for a sequence residual.

    The final 1x1 convolution is zero-initialized, so attaching the adapter to
    an existing surrogate preserves that surrogate exactly at initialization.
    The cache contains only past input features and makes chunked inference
    equivalent to a single full-sequence call.
    """

    def __init__(
        self,
        feature_dim: int,
        state_dim: int,
        channels: int = 96,
        dilations: tuple[int, ...] = (1, 2, 4, 8),
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        if not dilations or any(dilation < 1 for dilation in dilations):
            raise ValueError("dilations must contain positive integers")
        if kernel_size < 2:
            raise ValueError("kernel_size must be at least 2")
        self.feature_dim = feature_dim
        self.state_dim = state_dim
        self.channels = channels
        self.dilations = tuple(dilations)
        self.kernel_size = kernel_size
        self.receptive_field = 1 + (kernel_size - 1) * sum(self.dilations)
        layers: list[nn.Module] = []
        input_channels = feature_dim
        for dilation in self.dilations:
            layers.extend((
                nn.Conv1d(
                    input_channels, channels, kernel_size,
                    dilation=dilation, padding=0,
                ),
                nn.SiLU(),
            ))
            input_channels = channels
        self.temporal = nn.Sequential(*layers)
        self.output = nn.Conv1d(channels, state_dim, kernel_size=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def encode(
        self,
        features: torch.Tensor,
        cache: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        keep = self.receptive_field - 1
        if cache is None:
            cache = features.new_zeros(
                features.shape[0], keep, self.feature_dim
            )
        if cache.shape != (features.shape[0], keep, self.feature_dim):
            raise ValueError(
                "adapter cache must have shape "
                f"{(features.shape[0], keep, self.feature_dim)}, got {tuple(cache.shape)}"
            )
        combined = torch.cat((cache, features), dim=1)
        temporal = self.temporal(combined.transpose(1, 2))
        next_cache = combined[:, -keep:] if keep else combined[:, :0]
        return temporal, next_cache

    def forward(
        self,
        features: torch.Tensor,
        cache: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        temporal, next_cache = self.encode(features, cache)
        residual = self.output(temporal)
        return residual.transpose(1, 2), next_cache


class InputOnlyResidualTCN(nn.Module):
    """Frozen input-only GRU plus a trainable causal TCN state residual.

    The TCN receives the frozen GRU sequence and the same packed spike inputs;
    it never receives a teacher state or a previous predicted physical state.
    """

    architecture = "frozen_gru_causal_residual_tcn_input_only"

    def __init__(
        self,
        baseline: InputOnlyGRU,
        input_dim: int,
        state_dim: int,
        channels: int = 96,
        dilations: tuple[int, ...] = (1, 2, 4, 8),
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        self.baseline = baseline
        for parameter in self.baseline.parameters():
            parameter.requires_grad_(False)
        self.adapter = CausalResidualTCNAdapter(
            baseline.hidden_dim + input_dim,
            state_dim,
            channels=channels,
            dilations=dilations,
            kernel_size=kernel_size,
        )
        self.hidden_dim = baseline.hidden_dim
        self.receptive_field = self.adapter.receptive_field

    def forward(self, inputs: torch.Tensor, hidden: Any = None, timespans: torch.Tensor | None = None):
        del timespans
        recurrent_hidden, adapter_cache = (None, None) if hidden is None else hidden
        encoded = self.baseline.input_encoder(inputs)
        sequence, recurrent_hidden = self.baseline.recurrent(encoded, recurrent_hidden)
        baseline_prediction = self.baseline.decoder(sequence)
        residual, adapter_cache = self.adapter(
            torch.cat((sequence, inputs), dim=-1), adapter_cache
        )
        return baseline_prediction + residual, (recurrent_hidden, adapter_cache)

    def decode_hidden(self, hidden: Any) -> torch.Tensor:
        return self.baseline.decode_hidden(hidden[0])

    @staticmethod
    def detach_hidden(hidden: Any) -> Any:
        if hidden is None:
            return None
        recurrent_hidden, adapter_cache = hidden
        return recurrent_hidden.detach(), adapter_cache.detach()


class InputOnlyGatedResidualTCN(InputOnlyResidualTCN):
    """Frozen GRU plus a causally gated fast residual expert.

    This is a two-expert gated residual/Mixture-of-Experts decomposition: the
    frozen GRU is the always-on slow expert, while a sigmoid gate sparsely
    activates the causal TCN correction.  Both gate and correction consume
    only GRU latent features and packed spike inputs.
    """

    architecture = "frozen_gru_causal_gated_residual_tcn_input_only"

    def __init__(
        self,
        baseline: InputOnlyGRU,
        input_dim: int,
        state_dim: int,
        channels: int = 96,
        dilations: tuple[int, ...] = (1, 2, 4, 8),
        kernel_size: int = 3,
        initial_gate_probability: float = 0.01,
    ) -> None:
        if not 0.0 < initial_gate_probability < 1.0:
            raise ValueError("initial_gate_probability must be in (0,1)")
        super().__init__(
            baseline,
            input_dim,
            state_dim,
            channels=channels,
            dilations=dilations,
            kernel_size=kernel_size,
        )
        self.gate = nn.Conv1d(channels, 1, kernel_size=1)
        nn.init.normal_(self.gate.weight, mean=0.0, std=1e-3)
        nn.init.constant_(
            self.gate.bias,
            math.log(initial_gate_probability / (1.0 - initial_gate_probability)),
        )

    def forward_with_gate(self, inputs: torch.Tensor, hidden: Any = None):
        recurrent_hidden, adapter_cache = (None, None) if hidden is None else hidden
        encoded = self.baseline.input_encoder(inputs)
        sequence, recurrent_hidden = self.baseline.recurrent(encoded, recurrent_hidden)
        baseline_prediction = self.baseline.decoder(sequence)
        temporal, adapter_cache = self.adapter.encode(
            torch.cat((sequence, inputs), dim=-1), adapter_cache
        )
        residual = self.adapter.output(temporal).transpose(1, 2)
        gate_logits = self.gate(temporal).transpose(1, 2)
        prediction = baseline_prediction + torch.sigmoid(gate_logits) * residual
        return prediction, (recurrent_hidden, adapter_cache), gate_logits

    def forward(self, inputs: torch.Tensor, hidden: Any = None, timespans: torch.Tensor | None = None):
        del timespans
        prediction, hidden, _ = self.forward_with_gate(inputs, hidden)
        return prediction, hidden


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
