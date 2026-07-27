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
        head_dim: int | None = None,
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
        head_width = head_dim or width
        self.head = nn.Sequential(
            nn.Linear(width, head_width), nn.SiLU(), nn.Linear(head_width, state_dim)
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


class ConvLSTMReceptorGRUSurrogate(ConvLSTMSurrogate):
    """ConvLSTM backbone with standard GRU experts for synaptic states.

    The known global backbone still predicts voltage, calcium, and all channel
    gates.  Only the three receptor conductances are delegated to independent
    GRUs that read their own conductance and matching event count.
    """

    receptor_specs = {
        "ampa": (14, 18),
        "nmda": (15, 19),
        "gabaa": (16, 20),
    }

    def __init__(
        self,
        input_dim: int,
        state_dim: int,
        hidden_dim: int = 128,
        layers: int = 2,
        dropout: float = 0.1,
        width_multiplier: int = 2,
        receptor_hidden_dim: int = 32,
        receptor_layers: int = 1,
        global_head_dim: int | None = None,
        **kwargs: object,
    ) -> None:
        if state_dim != 17 or input_dim != 21:
            raise ValueError("conv_lstm_receptor_gru requires the 17-state, 4-input schema")
        super().__init__(
            input_dim=input_dim,
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            layers=layers,
            dropout=dropout,
            width_multiplier=width_multiplier,
            **kwargs,
        )
        width = hidden_dim * width_multiplier
        global_head_width = global_head_dim or width
        self.architecture = "conv_lstm_receptor_gru"
        # The shared backbone owns only V, calcium, and the 12 channel gates.
        self.head = nn.Sequential(
            nn.Linear(width, global_head_width),
            nn.SiLU(),
            nn.Linear(global_head_width, 14),
        )
        self.receptor_encoders = nn.ModuleDict()
        self.receptor_recurrents = nn.ModuleDict()
        self.receptor_heads = nn.ModuleDict()
        for name in self.receptor_specs:
            self.receptor_encoders[name] = nn.Sequential(
                nn.Linear(2, receptor_hidden_dim), nn.SiLU()
            )
            self.receptor_recurrents[name] = nn.GRU(
                receptor_hidden_dim,
                receptor_hidden_dim,
                num_layers=receptor_layers,
                batch_first=True,
                dropout=dropout if receptor_layers > 1 else 0.0,
            )
            self.receptor_heads[name] = nn.Sequential(
                nn.Linear(receptor_hidden_dim, receptor_hidden_dim),
                nn.SiLU(),
                nn.Linear(receptor_hidden_dim, 1),
            )

    def forward(self, features, hidden=None, return_hidden: bool = False):
        hidden = hidden if isinstance(hidden, dict) else {}
        history = hidden.get("conv_history")
        conv_features = torch.cat([history, features], dim=1) if history is not None else features
        encoded = self._encode(conv_features)[:, -features.shape[1] :]
        memory, recurrent_hidden = self.recurrent(encoded, hidden.get("recurrent"))

        prediction = features[..., : self.state_dim].clone()
        prediction[..., :14] = features[..., :14] + self.head(memory)
        next_hidden = {
            "recurrent": recurrent_hidden,
            "conv_history": conv_features[:, -self.context_steps :].detach(),
        }
        for name, (state_index, event_index) in self.receptor_specs.items():
            local_features = features[..., [state_index, event_index]]
            local_encoded = self.receptor_encoders[name](local_features)
            local_memory, local_hidden = self.receptor_recurrents[name](
                local_encoded, hidden.get(f"receptor_{name}")
            )
            prediction[..., state_index] = (
                features[..., state_index]
                + self.receptor_heads[name](local_memory).squeeze(-1)
            )
            next_hidden[f"receptor_{name}"] = local_hidden
        return (prediction, next_hidden) if return_hidden else prediction


class ConvLSTMReceptorHCNGRUSurrogate(ConvLSTMReceptorGRUSurrogate):
    """Previous winning composite plus one local GRU for state coordinate 8."""

    global_state_indices = tuple(index for index in range(14) if index != 8)
    hcn_feature_indices = (0, 8, 17, 18, 19, 20)

    def __init__(
        self,
        input_dim: int,
        state_dim: int,
        hidden_dim: int = 128,
        layers: int = 2,
        dropout: float = 0.1,
        width_multiplier: int = 2,
        receptor_hidden_dim: int = 32,
        receptor_layers: int = 1,
        hcn_hidden_dim: int = 32,
        hcn_layers: int = 1,
        global_head_dim: int | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            layers=layers,
            dropout=dropout,
            width_multiplier=width_multiplier,
            receptor_hidden_dim=receptor_hidden_dim,
            receptor_layers=receptor_layers,
            **kwargs,
        )
        width = hidden_dim * width_multiplier
        global_head_width = global_head_dim or width
        self.architecture = "conv_lstm_receptor_hcn_gru"
        self.head = nn.Sequential(
            nn.Linear(width, global_head_width),
            nn.SiLU(),
            nn.Linear(global_head_width, 13),
        )
        self.hcn_encoder = nn.Sequential(
            nn.Linear(len(self.hcn_feature_indices), hcn_hidden_dim), nn.SiLU()
        )
        self.hcn_recurrent = nn.GRU(
            hcn_hidden_dim,
            hcn_hidden_dim,
            num_layers=hcn_layers,
            batch_first=True,
            dropout=dropout if hcn_layers > 1 else 0.0,
        )
        self.hcn_head = nn.Sequential(
            nn.Linear(hcn_hidden_dim, hcn_hidden_dim),
            nn.SiLU(),
            nn.Linear(hcn_hidden_dim, 1),
        )

    def forward(self, features, hidden=None, return_hidden: bool = False):
        hidden = hidden if isinstance(hidden, dict) else {}
        history = hidden.get("conv_history")
        conv_features = (
            torch.cat([history, features], dim=1) if history is not None else features
        )
        encoded = self._encode(conv_features)[:, -features.shape[1] :]
        memory, recurrent_hidden = self.recurrent(encoded, hidden.get("recurrent"))

        prediction = features[..., : self.state_dim].clone()
        global_indices = list(self.global_state_indices)
        prediction[..., global_indices] = (
            features[..., global_indices] + self.head(memory)
        )
        next_hidden = {
            "recurrent": recurrent_hidden,
            "conv_history": conv_features[:, -self.context_steps :].detach(),
        }
        for name, (state_index, event_index) in self.receptor_specs.items():
            local_features = features[..., [state_index, event_index]]
            local_encoded = self.receptor_encoders[name](local_features)
            local_memory, local_hidden = self.receptor_recurrents[name](
                local_encoded, hidden.get(f"receptor_{name}")
            )
            prediction[..., state_index] = (
                features[..., state_index]
                + self.receptor_heads[name](local_memory).squeeze(-1)
            )
            next_hidden[f"receptor_{name}"] = local_hidden

        hcn_features = features[..., list(self.hcn_feature_indices)]
        hcn_encoded = self.hcn_encoder(hcn_features)
        hcn_memory, hcn_hidden = self.hcn_recurrent(
            hcn_encoded, hidden.get("hcn")
        )
        prediction[..., 8] = (
            features[..., 8] + self.hcn_head(hcn_memory).squeeze(-1)
        )
        next_hidden["hcn"] = hcn_hidden
        return (prediction, next_hidden) if return_hidden else prediction


class ConvLSTMReceptorHCNAuxSurrogate(ConvLSTMReceptorHCNGRUSurrogate):
    """Local coordinate-8 GRU with a standard global auxiliary prediction head."""

    def __init__(
        self,
        input_dim: int,
        state_dim: int,
        hidden_dim: int = 128,
        layers: int = 2,
        dropout: float = 0.1,
        width_multiplier: int = 2,
        auxiliary_hidden_dim: int = 32,
        **kwargs: object,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            layers=layers,
            dropout=dropout,
            width_multiplier=width_multiplier,
            **kwargs,
        )
        width = hidden_dim * width_multiplier
        self.architecture = "conv_lstm_receptor_hcn_aux"
        self.auxiliary_head = nn.Sequential(
            nn.Linear(width, auxiliary_hidden_dim),
            nn.SiLU(),
            nn.Linear(auxiliary_hidden_dim, 1),
        )

    def _forward_all(self, features, hidden=None):
        hidden = hidden if isinstance(hidden, dict) else {}
        history = hidden.get("conv_history")
        conv_features = (
            torch.cat([history, features], dim=1) if history is not None else features
        )
        encoded = self._encode(conv_features)[:, -features.shape[1] :]
        memory, recurrent_hidden = self.recurrent(encoded, hidden.get("recurrent"))

        prediction = features[..., : self.state_dim].clone()
        global_indices = list(self.global_state_indices)
        prediction[..., global_indices] = (
            features[..., global_indices] + self.head(memory)
        )
        next_hidden = {
            "recurrent": recurrent_hidden,
            "conv_history": conv_features[:, -self.context_steps :].detach(),
        }
        for name, (state_index, event_index) in self.receptor_specs.items():
            local_features = features[..., [state_index, event_index]]
            local_encoded = self.receptor_encoders[name](local_features)
            local_memory, local_hidden = self.receptor_recurrents[name](
                local_encoded, hidden.get(f"receptor_{name}")
            )
            prediction[..., state_index] = (
                features[..., state_index]
                + self.receptor_heads[name](local_memory).squeeze(-1)
            )
            next_hidden[f"receptor_{name}"] = local_hidden

        hcn_features = features[..., list(self.hcn_feature_indices)]
        hcn_encoded = self.hcn_encoder(hcn_features)
        hcn_memory, hcn_hidden = self.hcn_recurrent(
            hcn_encoded, hidden.get("hcn")
        )
        prediction[..., 8] = (
            features[..., 8] + self.hcn_head(hcn_memory).squeeze(-1)
        )
        next_hidden["hcn"] = hcn_hidden
        auxiliary_prediction = features[..., 8] + self.auxiliary_head(memory).squeeze(-1)
        return prediction, next_hidden, auxiliary_prediction

    def forward(self, features, hidden=None, return_hidden: bool = False):
        prediction, next_hidden, _ = self._forward_all(features, hidden)
        return (prediction, next_hidden) if return_hidden else prediction

    def forward_with_auxiliary(self, features):
        prediction, _, auxiliary_prediction = self._forward_all(features)
        return prediction, auxiliary_prediction


class ConvLSTMReceptorHCNMLPAuxSurrogate(ConvLSTMReceptorGRUSurrogate):
    """Markovian local coordinate-8 MLP plus global auxiliary supervision."""

    global_state_indices = tuple(index for index in range(14) if index != 8)
    hcn_feature_indices = (0, 8, 17, 18, 19, 20)

    def __init__(
        self,
        input_dim: int,
        state_dim: int,
        hidden_dim: int = 128,
        layers: int = 2,
        dropout: float = 0.1,
        width_multiplier: int = 2,
        hcn_mlp_hidden_dim: int = 82,
        auxiliary_hidden_dim: int = 32,
        **kwargs: object,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            layers=layers,
            dropout=dropout,
            width_multiplier=width_multiplier,
            **kwargs,
        )
        width = hidden_dim * width_multiplier
        self.architecture = "conv_lstm_receptor_hcn_mlp_aux"
        self.head = nn.Sequential(
            nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 13)
        )
        self.hcn_mlp = nn.Sequential(
            nn.Linear(len(self.hcn_feature_indices), hcn_mlp_hidden_dim),
            nn.SiLU(),
            nn.Linear(hcn_mlp_hidden_dim, hcn_mlp_hidden_dim),
            nn.SiLU(),
            nn.Linear(hcn_mlp_hidden_dim, 1),
        )
        self.auxiliary_head = nn.Sequential(
            nn.Linear(width, auxiliary_hidden_dim),
            nn.SiLU(),
            nn.Linear(auxiliary_hidden_dim, 1),
        )

    def _forward_all(self, features, hidden=None):
        hidden = hidden if isinstance(hidden, dict) else {}
        history = hidden.get("conv_history")
        conv_features = (
            torch.cat([history, features], dim=1) if history is not None else features
        )
        encoded = self._encode(conv_features)[:, -features.shape[1] :]
        memory, recurrent_hidden = self.recurrent(encoded, hidden.get("recurrent"))

        prediction = features[..., : self.state_dim].clone()
        global_indices = list(self.global_state_indices)
        prediction[..., global_indices] = (
            features[..., global_indices] + self.head(memory)
        )
        next_hidden = {
            "recurrent": recurrent_hidden,
            "conv_history": conv_features[:, -self.context_steps :].detach(),
        }
        for name, (state_index, event_index) in self.receptor_specs.items():
            local_features = features[..., [state_index, event_index]]
            local_encoded = self.receptor_encoders[name](local_features)
            local_memory, local_hidden = self.receptor_recurrents[name](
                local_encoded, hidden.get(f"receptor_{name}")
            )
            prediction[..., state_index] = (
                features[..., state_index]
                + self.receptor_heads[name](local_memory).squeeze(-1)
            )
            next_hidden[f"receptor_{name}"] = local_hidden

        hcn_features = features[..., list(self.hcn_feature_indices)]
        prediction[..., 8] = (
            features[..., 8] + self.hcn_mlp(hcn_features).squeeze(-1)
        )
        auxiliary_prediction = features[..., 8] + self.auxiliary_head(memory).squeeze(-1)
        return prediction, next_hidden, auxiliary_prediction

    def forward(self, features, hidden=None, return_hidden: bool = False):
        prediction, next_hidden, _ = self._forward_all(features, hidden)
        return (prediction, next_hidden) if return_hidden else prediction

    def forward_with_auxiliary(self, features):
        prediction, _, auxiliary_prediction = self._forward_all(features)
        return prediction, auxiliary_prediction


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
    if architecture in {
        "conv_lstm_receptor_gru",
        "convlstm_receptor_gru",
        "conv_receptor_gru",
    }:
        return ConvLSTMReceptorGRUSurrogate(input_dim, state_dim, **kwargs)
    if architecture in {
        "conv_lstm_receptor_hcn_gru",
        "convlstm_receptor_hcn_gru",
        "conv_receptor_hcn_gru",
    }:
        return ConvLSTMReceptorHCNGRUSurrogate(input_dim, state_dim, **kwargs)
    if architecture in {
        "conv_lstm_receptor_hcn_aux",
        "convlstm_receptor_hcn_aux",
        "conv_receptor_hcn_aux",
    }:
        return ConvLSTMReceptorHCNAuxSurrogate(input_dim, state_dim, **kwargs)
    if architecture in {
        "conv_lstm_receptor_hcn_mlp_aux",
        "convlstm_receptor_hcn_mlp_aux",
        "conv_receptor_hcn_mlp_aux",
    }:
        return ConvLSTMReceptorHCNMLPAuxSurrogate(input_dim, state_dim, **kwargs)
    if architecture in {"ontology_gru", "causal_gru", "gru_mosaic"}:
        return OntologyGRUMosaic(input_dim, state_dim, **kwargs)
    return RecurrentSurrogate(input_dim, state_dim, architecture=architecture, **kwargs)
