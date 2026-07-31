import importlib.util

import numpy as np
import pytest
import torch

from hay_single_compartment import (
    MICRO_STATE_NAMES,
    ConservativeSpikeFineTuneLoss,
    EventAwareStateLoss,
    InputOnlyBranchELM,
    InputOnlyCfC,
    InputOnlyConvGRU,
    InputOnlyConvLSTM,
    InputOnlyGRU,
    MicroEventConfig,
    StratifiedWindowSampler,
    classify_micro_events,
    replay_gru_gates,
)


def test_input_only_gru_carries_hidden_without_teacher_state():
    torch.manual_seed(3)
    model = InputOnlyGRU(input_dim=24, state_dim=61, hidden_dim=16)
    spikes = torch.randint(0, 2, (2, 9, 24)).float()
    full, hidden = model(spikes)
    first, hidden_first = model(spikes[:, :4])
    second, hidden_second = model(spikes[:, 4:], hidden_first)
    torch.testing.assert_close(torch.cat((first, second), dim=1), full)
    torch.testing.assert_close(hidden_second, hidden)
    assert model.decode_hidden(hidden).shape == (2, 61)


def test_gru_gate_replay_matches_fused_gru():
    torch.manual_seed(4)
    model = InputOnlyGRU(input_dim=12, state_dim=61, hidden_dim=9)
    inputs = torch.randn(2, 7, 12)
    encoded = model.input_encoder(inputs)
    fused, _ = model.recurrent(encoded)
    replay = replay_gru_gates(model, inputs)
    torch.testing.assert_close(replay["hidden"], fused, atol=2e-6, rtol=2e-6)
    assert torch.all((replay["update"] >= 0) & (replay["update"] <= 1))


@pytest.mark.parametrize("model_class", [InputOnlyConvGRU, InputOnlyConvLSTM])
def test_causal_conv_recurrent_is_chunk_equivalent(model_class):
    torch.manual_seed(5)
    model = model_class(12, 61, hidden_dim=10, conv_channels=8, dilations=(1, 2))
    inputs = torch.randn(2, 11, 12)
    full, full_hidden = model(inputs)
    first, hidden = model(inputs[:, :4])
    second, chunk_hidden = model(inputs[:, 4:], hidden)
    torch.testing.assert_close(torch.cat((first, second), dim=1), full, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(model.decode_hidden(chunk_hidden), model.decode_hidden(full_hidden))


def test_branch_elm_routes_packed_microbins_and_carries_state():
    torch.manual_seed(6)
    model = InputOnlyBranchELM(24 * 5, 61, num_branch=24, num_memory=12)
    inputs = torch.randint(0, 2, (2, 9, 24 * 5)).float()
    full, full_hidden = model(inputs)
    first, hidden = model(inputs[:, :3])
    second, chunk_hidden = model(inputs[:, 3:], hidden)
    torch.testing.assert_close(torch.cat((first, second), dim=1), full)
    torch.testing.assert_close(model.decode_hidden(chunk_hidden), model.decode_hidden(full_hidden))


def test_event_catalog_sampler_and_loss_cover_rare_spikes():
    states = np.zeros((1, 100, len(MICRO_STATE_NAMES)), dtype=np.float32)
    states[..., MICRO_STATE_NAMES.index("soma.v_mV")] = -70.0
    states[..., MICRO_STATE_NAMES.index("tuft.v_mV")] = -70.0
    spikes = np.zeros((1, 100), dtype=np.uint8)
    for index in (20, 50, 55, 60):
        spikes[0, index] = 1
        states[0, index, MICRO_STATE_NAMES.index("soma.v_mV")] = 10.0
    states[0, 45:75, MICRO_STATE_NAMES.index("tuft.v_mV")] = -25.0
    labels = classify_micro_events(
        states, spikes, MICRO_STATE_NAMES, 1.0,
        MicroEventConfig(minimum_plateau_ms=5.0),
    )
    assert labels[..., 2].any() and labels[..., 3].any()
    assert labels[..., 4].any() and labels[..., 5].any() and labels[..., 6].any()
    sampler = StratifiedWindowSampler(labels, 16, {"burst_spike": 1.0}, seed=2)
    windows = sampler.sample(8)
    assert windows.shape == (8, 2)
    mean = np.zeros(len(MICRO_STATE_NAMES), dtype=np.float32)
    std = np.ones_like(mean)
    loss = EventAwareStateLoss(MICRO_STATE_NAMES, mean, std, event_radius_steps=2)
    prediction = torch.zeros(2, 20, len(MICRO_STATE_NAMES), requires_grad=True)
    target = torch.zeros_like(prediction)
    soma = MICRO_STATE_NAMES.index("soma.v_mV")
    target.data[:, 10, soma] = 5.0
    value, terms = loss(prediction, target)
    value.backward()
    assert torch.isfinite(value) and prediction.grad is not None
    assert set(terms) == {"global", "event_voltage", "derivative", "rapid_gate", "soft_spike"}


def test_conservative_spike_finetune_loss_is_amp_safe_and_curriculum_preserves_mse():
    mean = np.zeros(len(MICRO_STATE_NAMES), dtype=np.float32)
    std = np.ones_like(mean)
    criterion = ConservativeSpikeFineTuneLoss(MICRO_STATE_NAMES, mean, std)
    prediction = torch.randn(2, 24, len(MICRO_STATE_NAMES), requires_grad=True)
    target = torch.randn_like(prediction)
    soma = MICRO_STATE_NAMES.index("soma.v_mV")
    target.data[..., soma] = -70.0
    target.data[:, 12, soma] = 10.0
    with torch.autocast("cpu", dtype=torch.bfloat16):
        mse_only, terms = criterion(prediction, target)
    torch.testing.assert_close(mse_only, terms["global"])
    criterion.set_event_scale(1.0)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        total, terms = criterion(prediction, target)
    total.backward()
    assert torch.isfinite(total) and torch.isfinite(prediction.grad).all()
    assert total > terms["global"]
    assert 1.0 <= float(terms["positive_weight"]) <= 64.0


@pytest.mark.skipif(importlib.util.find_spec("ncps") is None, reason="optional ncps package")
def test_input_only_cfc_shapes_and_timespans():
    model = InputOnlyCfC(
        input_dim=24,
        state_dim=61,
        hidden_dim=12,
        input_embedding_dim=8,
        backbone_units=10,
        backbone_layers=1,
    )
    spikes = torch.zeros(2, 5, 24)
    timespans = torch.full((2, 5), 0.1)
    prediction, hidden = model(spikes, timespans=timespans)
    assert prediction.shape == (2, 5, 61)
    assert model.decode_hidden(hidden).shape == (2, 61)
