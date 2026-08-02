import importlib.util

import numpy as np
import pytest
import torch

from hay_single_compartment import (
    AuxiliarySpikePhaseLoss,
    MICRO_STATE_NAMES,
    ConservativeSpikeFineTuneLoss,
    EventAwareStateLoss,
    WaveformConstrainedFineTuneLoss,
    InputOnlyBranchELM,
    InputOnlyCfC,
    InputOnlyConvGRU,
    InputOnlyConvLSTM,
    InputOnlyGRU,
    InputOnlyGatedResidualTCN,
    InputOnlyResidualTCN,
    StateContextGRU,
    MicroEventConfig,
    StratifiedWindowSampler,
    SpikeGateFocalLoss,
    classify_micro_events,
    replay_gru_gates,
)


def test_state_context_gru_is_parameter_matched_and_chunk_equivalent():
    torch.manual_seed(11)
    reference = StateContextGRU(6, 4, hidden_dim=9, decoder_dim=7, mode="none")
    models = {}
    for mode in StateContextGRU.valid_modes:
        model = StateContextGRU(6, 4, hidden_dim=9, decoder_dim=7, mode=mode)
        model.load_state_dict(reference.state_dict())
        models[mode] = model
    assert len({sum(p.numel() for p in model.parameters()) for model in models.values()}) == 1

    inputs = torch.randn(2, 8, 6)
    initial = torch.randn(2, 4)
    for model in models.values():
        full, full_hidden = model(inputs, initial_state=initial)
        first, hidden = model(inputs[:, :3], initial_state=initial)
        second, chunk_hidden = model(inputs[:, 3:], hidden)
        torch.testing.assert_close(torch.cat((first, second), dim=1), full)
        torch.testing.assert_close(chunk_hidden[0], full_hidden[0])
        torch.testing.assert_close(chunk_hidden[1], full_hidden[1])

    initial_output, _ = models["initial_only"](inputs, initial_state=initial)
    feedback_output, _ = models["predicted_feedback"](inputs, initial_state=initial)
    torch.testing.assert_close(initial_output[:, :1], feedback_output[:, :1])
    assert not torch.allclose(initial_output[:, 1:], feedback_output[:, 1:])


def test_state_context_gru_rejects_teacher_reinjection_after_initialization():
    model = StateContextGRU(6, 4, hidden_dim=5, mode="predicted_feedback")
    inputs = torch.randn(2, 3, 6)
    initial = torch.randn(2, 4)
    _, hidden = model(inputs[:, :1], initial_state=initial)
    with pytest.raises(ValueError, match="only be supplied"):
        model(inputs[:, 1:], hidden, initial_state=initial)


def test_residual_tcn_starts_as_exact_frozen_gru_and_is_chunk_equivalent():
    torch.manual_seed(2)
    baseline = InputOnlyGRU(input_dim=12, state_dim=61, hidden_dim=10)
    model = InputOnlyResidualTCN(
        baseline, input_dim=12, state_dim=61, channels=8, dilations=(1, 2)
    )
    inputs = torch.randn(2, 13, 12)
    expected, _ = baseline(inputs)
    full, _ = model(inputs)
    torch.testing.assert_close(full, expected)
    assert all(not parameter.requires_grad for parameter in model.baseline.parameters())
    assert any(parameter.requires_grad for parameter in model.adapter.parameters())

    with torch.no_grad():
        model.adapter.output.weight.normal_(std=0.05)
    full, full_hidden = model(inputs)
    first, hidden = model(inputs[:, :5])
    second, chunk_hidden = model(inputs[:, 5:], hidden)
    torch.testing.assert_close(
        torch.cat((first, second), dim=1), full, atol=2e-6, rtol=2e-6
    )
    torch.testing.assert_close(
        model.decode_hidden(chunk_hidden), model.decode_hidden(full_hidden)
    )


def test_residual_tcn_is_strictly_causal():
    torch.manual_seed(7)
    model = InputOnlyResidualTCN(
        InputOnlyGRU(6, 4, hidden_dim=5),
        input_dim=6,
        state_dim=4,
        channels=7,
        dilations=(1, 2),
    )
    with torch.no_grad():
        model.adapter.output.weight.normal_(std=0.1)
    inputs = torch.randn(1, 12, 6)
    changed_future = inputs.clone()
    changed_future[:, 7:] = torch.randn_like(changed_future[:, 7:]) * 10
    original, _ = model(inputs)
    changed, _ = model(changed_future)
    torch.testing.assert_close(original[:, :7], changed[:, :7], atol=2e-6, rtol=2e-6)


def test_gated_residual_tcn_starts_at_baseline_and_streams_exactly():
    torch.manual_seed(8)
    baseline = InputOnlyGRU(6, 4, hidden_dim=5)
    model = InputOnlyGatedResidualTCN(
        baseline, input_dim=6, state_dim=4, channels=7, dilations=(1, 2)
    )
    inputs = torch.randn(2, 12, 6)
    expected, _ = baseline(inputs)
    initial, _ = model(inputs)
    torch.testing.assert_close(initial, expected)
    with torch.no_grad():
        model.adapter.output.weight.normal_(std=0.1)
        model.gate.bias.fill_(0.0)
    full, _ = model(inputs)
    first, hidden = model(inputs[:, :5])
    second, _ = model(inputs[:, 5:], hidden)
    torch.testing.assert_close(
        torch.cat((first, second), dim=1), full, atol=2e-6, rtol=2e-6
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


def test_waveform_constrained_loss_is_symmetric_and_anchors_non_events():
    mean = np.zeros(len(MICRO_STATE_NAMES), dtype=np.float32)
    std = np.ones_like(mean)
    soma = MICRO_STATE_NAMES.index("soma.v_mV")
    mean[soma] = -70.0
    criterion = WaveformConstrainedFineTuneLoss(
        MICRO_STATE_NAMES, mean, std, event_radius_steps=2
    )
    target = torch.zeros(1, 20, len(MICRO_STATE_NAMES))
    target[:, 10, soma] = 80.0
    reference = torch.zeros_like(target)
    prediction = reference.clone().requires_grad_(True)

    mse_only, terms = criterion(prediction, target, reference)
    torch.testing.assert_close(mse_only, terms["global"])
    criterion.set_event_scale(1.0)
    total, terms = criterion(prediction, target, reference)
    total.backward()
    assert torch.isfinite(total) and torch.isfinite(prediction.grad).all()
    assert total > terms["global"]
    assert set(terms) == {
        "global", "event_state", "waveform", "derivative", "occupancy",
        "reference", "soma_reference", "event", "event_scale", "event_fraction",
    }

    undershoot = target.clone()
    overshoot = target.clone()
    undershoot[:, 10, soma] -= 5.0
    overshoot[:, 10, soma] += 5.0
    _, under_terms = criterion(undershoot, target, reference)
    _, over_terms = criterion(overshoot, target, reference)
    torch.testing.assert_close(under_terms["waveform"], over_terms["waveform"])

    late_plateau = reference.clone()
    late_plateau[:, 16:, soma] = 80.0
    _, plateau_terms = criterion(late_plateau, target, reference)
    assert plateau_terms["reference"] > 0


def test_auxiliary_spike_phase_loss_supervises_logit_and_derivative():
    mean = np.zeros(len(MICRO_STATE_NAMES), dtype=np.float32)
    std = np.ones_like(mean)
    soma = MICRO_STATE_NAMES.index("soma.v_mV")
    mean[soma] = -70.0
    criterion = AuxiliarySpikePhaseLoss(MICRO_STATE_NAMES, mean, std)
    target = torch.zeros(2, 20, len(MICRO_STATE_NAMES))
    target[:, 10, soma] = 80.0
    auxiliary = torch.zeros(2, 20, 2, requires_grad=True)
    total, terms = criterion(auxiliary, target)
    total.backward()
    assert torch.isfinite(total) and torch.isfinite(auxiliary.grad).all()
    assert set(terms) == {
        "auxiliary_bce", "auxiliary_derivative", "auxiliary_total",
        "auxiliary_positive_weight",
    }
    assert float(terms["auxiliary_positive_weight"]) > 1.0


def test_spike_gate_focal_loss_balances_sparse_support():
    mean = np.zeros(len(MICRO_STATE_NAMES), dtype=np.float32)
    std = np.ones_like(mean)
    soma = MICRO_STATE_NAMES.index("soma.v_mV")
    mean[soma] = -70.0
    criterion = SpikeGateFocalLoss(
        MICRO_STATE_NAMES, mean, std, support_radius_steps=2
    )
    target = torch.zeros(2, 20, len(MICRO_STATE_NAMES))
    target[:, 10, soma] = 80.0
    logits = torch.full((2, 20, 1), -4.0, requires_grad=True)
    total, terms = criterion(logits, target)
    total.backward()
    assert torch.isfinite(total) and torch.isfinite(logits.grad).all()
    assert set(terms) == {
        "gate_focal", "gate_positive_focal", "gate_negative_focal",
        "gate_target_fraction", "gate_predicted_fraction",
        "gate_true_positive_rate", "gate_false_positive_rate",
    }
    torch.testing.assert_close(terms["gate_target_fraction"], torch.tensor(0.25))


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
