import pytest
import torch
import numpy as np

from hay_single_compartment.models import build_model
from hay_single_compartment.dataset import Normalization
from hay_single_compartment.training import rollout_batch, rollout_trajectory
from hay_single_compartment.ontology import ONTOLOGY_GROUPS


@pytest.mark.parametrize(
    "architecture",
    [
        "mlp", "rnn", "gru", "lstm", "conv_lstm", "ontology_gru",
        "conv_lstm_receptor_gru",
        "conv_lstm_receptor_hcn_gru",
    ],
)
def test_model_shape_and_gradient(architecture):
    model = build_model(architecture, input_dim=21, state_dim=17, hidden_dim=16, layers=1)
    features = torch.randn(3, 8, 21)
    prediction = model(features)
    assert prediction.shape == (3, 8, 17)
    prediction.square().mean().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_unknown_architecture_is_rejected():
    with pytest.raises(ValueError):
        build_model("transformer", input_dim=21, state_dim=17)


def test_ontology_covers_every_state_once():
    output_indices = [index for group in ONTOLOGY_GROUPS for index in group.output_indices]
    assert sorted(output_indices) == list(range(17))
    ampa = next(group for group in ONTOLOGY_GROUPS if group.name == "ampa_receptor")
    assert ampa.dependency_names == ("g_AMPA_uS", "ampa_event_count")


def test_scaled_conv_lstm_has_more_capacity():
    base = build_model("conv_lstm", 21, 17, hidden_dim=16, layers=2)
    scaled = build_model(
        "conv_lstm", 21, 17, hidden_dim=24, layers=3, width_multiplier=3
    )
    assert sum(p.numel() for p in scaled.parameters()) > 4 * sum(
        p.numel() for p in base.parameters()
    )


def test_conv_lstm_stepwise_matches_causal_full_sequence():
    torch.manual_seed(4)
    model = build_model("conv_lstm", input_dim=21, state_dim=17, hidden_dim=8, layers=1)
    model.eval()
    features = torch.randn(2, 12, 21)
    full = model(features)
    hidden = None
    stepwise = []
    for step in range(features.shape[1]):
        prediction, hidden = model(
            features[:, step : step + 1], hidden=hidden, return_hidden=True
        )
        stepwise.append(prediction)
    torch.testing.assert_close(torch.cat(stepwise, dim=1), full, atol=1e-5, rtol=1e-5)


def test_ontology_gru_stepwise_matches_full_sequence():
    torch.manual_seed(5)
    model = build_model("ontology_gru", input_dim=21, state_dim=17, hidden_dim=8, layers=1)
    model.eval()
    features = torch.randn(2, 7, 21)
    full = model(features)
    hidden = None
    stepwise = []
    for step in range(features.shape[1]):
        prediction, hidden = model(
            features[:, step : step + 1], hidden=hidden, return_hidden=True
        )
        stepwise.append(prediction)
    torch.testing.assert_close(torch.cat(stepwise, dim=1), full, atol=1e-5, rtol=1e-5)


def test_receptor_composite_stepwise_matches_full_sequence():
    torch.manual_seed(6)
    model = build_model(
        "conv_lstm_receptor_gru",
        input_dim=21,
        state_dim=17,
        hidden_dim=8,
        layers=1,
        receptor_hidden_dim=5,
    )
    model.eval()
    features = torch.randn(2, 9, 21)
    full = model(features)
    hidden = None
    stepwise = []
    for step in range(features.shape[1]):
        prediction, hidden = model(
            features[:, step : step + 1], hidden=hidden, return_hidden=True
        )
        stepwise.append(prediction)
    torch.testing.assert_close(torch.cat(stepwise, dim=1), full, atol=1e-5, rtol=1e-5)


def test_receptor_composite_has_near_matched_capacity_control():
    composite = build_model(
        "conv_lstm_receptor_gru", 21, 17, hidden_dim=128, layers=3,
        width_multiplier=2, receptor_hidden_dim=32,
    )
    control = build_model(
        "conv_lstm", 21, 17, hidden_dim=128, layers=3,
        width_multiplier=2, head_dim=336,
    )
    composite_parameters = sum(parameter.numel() for parameter in composite.parameters())
    control_parameters = sum(parameter.numel() for parameter in control.parameters())
    assert abs(composite_parameters - control_parameters) / composite_parameters < 0.002


def test_hcn_composite_stepwise_matches_full_sequence():
    torch.manual_seed(7)
    model = build_model(
        "conv_lstm_receptor_hcn_gru",
        21,
        17,
        hidden_dim=8,
        layers=1,
        receptor_hidden_dim=5,
        hcn_hidden_dim=6,
    )
    model.eval()
    features = torch.randn(2, 9, 21)
    full = model(features)
    hidden = None
    stepwise = []
    for step in range(features.shape[1]):
        prediction, hidden = model(
            features[:, step : step + 1], hidden=hidden, return_hidden=True
        )
        stepwise.append(prediction)
    torch.testing.assert_close(torch.cat(stepwise, dim=1), full, atol=1e-5, rtol=1e-5)


def test_hcn_composite_has_near_matched_composite_control():
    hcn_composite = build_model(
        "conv_lstm_receptor_hcn_gru", 21, 17, hidden_dim=128, layers=3,
        width_multiplier=2, receptor_hidden_dim=32, hcn_hidden_dim=32,
    )
    control = build_model(
        "conv_lstm_receptor_gru", 21, 17, hidden_dim=128, layers=3,
        width_multiplier=2, receptor_hidden_dim=32, global_head_dim=283,
    )
    candidate_parameters = sum(parameter.numel() for parameter in hcn_composite.parameters())
    control_parameters = sum(parameter.numel() for parameter in control.parameters())
    assert abs(candidate_parameters - control_parameters) / candidate_parameters < 0.001


def test_hcn_auxiliary_head_trains_and_is_inference_transparent():
    torch.manual_seed(9)
    model = build_model(
        "conv_lstm_receptor_hcn_aux", 21, 17, hidden_dim=8, layers=1,
        receptor_hidden_dim=5, hcn_hidden_dim=6, auxiliary_hidden_dim=7,
    )
    features = torch.randn(2, 6, 21)
    prediction, auxiliary = model.forward_with_auxiliary(features)
    torch.testing.assert_close(model(features), prediction)
    assert auxiliary.shape == (2, 6)
    (prediction.square().mean() + auxiliary.square().mean()).backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_hcn_auxiliary_has_near_matched_capacity_control():
    candidate = build_model(
        "conv_lstm_receptor_hcn_aux", 21, 17, hidden_dim=128, layers=3,
        width_multiplier=2, receptor_hidden_dim=32, hcn_hidden_dim=32,
        auxiliary_hidden_dim=32,
    )
    control = build_model(
        "conv_lstm_receptor_hcn_gru", 21, 17, hidden_dim=128, layers=3,
        width_multiplier=2, receptor_hidden_dim=32, hcn_hidden_dim=32,
        global_head_dim=287,
    )
    candidate_parameters = sum(parameter.numel() for parameter in candidate.parameters())
    control_parameters = sum(parameter.numel() for parameter in control.parameters())
    assert abs(candidate_parameters - control_parameters) / candidate_parameters < 0.001


def test_hcn_mlp_auxiliary_is_stepwise_and_trains_all_parameters():
    torch.manual_seed(10)
    model = build_model(
        "conv_lstm_receptor_hcn_mlp_aux", 21, 17, hidden_dim=8, layers=1,
        receptor_hidden_dim=5, hcn_mlp_hidden_dim=9, auxiliary_hidden_dim=7,
    )
    model.eval()
    features = torch.randn(2, 8, 21)
    full, auxiliary = model.forward_with_auxiliary(features)
    hidden = None
    stepwise = []
    for step in range(features.shape[1]):
        prediction, hidden = model(
            features[:, step : step + 1], hidden=hidden, return_hidden=True
        )
        stepwise.append(prediction)
    torch.testing.assert_close(torch.cat(stepwise, dim=1), full, atol=1e-5, rtol=1e-5)
    (full.square().mean() + auxiliary.square().mean()).backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_hcn_mlp_and_gru_auxiliary_have_matched_capacity():
    common = dict(
        hidden_dim=128, layers=3, width_multiplier=2,
        receptor_hidden_dim=32, auxiliary_hidden_dim=32,
    )
    gru = build_model(
        "conv_lstm_receptor_hcn_aux", 21, 17,
        hcn_hidden_dim=32, **common,
    )
    mlp = build_model(
        "conv_lstm_receptor_hcn_mlp_aux", 21, 17,
        hcn_mlp_hidden_dim=82, **common,
    )
    gru_parameters = sum(parameter.numel() for parameter in gru.parameters())
    mlp_parameters = sum(parameter.numel() for parameter in mlp.parameters())
    assert abs(gru_parameters - mlp_parameters) / gru_parameters < 0.001


def test_batched_rollout_matches_individual_rollouts(capsys):
    torch.manual_seed(8)
    model = build_model("conv_lstm", 21, 17, hidden_dim=8, layers=1)
    normalization = Normalization(
        state_mean=np.zeros(17), state_std=np.ones(17),
        input_mean=np.zeros(4), input_std=np.ones(4),
    )
    initial = np.zeros((2, 17), dtype=np.float32)
    inputs = np.random.default_rng(8).normal(size=(2, 6, 4)).astype(np.float32)
    batched = rollout_batch(model, initial, inputs, normalization, progress=True)
    assert "[rollout]" in capsys.readouterr().out
    individual = np.stack([
        rollout_trajectory(model, initial[index], inputs[index], normalization)
        for index in range(2)
    ])
    np.testing.assert_allclose(batched, individual, atol=1e-5, rtol=1e-5)
