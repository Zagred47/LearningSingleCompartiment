import pytest
import torch
import numpy as np

from hay_single_compartment.models import build_model
from hay_single_compartment.dataset import Normalization
from hay_single_compartment.training import rollout_batch, rollout_trajectory


@pytest.mark.parametrize("architecture", ["mlp", "rnn", "gru", "lstm", "conv_lstm"])
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
