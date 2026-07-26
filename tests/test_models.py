import pytest
import torch

from hay_single_compartment.models import build_model


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
