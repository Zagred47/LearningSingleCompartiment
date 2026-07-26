import pytest
import torch

from hay_single_compartment.models import build_model


@pytest.mark.parametrize("architecture", ["mlp", "rnn", "gru", "lstm"])
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
