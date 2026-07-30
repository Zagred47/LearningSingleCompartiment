import importlib.util

import pytest
import torch

from hay_single_compartment import InputOnlyCfC, InputOnlyGRU


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
