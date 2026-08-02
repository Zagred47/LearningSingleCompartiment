import math

import torch
from torch import nn

from hay_single_compartment.loss_landscape import (
    cosine,
    filter_normalized_direction,
    gradient_snr,
    hutchinson_trace,
    parameter_snapshot,
    rank_correlation,
    set_parameter_point,
    top_hessian_eigenvalue,
)


def test_cosine_and_gradient_snr() -> None:
    left = torch.tensor([1.0, 0.0])
    assert cosine(left, left) == 1.0
    assert cosine(left, -left) == -1.0
    assert gradient_snr([left, left]) > 1e10


def test_filter_direction_and_parameter_point() -> None:
    model = nn.Linear(3, 2)
    base = parameter_snapshot(model)
    direction = filter_normalized_direction(model, generator=torch.Generator().manual_seed(7))
    weight_norm = torch.linalg.vector_norm(model.weight.detach().reshape(2, -1), dim=1)
    direction_norm = torch.linalg.vector_norm(direction["weight"].reshape(2, -1), dim=1)
    torch.testing.assert_close(direction_norm, weight_norm)
    set_parameter_point(model, base, [(0.25, direction)])
    torch.testing.assert_close(model.weight, base["weight"] + 0.25 * direction["weight"])


def test_quadratic_hessian_estimators() -> None:
    parameter = nn.Parameter(torch.tensor([1.0, -2.0]))

    def closure() -> torch.Tensor:
        return 0.5 * torch.sum(torch.tensor([2.0, 4.0]) * parameter.square())

    eigenvalue, residual, _ = top_hessian_eigenvalue(
        closure, [parameter], iterations=20, generator=torch.Generator().manual_seed(3)
    )
    trace, standard_error, _ = hutchinson_trace(
        closure, [parameter], probes=4, generator=torch.Generator().manual_seed(4)
    )
    assert math.isclose(eigenvalue, 4.0, rel_tol=1e-3)
    assert residual < 1e-2
    assert math.isclose(trace, 6.0, rel_tol=1e-6)
    assert standard_error == 0.0


def test_rank_correlation() -> None:
    assert math.isclose(rank_correlation([1, 2, 3], [4, 5, 6]), 1.0)
    assert math.isclose(rank_correlation([1, 2, 3], [6, 5, 4]), -1.0)
