"""Small, architecture-agnostic loss-landscape diagnostics.

The functions in this module never update model parameters through an
optimizer.  They expose gradient geometry and Hessian-vector products for
frozen checkpoints and deliberately avoid third-party landscape packages.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

import torch
from torch import nn


TensorTree = torch.Tensor | tuple["TensorTree", ...] | list["TensorTree"] | None


def clone_hidden(hidden: TensorTree) -> TensorTree:
    """Clone and detach a nested recurrent-state tree."""

    if hidden is None:
        return None
    if isinstance(hidden, tuple):
        return tuple(clone_hidden(value) for value in hidden)
    if isinstance(hidden, list):
        return [clone_hidden(value) for value in hidden]
    return hidden.detach().clone()


def concatenate_hidden(states: Sequence[TensorTree], batch_dimension: int = 1) -> TensorTree:
    """Concatenate equal-structure recurrent states into a batch.

    PyTorch recurrent states use batch dimension one, whereas temporal caches
    use batch dimension zero.  Callers with mixed structures should batch each
    window separately; the helper is intended for homogeneous trees.
    """

    if not states:
        raise ValueError("states must not be empty")
    first = states[0]
    if first is None:
        if any(state is not None for state in states):
            raise ValueError("hidden-state structures differ")
        return None
    if isinstance(first, tuple):
        if not all(isinstance(state, tuple) and len(state) == len(first) for state in states):
            raise ValueError("hidden-state structures differ")
        return tuple(
            concatenate_hidden([state[index] for state in states], batch_dimension)
            for index in range(len(first))
        )
    if isinstance(first, list):
        if not all(isinstance(state, list) and len(state) == len(first) for state in states):
            raise ValueError("hidden-state structures differ")
        return [
            concatenate_hidden([state[index] for state in states], batch_dimension)
            for index in range(len(first))
        ]
    if not all(torch.is_tensor(state) for state in states):
        raise ValueError("hidden-state structures differ")
    return torch.cat(states, dim=batch_dimension)


def parameter_block(name: str) -> str:
    """Map a parameter name to an interpretable architecture block."""

    if name.startswith("frontend") or name.startswith("input_encoder"):
        return "input_or_frontend"
    if name.startswith("recurrent"):
        return "recurrent"
    if name.startswith("decoder"):
        return "decoder"
    return name.split(".", 1)[0]


def trainable_named_parameters(model: nn.Module) -> list[tuple[str, nn.Parameter]]:
    return [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]


def flatten_gradients(
    named_parameters: Iterable[tuple[str, nn.Parameter]],
    *,
    fill_none: bool = True,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Flatten current gradients globally and by functional parameter block."""

    complete: list[torch.Tensor] = []
    blocks: dict[str, list[torch.Tensor]] = {}
    for name, parameter in named_parameters:
        gradient = parameter.grad
        if gradient is None:
            if not fill_none:
                continue
            gradient = torch.zeros_like(parameter)
        flat = gradient.detach().reshape(-1)
        complete.append(flat)
        blocks.setdefault(parameter_block(name), []).append(flat)
    if not complete:
        raise ValueError("no trainable gradients found")
    return torch.cat(complete), {name: torch.cat(values) for name, values in blocks.items()}


def cosine(left: torch.Tensor, right: torch.Tensor, epsilon: float = 1e-12) -> float:
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    if float(denominator) <= epsilon:
        return float("nan")
    return float(torch.dot(left, right) / denominator)


def gradient_snr(vectors: Sequence[torch.Tensor], epsilon: float = 1e-12) -> float:
    """Norm of the mean gradient divided by RMS stochastic deviation."""

    if not vectors:
        raise ValueError("vectors must not be empty")
    stacked = torch.stack(tuple(vectors))
    mean = stacked.mean(dim=0)
    noise = torch.sqrt(torch.mean(torch.sum(torch.square(stacked - mean), dim=1)))
    return float(torch.linalg.vector_norm(mean) / torch.clamp(noise, min=epsilon))


def filter_normalized_direction(
    model: nn.Module,
    *,
    generator: torch.Generator | None = None,
    epsilon: float = 1e-12,
) -> dict[str, torch.Tensor]:
    """Draw a random direction normalized to each output filter's norm.

    This follows the filter-normalization principle used for comparable neural
    loss-landscape slices. Biases and vectors are normalized as single filters.
    """

    direction: dict[str, torch.Tensor] = {}
    for name, parameter in trainable_named_parameters(model):
        random = torch.randn(
            parameter.shape,
            dtype=parameter.dtype,
            device=parameter.device,
            generator=generator,
        )
        if parameter.ndim <= 1:
            target_norm = torch.linalg.vector_norm(parameter.detach())
            random_norm = torch.linalg.vector_norm(random)
            direction[name] = random * target_norm / torch.clamp(random_norm, min=epsilon)
            continue
        target = parameter.detach().reshape(parameter.shape[0], -1)
        candidate = random.reshape(parameter.shape[0], -1)
        target_norm = torch.linalg.vector_norm(target, dim=1, keepdim=True)
        candidate_norm = torch.linalg.vector_norm(candidate, dim=1, keepdim=True)
        normalized = candidate * target_norm / torch.clamp(candidate_norm, min=epsilon)
        direction[name] = normalized.reshape_as(parameter)
    return direction


def parameter_snapshot(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().clone()
        for name, parameter in trainable_named_parameters(model)
    }


@torch.no_grad()
def set_parameter_point(
    model: nn.Module,
    base: Mapping[str, torch.Tensor],
    directions: Sequence[tuple[float, Mapping[str, torch.Tensor]]] = (),
) -> None:
    """Set trainable parameters to a base point plus weighted directions."""

    for name, parameter in trainable_named_parameters(model):
        value = base[name]
        for coefficient, direction in directions:
            value = value + coefficient * direction[name]
        parameter.copy_(value)


def _normalize_vector(vector: Sequence[torch.Tensor], epsilon: float = 1e-12) -> list[torch.Tensor]:
    norm = torch.sqrt(sum(torch.sum(torch.square(value)) for value in vector))
    return [value / torch.clamp(norm, min=epsilon) for value in vector]


def hessian_vector_product(
    loss: torch.Tensor,
    parameters: Sequence[nn.Parameter],
    vector: Sequence[torch.Tensor],
) -> list[torch.Tensor]:
    gradients = torch.autograd.grad(loss, parameters, create_graph=True, allow_unused=True)
    safe_gradients = [
        torch.zeros_like(parameter) if gradient is None else gradient
        for parameter, gradient in zip(parameters, gradients)
    ]
    product = sum(torch.sum(gradient * value) for gradient, value in zip(safe_gradients, vector))
    hvp = torch.autograd.grad(product, parameters, allow_unused=True)
    return [
        torch.zeros_like(parameter) if value is None else value.detach()
        for parameter, value in zip(parameters, hvp)
    ]


def top_hessian_eigenvalue(
    loss_closure: Callable[[], torch.Tensor],
    parameters: Sequence[nn.Parameter],
    *,
    iterations: int = 8,
    generator: torch.Generator | None = None,
) -> tuple[float, float, list[float]]:
    """Estimate the algebraically largest Hessian eigenvalue by power iteration."""

    if iterations < 1:
        raise ValueError("iterations must be positive")
    vector = _normalize_vector([
        torch.randn(parameter.shape, device=parameter.device, dtype=parameter.dtype, generator=generator)
        for parameter in parameters
    ])
    history: list[float] = []
    residual = float("nan")
    for _ in range(iterations):
        hvp = hessian_vector_product(loss_closure(), parameters, vector)
        eigenvalue = float(sum(torch.sum(left * right) for left, right in zip(vector, hvp)))
        next_vector = _normalize_vector(hvp)
        residual_tensor = torch.sqrt(sum(
            torch.sum(torch.square(left - eigenvalue * right))
            for left, right in zip(hvp, vector)
        ))
        residual = float(residual_tensor)
        vector = next_vector
        history.append(eigenvalue)
    return history[-1], residual, history


def hutchinson_trace(
    loss_closure: Callable[[], torch.Tensor],
    parameters: Sequence[nn.Parameter],
    *,
    probes: int = 6,
    generator: torch.Generator | None = None,
) -> tuple[float, float, list[float]]:
    """Estimate Hessian trace with Rademacher Hutchinson probes."""

    if probes < 1:
        raise ValueError("probes must be positive")
    estimates: list[float] = []
    for _ in range(probes):
        vector = [
            torch.randint(
                0, 2, parameter.shape, device=parameter.device,
                generator=generator, dtype=torch.int64,
            ).to(parameter.dtype).mul_(2).sub_(1)
            for parameter in parameters
        ]
        hvp = hessian_vector_product(loss_closure(), parameters, vector)
        estimates.append(float(sum(torch.sum(left * right) for left, right in zip(vector, hvp))))
    tensor = torch.tensor(estimates, dtype=torch.float64)
    standard_error = float(tensor.std(unbiased=True) / len(estimates) ** 0.5) if len(estimates) > 1 else 0.0
    return float(tensor.mean()), standard_error, estimates


def rank_correlation(left: Sequence[float], right: Sequence[float]) -> float:
    """Spearman correlation without a SciPy dependency (average ties omitted)."""

    if len(left) != len(right) or len(left) < 2:
        raise ValueError("rank vectors must have equal length >= 2")
    left_tensor = torch.tensor(left, dtype=torch.float64)
    right_tensor = torch.tensor(right, dtype=torch.float64)
    left_rank = torch.argsort(torch.argsort(left_tensor)).to(torch.float64)
    right_rank = torch.argsort(torch.argsort(right_tensor)).to(torch.float64)
    left_rank -= left_rank.mean()
    right_rank -= right_rank.mean()
    denominator = torch.linalg.vector_norm(left_rank) * torch.linalg.vector_norm(right_rank)
    return float(torch.dot(left_rank, right_rank) / denominator)
