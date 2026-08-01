"""Standard spectral reconstruction losses for continuous state trajectories."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F


class MultiResolutionSTFTLoss(nn.Module):
    """Multi-resolution STFT magnitude loss.

    This follows the spectral-convergence plus log-magnitude construction used
    for neural waveform modelling.  Leading dimensions are treated as
    independent traces and the final dimension is time.
    """

    def __init__(
        self,
        fft_sizes: Sequence[int] = (64, 32, 16),
        hop_sizes: Sequence[int] = (16, 8, 4),
        win_lengths: Sequence[int] = (64, 32, 16),
        spectral_convergence_weight: float = 1.0,
        log_magnitude_weight: float = 1.0,
        epsilon: float = 1e-7,
    ) -> None:
        super().__init__()
        if not (len(fft_sizes) == len(hop_sizes) == len(win_lengths)) or not fft_sizes:
            raise ValueError("fft_sizes, hop_sizes and win_lengths must be nonempty and aligned")
        resolutions = tuple(zip(map(int, fft_sizes), map(int, hop_sizes), map(int, win_lengths)))
        if any(fft < 2 or hop < 1 or win < 2 or win > fft for fft, hop, win in resolutions):
            raise ValueError("invalid STFT resolution")
        if spectral_convergence_weight < 0 or log_magnitude_weight < 0 or epsilon <= 0:
            raise ValueError("loss weights must be nonnegative and epsilon positive")
        self.resolutions = resolutions
        self.spectral_convergence_weight = float(spectral_convergence_weight)
        self.log_magnitude_weight = float(log_magnitude_weight)
        self.epsilon = float(epsilon)
        for index, (_, _, win_length) in enumerate(resolutions):
            self.register_buffer(f"window_{index}", torch.hann_window(win_length), persistent=False)

    def components(self, prediction: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        if prediction.shape != target.shape or prediction.ndim < 2:
            raise ValueError("prediction and target must have matching [...,time] shapes")
        prediction = prediction.float().reshape(-1, prediction.shape[-1])
        target = target.float().reshape(-1, target.shape[-1])
        convergence_terms, magnitude_terms = [], []
        for index, (fft_size, hop_size, win_length) in enumerate(self.resolutions):
            if fft_size > prediction.shape[-1]:
                continue
            window = getattr(self, f"window_{index}").to(prediction)
            predicted_stft = torch.stft(
                prediction,
                n_fft=fft_size,
                hop_length=hop_size,
                win_length=win_length,
                window=window,
                center=False,
                return_complex=True,
            )
            target_stft = torch.stft(
                target,
                n_fft=fft_size,
                hop_length=hop_size,
                win_length=win_length,
                window=window,
                center=False,
                return_complex=True,
            )
            predicted_magnitude = predicted_stft.abs().clamp_min(self.epsilon)
            target_magnitude = target_stft.abs().clamp_min(self.epsilon)
            difference = torch.linalg.vector_norm(
                target_magnitude - predicted_magnitude, dim=(-2, -1)
            )
            denominator = torch.linalg.vector_norm(target_magnitude, dim=(-2, -1)).clamp_min(
                self.epsilon
            )
            convergence_terms.append((difference / denominator).mean())
            magnitude_terms.append(
                F.l1_loss(predicted_magnitude.log(), target_magnitude.log())
            )
        if not convergence_terms:
            raise ValueError("time dimension is shorter than every configured FFT size")
        return {
            "spectral_convergence": torch.stack(convergence_terms).mean(),
            "log_magnitude": torch.stack(magnitude_terms).mean(),
        }

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        terms = self.components(prediction, target)
        return (
            self.spectral_convergence_weight * terms["spectral_convergence"]
            + self.log_magnitude_weight * terms["log_magnitude"]
        )


class StateMSEMultiResolutionSTFTLoss(nn.Module):
    """Normalized-state MSE plus standard MR-STFT over every state trace."""

    def __init__(
        self,
        spectral_weight: float = 0.1,
        fft_sizes: Sequence[int] = (64, 32, 16),
        hop_sizes: Sequence[int] = (16, 8, 4),
        win_lengths: Sequence[int] = (64, 32, 16),
    ) -> None:
        super().__init__()
        if spectral_weight < 0:
            raise ValueError("spectral_weight must be nonnegative")
        self.spectral_weight = float(spectral_weight)
        self.spectral_scale = 1.0
        self.spectral = MultiResolutionSTFTLoss(fft_sizes, hop_sizes, win_lengths)

    def set_spectral_scale(self, value: float) -> None:
        if value < 0:
            raise ValueError("spectral scale must be nonnegative")
        self.spectral_scale = float(value)

    def forward(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if prediction.shape != target.shape or prediction.ndim != 3:
            raise ValueError("normalized states must have matching [batch,time,state] shapes")
        mse = F.mse_loss(prediction, target)
        if self.spectral_scale == 0.0 or self.spectral_weight == 0.0:
            zero = mse.new_zeros(())
            return mse, {
                "mse": mse,
                "mrstft": zero,
                "spectral_convergence": zero,
                "log_magnitude": zero,
            }
        spectral_components = self.spectral.components(
            prediction.transpose(1, 2), target.transpose(1, 2)
        )
        spectral = (
            self.spectral.spectral_convergence_weight
            * spectral_components["spectral_convergence"]
            + self.spectral.log_magnitude_weight * spectral_components["log_magnitude"]
        )
        total = mse + self.spectral_weight * self.spectral_scale * spectral
        return total, {
            "mse": mse,
            "mrstft": spectral,
            "spectral_convergence": spectral_components["spectral_convergence"],
            "log_magnitude": spectral_components["log_magnitude"],
        }
