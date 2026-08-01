import torch

from hay_single_compartment import MultiResolutionSTFTLoss, StateMSEMultiResolutionSTFTLoss


def test_multi_resolution_stft_is_zero_for_identical_signal_and_has_gradients():
    torch.manual_seed(11)
    target = torch.randn(2, 3, 128)
    prediction = target.clone().requires_grad_(True)
    loss = MultiResolutionSTFTLoss()(prediction, target)
    assert float(loss.detach()) < 1e-6
    loss.backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()


def test_multi_resolution_stft_detects_removed_high_frequency_content():
    time = torch.arange(128, dtype=torch.float32)
    target = (torch.sin(2.0 * torch.pi * time / 4.0) + 0.2 * torch.sin(2.0 * torch.pi * time / 32.0))[None, None]
    smoothed = torch.nn.functional.avg_pool1d(target, kernel_size=9, stride=1, padding=4)
    criterion = MultiResolutionSTFTLoss()
    assert float(criterion(smoothed, target)) > float(criterion(target, target)) + 0.5


def test_state_composite_scale_zero_is_exact_mse():
    torch.manual_seed(12)
    prediction = torch.randn(2, 128, 5, requires_grad=True)
    target = torch.randn_like(prediction)
    criterion = StateMSEMultiResolutionSTFTLoss(spectral_weight=0.1)
    criterion.set_spectral_scale(0.0)
    loss, terms = criterion(prediction, target)
    torch.testing.assert_close(loss, torch.mean(torch.square(prediction - target)))
    assert set(terms) == {"mse", "mrstft", "spectral_convergence", "log_magnitude"}
