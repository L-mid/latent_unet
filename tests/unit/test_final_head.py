
import torch
import pytest
from modules.final_head import FinalHead


class TestFinalHead:

    def test_output_shape(self):
        model = FinalHead(in_ch=128, out_ch=3)
        x = torch.randn(4, 128, 32, 32)
        y = model(x)
        assert y.shape == (4, 3, 32, 32)

    def test_backprop(self):
        model = FinalHead(in_ch=64, out_ch=3)
        x = torch.randn(2, 64, 16, 16, requires_grad=True)
        y = model(x)
        y.mean().backward()
        assert x.grad is not None, "No gradient"
        assert torch.isfinite(x.grad).all(), "Gradient contains NaNs"

    def test_zero_output_channels(self):
        with pytest.raises(ValueError):
            _ = FinalHead(in_ch=64, out_ch=0)

    def test_spatial_consistency(self):
        model = FinalHead(in_ch=32, out_ch=3)
        x = torch.randn(1, 32, 64, 64)
        y = model(x)
        assert y.shape == (1, 3, 64, 64), "Spatial size mismatch"

    def test_parameter_count(self):
        in_ch = 16
        out_ch = 4
        model = FinalHead(in_ch, out_ch)
        param_count = sum(p.numel() for p in model.parameters())
        expected = 16 * 4 * 3 * 3 + 4 + 2 * in_ch   # Weights + biases of Conv2d + GroupNorm
        assert model.block[-1].kernel_size == (3, 3)
        assert param_count == expected, f"Expected {expected}, got {param_count}"

    def test_output_distribution(self):
        model = FinalHead(64, 3)
        x = torch.randn(16, 64, 32, 32)
        y = model(x)
        assert torch.abs(y.mean()) < 1.0, "Mean too large, mabye uninitalized?"
        assert 0.1 < torch.abs(y.std()) < 1.5, "Standard deviation too small"

