
import torch
import pytest 
from diffusion.sampler_utils import (
    dynamic_thresholding, 
    cfg_sample, 
    scheduled_guidance_scale, 
    apply_self_conditioning,
    compute_snr,
    snr_weight, 
    importance_sample_timesteps,
    debug_denoising,
)



# === Test: Dynamic Thresholding ===
class TestDynamicThresholding:
    def test_output_range(self):
        x = torch.randn(4, 3, 64, 64) * 0.5     # exagerated values
        out = dynamic_thresholding(x, percentile=0.95)
        assert out.min() >= -1.0 and out.max() <= 1.0, "output not clamped correctly"

    def test_preserves_shape(self):
        x = torch.randn(2, 3, 32, 32)
        out = dynamic_thresholding(x)
        assert out.shape == x.shape, "Shape mismacth after dynamic thresholding"

# === Test: Classifier-Free Guidance === 
class TestClassifierFreeGuidance:
    class DummyModel(torch.nn.Module):
        def forward(self, x, t, cond):
            return cond + 0.1 * x   # toy fn
        
    def test_cfg_output_diff(self):
        model = self.DummyModel()
        x = torch.randn(2, 4)
        t = torch.tensor([10, 20])
        cond = torch.ones_like(x)
        uncond = torch.zeros_like(x)

        out = cfg_sample(model, x, t, cond, uncond, guidance_scale=2.0)
        expected = 0.0 + 2.0 * (1.0 - 0.0)
        assert torch.allclose(out.mean(), torch.tensor(expected), atol=0.2)

    def test_scheduled_guidance_values(self):
        t = torch.tensor([0, 250, 500, 750, 1000])
        scaled = scheduled_guidance_scale(t, max_scale=7.0)
        assert torch.all(scaled <= 7.0) and torch.all(scaled >= 0.0)
        assert scaled[0] > scaled[-1], "scheduled scale should decay over time."


# === Test: Self-Conditioning ===
class TestSelfConditioning:
    class DummyModel(torch.nn.Module):
        def forward(self, x, t, cond=None):
            return x[:, :3] # just return the x-part (simulate truction)
        
        def test_concat_behavior(self):
            model = self.DummyModel()
            x = torch.randn(2, 3, 64, 64)
            t = torch.tensor([10, 10])
            cond = None
            x_prev = torch.randn_like(x)
            out = apply_self_conditioning(model, x, t, cond=cond, x_prev=x_prev)
            assert out.shape == x.shape, "Self-conditioned output shape mismatch"


# === Test: SNR and Loss Weights === 
class TestSNRAndWeights:
    def test_snr_monotonicity(self):
        alphas_cumprod = torch.linspace(0.01, 0.99, 1000)
        t  = torch.tensor([10, 100, 500])
        snr = compute_snr(t, alphas_cumprod)
        assert snr[0] < snr[-1], "SNR should grow with t"

    def test_snr_weight_output(self):
        alphas_cumprod = torch.linspace(0.01, 0.99, 1000)
        t = torch.randint(0, 999, (32,))
        weights = snr_weight(t, alphas_cumprod, gamma=2.0)
        assert torch.isfinite(weights).all()
        assert (weights >= 0).all()

# === Test: Curriculum Timestep Sampling === 
class TestTimestepCurriculum:
    def test_progressive_bounds(self):
        for epoch in [1, 25, 50, 75, 100]:  # problem with epoch 0
            t = importance_sample_timesteps(batch_size=128, epoch=epoch, max_epoch=100)
            assert t.min() >= 20 and t.max() <= 999 

    def test_shape_correctness(self):
        t = importance_sample_timesteps(batch_size=64, epoch=50, max_epoch=100)
        assert t.shape == (64,)


# === Test: Debug Hook (smoke only) ===
class TestDebugHooks:
    class DummyModel(torch.nn.Module):
        def forward(self, x, t, cond=None):
            return x * 0.5
        
    def test_debug_prints_and_runs(self, capsys):
        x = torch.randn(1, 3, 32, 32)
        t = torch.tensor([100])
        model = self.DummyModel()
        _ = debug_denoising(model, x, t)
        captured = capsys.readouterr()
        assert "Step 100" in captured.out



















































