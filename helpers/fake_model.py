
import torch
import torch.nn as nn

# === NOTES:
"""
Unused. Untested.

"""

class FakeNoisePredictor(nn.Module):
    # A fully mockable noise predictor model.

    def __init__(self, channels=3, constant_value=0.0, scale_t_embedding=False):
        super().__init__()
        self.constant_value = constant_value
        self.channels = channels
        self.scale_t_embedding = scale_t_embedding

    def forward(self, x, t, *args, **kwargs):
        """
        Simulates model forward: predicts noise given x_t and timestep t.
        'x' is (B, C, H, W)
        't' is (B,) timesteps
        If 'scale_t_embedding is True, injects deterministic timestep scaling.        
        """
          
        B, C, H, W = x.shape
        out = torch.full((B, self.channels, H, W), self.constant_value, device=x.device, dtype=x.dtype)

        if self.scale_t_embedding:
            # Inject simple dependency on timestep for testing samplers
            scale = (t.float().unsqueeze(-1).unqueeze(-1).unsqueeze(-1) / 1000.0)
            out += scale

        return out
    

class NoidyIdentityModel(nn.Module):
    # Another toy model: outputs slightly noised inputs instead of true prediction.

    def __init__(self, noise_scale=0.01):
        super().__init__()
        self.noise_scale = noise_scale

    def forward(self, x, t, *args, **kwargs):
        noise = torch.randn_like(x) * self.noise_scale
        return x + noise
    

# ---- Factory methods for easy instantiation ----

def build_fake_model_for_loss_tests():
    # Cheap pure-zero noise prediction for testing loss fns.
    return FakeNoisePredictor(constant_value=0.0)

def build_fake_model_for_sampler_tests():
    # Predicts small positive outputs depending on timestep, simulating basic learning
    return FakeNoisePredictor(constant_value=0.1, scale_t_embedding=True)


























