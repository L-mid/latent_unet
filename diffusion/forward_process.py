
import torch
from diffusion.schedule import get_diffusion_schedule

class ForwardProcess:
    def __init__(self, schedule="cosine", timesteps=1000, device="cpu"):
        self.timesteps = timesteps
        self.device = device

        # === 1. Cache the full noise schedule ===
        sched = get_diffusion_schedule(schedule_type=schedule, timesteps=timesteps)
        self.betas = sched.betas
        self.alphas = sched.alphas
        self.alphas_cumprod = sched.alphas_cumprod
        self.sqrt_alphas_cumprod = sched.sqrt_alphas_cumprod
        self.sqrt_one_minus_alphas_cumprod = sched.sqrt_one_minus_alphas_cumprod

    def extract(self, tensor, t, shape):
        # Grab per-timestep values and broadcast to input shape
        out = tensor.gather(0, t).float()
        return out.reshape(-1, 1, 1, 1).expand(shape)
    
    def q_sample(self, x_start, t, noise=None, return_noise=False):
        # Sample from q(x_t | x_0)
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alpha = self.extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) #issue was here
        sqrt_one_minus = self.extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        x_t = sqrt_alpha * x_start + sqrt_one_minus * noise
        return (x_t, noise) if return_noise else x_t
    
    def get_snr_weights(self, t):
        # Compute SNR = alpha / (1 - alpha) for loss weighting.
        alpha_bar = self.extract(self.alphas_cumprod, t, (t.shape[0], 1, 1, 1))
        snr = alpha_bar / (1 - alpha_bar)
        return snr
    
    def q_posterior(self, x_start, x_t, t):
        # (Optional) q(x_{t-1} | x_t, x_0) for ELB0 / DDPM++ loss.
        raise NotImplementedError

    def visualize_trajectory(self, x0, steps=[0, 250, 500, 750, 999]):
        # Return noise samples for multiple timesteps (for GIFs or debugging)
        x_t_series = []
        for t_scalar in steps:
            t = torch.full((x0.shape[0],), t_scalar, dtype=torch.long, device=x0.device)
            x_t = self.q_sample(x0, t)
            x_t_series.append(x_t)
        return x_t_series
    

