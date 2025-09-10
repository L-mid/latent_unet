
import torch 
import torch.nn.functional as F
from typing import Optional, Tuple

# === Dynamic Threshholding (Imagen-style) ===
@torch.no_grad()
def dynamic_thresholding(x0: torch.Tensor, percentile: float = 0.995) -> torch.Tensor:
    # Clamp and normalize outputs based on per-sample dynamic range.
    # Helps prevent oversaturated reconstructions.

    s = torch.quantile(x0.abs().flatten(1), percentile, dim=1).clamp(min=1.0)
    s = s.view(-1, 1, 1, 1)
    return x0.clamp(-s, s) / s


# === Classifier-Free Guidance (with scheduled scale) ===
@torch.no_grad()
def cfg_sample(model, x, t, cond, uncond, guidance_scale: float = 5.0) -> torch.Tensor:
    # Classic classifier-free guidance (cond + uncond interpolation).
    eps_uncond = model(x, t, uncond)
    eps_cond = model(x, t, cond)
    return eps_uncond + guidance_scale * (eps_cond - eps_uncond)

def scheduled_guidance_scale(t: torch.Tensor, max_scale: float = 5.0) -> torch.Tensor:
    # Returns a timestep-dependent guidance scale. Useful in CFG where
    # you want to suppress scale at small t (late steps).

    return max_scale * (1.0 - t.float() / 1000.0)

@torch.no_grad()
def apply_self_conditioning(model, x, t, cond=None, x_prev=None):
    # Self conditioning wrapper: pass previous x0 as extra input.
    # Assumes model accepts extra channel input (e.g. 6 channels).
    if x_prev is None:
        x_prev = torch.zeros_like(x)
    x_input = torch.cat([x, x_prev], dim=1)
    return model(x_input, t, cond)


# === SNR-Based Sample Weighting (for losses) ===
def compute_snr(t: torch.Tensor, alphas_cumprod: torch.Tensor) -> torch.Tensor:
    # Compute signal-to-noise ratio for each timestep.
    alpha_bar = alphas_cumprod[t.long()]
    return alpha_bar / (1 - alpha_bar + 1e-8)


def snr_weight(t: torch.Tensor, alphas_cumprod: torch.Tensor, gamma: float = 5.0) -> torch.Tensor:
    snr = compute_snr(t, alphas_cumprod)
    return (snr / snr.mean()).pow(gamma)



# === Timestep Curriculm Sampling ===
def importance_sample_timesteps(batch_size: int, epoch: int, max_epoch: int,
                                t_min: int = 20, t_max: int = 999) -> torch.Tensor:
        # Prefer smaller timesteps in early epochs, for curriculum training.
        max_t = int(t_min + (t_max - t_min) * min(epoch / max_epoch, 1.0))
        return torch.randint(t_min, max_t, (batch_size,))


# === Debugging Hooks ===
@torch.no_grad()
def debug_denoising(model, x, t, cond=None):
    # Logs mean/std of prediction and runtime. Can be used for profiling.
    import time
    start = time.time()
    pred = model(x, t, cond)
    elapsed = time.time() - start
    print(f"[Step {t.item()}] u: {pred.mean().item():3f} σ: {pred.std().item():.3f} | Time: {elapsed:3f}s")
    return pred








