
import torch
import torch.nn.functional as F
from tqdm import tqdm
from diffusion.sampler_registry import register_sampler
from typing import Optional, Callable

from .schedule import make_beta_schedule
from .sampler_utils import cfg_sample


@register_sampler("ddim")
@torch.no_grad()
def ddim_sample(
    model: torch.nn.Module,
    shape: tuple,
    cfg=None,
    num_steps: int = 50,
    eta: float = 0.0,
    guidance_scale: float = 1.0,
    cond: Optional[dict] = None,
    unconditional_cond: Optional[dict] = None,
    cfg_fn: Callable = cfg_sample,
    noise_schedule: str = "cosine",
    device: str = "cuda"
):
    if cfg:                         
        eta = cfg.sampler.ddim_eta          # not gonna omegaconf
        num_steps = cfg.sampler.num_steps
        device = cfg.device

    B, C, H, W = shape                      # just image size?
    x = torch.randn(shape, device=device)
    
    def get_ddim_schedule(num_steps, schedule="cosine", eta=0.0, device="cuda"):
    # Compute the DDIM alpha, sigma, and timestep schedules.

        alphas_cumprod = make_beta_schedule(schedule_type=schedule, timesteps=1000).to(device)
        total_steps = len(alphas_cumprod)

        step_indices = torch.linspace(0, total_steps - 1, num_steps, device=device).long()

        alphas = alphas_cumprod[step_indices]
        alphas_prev = torch.cat([alphas[:1], alphas[:-1]])
        sigmas = eta * torch.sqrt((1 - alphas_prev) / (1 - alphas) * (1 - alphas / alphas_prev))

        return step_indices, alphas, alphas_prev, sigmas    
    

    # Schedule
    timesteps, alphas, alphas_prev, sigmas = get_ddim_schedule(num_steps, schedule=noise_schedule, eta=eta, device=device)

    for i, t in enumerate(timesteps):
        t_batch = torch.full((B,), t, device=device, dtype=torch.long)

        # Classifier-Free Guidance
        model_input = cfg_fn(model, x, t_batch, cond=cond, uncond=unconditional_cond, guidance_scale=guidance_scale)

        alpha = alphas[i]
        alpha_prev = alphas_prev[i]
        sigma = sigmas[i]
        sqrt_one_minus_alpha = torch.sqrt(1.0 - alpha)

        # Predict noise e_theta
        eps = model_input

        # Predict x_0
        x0 = (x - sqrt_one_minus_alpha * eps) / torch.sqrt(alpha)
        
        # Compute sigma_t for DDIM
        sigma = eta * torch.sqrt((1 - alpha_prev)/(1 - alpha) * (1 - alpha/alpha_prev))
        
        # Direction pointing to x_t
        dir_x = torch.sqrt(1 - alpha_prev - sigma**2)
        
        # Add noise if not the last step
        noise = sigma * torch.randn_like(x)

        x_prev = torch.sqrt(alpha_prev) * x0 + dir_x + sigma + noise 

        return x_prev
    
    
import os
from utils.visualizer_refactor import _save_image_grid, _ensure_dir

# vizualizer example helper (ddim)

def sample_ddim_grid(
        cfg, model, shape, step=None, tag: str = "samples/ddim_samples"           # if calling changes
):
    nrow = cfg.viz.nrow
    root = os.path.abspath(cfg.viz.output_dir)
    _ensure_dir(root)


    imgs = ddim_sample(
        cfg=cfg, model=model, shape=shape,
    ) 

    # file naming helpers:
    def _path(root, *parts: str) -> str:        # root is nomally vis's head, here it's the tag
        path = os.path.join(root, *parts)
        return path
    
    def _fmt_step(step: Optional[int]) -> str:
        return f"{int(step):06d}" if step is not None else "no_step"
    
    # making the full path name
    path = _path(root, f"{tag}_step{_fmt_step(step)}.png")


    # saving the final image
    _save_image_grid(tensors=imgs, save_path=path, nrow=nrow)

    # for loggers
    from torchvision.utils import make_grid
    grid = make_grid(imgs.detach().cpu(), nrow=nrow, pad_value=0.5)     # or 'x', the final saved tensor

    return grid          # tensor for logging 

































