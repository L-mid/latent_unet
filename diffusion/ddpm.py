
import torch
import torch.nn.functional as F
from typing import Tuple, Optional

from diffusion.schedule import make_beta_schedule
from diffusion.sampler_registry import register_sampler

@register_sampler("ddpm")
class DDPM:
    def __init__(
            self,
            model,
            timesteps: int = 1000,
            prediction_type: str = "epsilon",   # or "x0"
            device: str = None
    ):
        self.model = model
        self.timesteps = timesteps
        self.device = device
        self.prediction_type = prediction_type

        # Load beta and precompute schedules
        betas = make_beta_schedule(schedule_type="cosine", timesteps=timesteps).to(device)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
        
        pv = betas[1:] * (1. - alphas_cumprod[:-1]) / (1. - alphas_cumprod[1:])
        self.posterior_varience = torch.cat([
            pv[:1],     # or a tiny epsilon instead of duplicating
            pv
        ], dim=0)


    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None):
        # Forward diffusion: sample x_t from x_0 with q(x_t | x_0)

        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alpha = self.extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alpha = self.extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alpha * x_start + sqrt_one_minus_alpha * noise
    
    def p_losses(self, x_0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None):
        # Compute model loss for given x_0 and time t.

        if noise is None:
            noise = torch.randn_like(x_0)
            x_noisy = self.q_sample(x_0, t, noise)
            pred = self.model(x_noisy, t)

            if self.prediction_type == "epslion":
                target = noise
            elif self.prediction_type == "x0":
                target = x_0
            else: 
                raise ValueError(f"Invalid prediction_type: {self.prediction_type}")
            
            return F.mse_loss(pred, target)
        
    @torch.no_grad()
    def p_sample(self, x_t: torch.Tensor, t: torch.Tensor):
        # One reverse sampling step: p(x_{t-1} | x_t)

        model_out = self.model(x_t, t)

        if self.prediction_type == "epsilon":
            x_0_pred = self._predict_x0_from_eps(x_t, t, model_out) #
        elif self.prediction_type == "x0":
            x_0_pred = model_out
        else: 
            raise ValueError("Invaild prediction type")
        
        coef1 = self._extract(self.betas, t, x_t.shape)
        coef2 = self._extract(self.alphas, t, x_t.shape)
        posterior_mean = (
            self._extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_0_pred +
            self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * model_out
        )

        if (t == 0).all():
            return x_0_pred
        else:
            noise = torch.randn_like(x_t)
            var = self._extract(self.posterior_varience, t - 1, x_t.shape)
            return posterior_mean + torch.sqrt(var) * noise
            
    @torch.no_grad()
    def sample(self, shape: Tuple[int], return_all=False):
    # Generate a full denoised image from pure noise using ancestral sampling.
        x = torch.randn(shape).to(self.device)
        xs = [x.clone()] if return_all else None

        for t in reversed(range(self.timesteps)):
            t_tensor = torch.full((shape[0],), t, device=self.device, dtype=torch.long)
            x = self.p_sample(x, t_tensor)
            if return_all:
                xs.append(x.clone())

        return x if not return_all else torch.stack(xs, dim=1)

    def _extract(self, a: torch.Tensor, t: torch.Tensor, shape: torch.Size):  
        # Helper to match tensor dimenstions
        out = a.gather(-1, t)
        return out.view(-1, *((1,) * (len(shape) - 1)))   
    
    def _predict_x0_from_eps(self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor):
        # Convert predicted noise into predicted x_0.
        return (x_t - self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) / \
                self._extract(self.sqrt_alphas_cumprod, t, x_t.shape))




