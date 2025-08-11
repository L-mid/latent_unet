
import torch
import math
from dataclasses import dataclass
from typing import Literal, Optional


# ---- BETA SCHEDULE ----

def make_beta_schedule(
        schedule_type: Literal["linear", "cosine", "quadractic"],
        timesteps: int,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
) -> torch.Tensor:
    if schedule_type == "linear":
        return torch.linspace(beta_start, beta_end, timesteps)
    
    elif schedule_type == "cosine":
        return betas_for_alpha_bar(timesteps)
    
    elif schedule_type == "quadratic":
        return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps)
    
    else:
        raise ValueError(f"Unknown beta schedule type: {schedule_type}")
    

def betas_for_alpha_bar(timesteps: int, max_beta=0.999, eps=1e-8, device=None, dtype=torch.float32) -> torch.Tensor:
    def alpha_bar(t):
        return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
    
    betas = []
    for i in range(timesteps):
        t1 = i / timesteps
        t2 = (i + 1) / timesteps
        ab1 = max(alpha_bar(t1), eps)       # avoid divide by zero
        ab2 = alpha_bar(t2)
        beta = min(1 - ab2 / ab1, max_beta) # keep in (0, max_beta)
        betas.append(beta)

    return torch.tensor(betas, dtype=dtype, device=device)


# ---- SCHEDULE WRAPPER ----

@dataclass
class DiffusionSchedule:
    betas: torch.Tensor
    alphas: torch.Tensor
    alphas_cumprod: torch.Tensor
    alphas_cumprod_prev: torch.Tensor
    sqrt_alphas_cumprod: torch.Tensor
    sqrt_one_minus_alphas_cumprod: torch.Tensor
    log_one_minus_alphas_cumprod: torch.Tensor
    sqrt_recip_alphas: torch.Tensor
    sqrt_recip_alphas_cumprod: torch.Tensor
    sqrt_recipm1_alphas_cumprod: torch.Tensor
    posterior_variance: torch.Tensor
    posterior_log_variance_clipped: torch.Tensor
    posterior_mean_coef1: torch.Tensor
    posterior_mean_coef2: torch.Tensor

    def to(self, device):
        for field in self.__dataclass_fields__:
            setattr(self, field, getattr(self, field).to(device))
            return self
        

def get_diffusion_schedule(
    schedule_type: str,
    timesteps: int,
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> DiffusionSchedule:
    # 1) betas
    betas = make_beta_schedule(schedule_type, timesteps, beta_start, beta_end)
    assert betas.ndim == 1 and betas.shape[0] == timesteps

    # 2) alphas (+ culmulative)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    # careful to keep device/dtype consitent
    one = torch.ones(1, device=device, dtype=dtype)
    alphas_cumprod_prev = torch.cat([one, alphas_cumprod[:-1]], dim=0)

    # 3) handy precomputes (correct formulas)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(torch.clamp(1.0 - alphas_cumprod, min=1e-20))
    log_one_minus_alphas_cumprod = torch.log(torch.clamp(1.0 - alphas_cumprod, min=1e-20))
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod)
    sqrt_recipm1_alphas_cumprod = torch.sqrt(torch.clamp(1.0 / alphas_cumprod - 1.0, min=0.0))

    # 4) posterior terms (DDPM eq. 7 & impl tricks)
    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / torch.clamp(1.0 - alphas_cumprod, min=1e-20)
    # log variance clipped: use t=1 value for t=0 to avoid -inf
    posterior_log_variance_clipped = torch.log(
        torch.cat([posterior_variance[1:2], posterior_variance[1:]], dim=0)
    )

    # 5) mean coefficients (for q(x_{t-1} | x_t, x_0))
    posterior_mean_coef1 = torch.sqrt(alphas_cumprod_prev) * betas / torch.clamp(1.0 - alphas_cumprod, min=1e-20)
    posterior_mean_coef2 = torch.sqrt(alphas) * (1.0 - alphas_cumprod_prev) / torch.clamp(1.0 - alphas_cumprod, min=1e-20)


    return DiffusionSchedule(
    betas=betas,
    alphas=alphas,
    alphas_cumprod=alphas_cumprod,
    alphas_cumprod_prev=alphas_cumprod_prev,
    sqrt_alphas_cumprod=sqrt_alphas_cumprod,
    sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod,
    log_one_minus_alphas_cumprod=log_one_minus_alphas_cumprod,
    sqrt_recip_alphas=sqrt_recip_alphas, #
    sqrt_recip_alphas_cumprod=sqrt_recip_alphas_cumprod,
    sqrt_recipm1_alphas_cumprod=sqrt_recipm1_alphas_cumprod,
    posterior_variance=posterior_variance,
    posterior_log_variance_clipped=posterior_log_variance_clipped,
    posterior_mean_coef1=posterior_mean_coef1,
    posterior_mean_coef2=posterior_mean_coef2,
)





