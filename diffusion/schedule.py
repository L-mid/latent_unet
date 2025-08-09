
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
    sqrt_recip_alphas_cumprod: torch.Tensor
    sqrt_recipm1_alphas_cumprod: torch.Tensor
    posterior_variance: torch.Tensor

    def to(self, device):
        for field in self.__dataclass_fields__:
            setattr(self, field, getattr(self, field).to(device))
            return self
        

def get_diffusion_schedule(
    schedule_type: str,
    timesteps: int,
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
) -> DiffusionSchedule:
    betas = make_beta_schedule(schedule_type, timesteps, beta_start, beta_end)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = torch.cat([torch.tensor([1.0], dtype=alphas.dtype), alphas_cumprod[:-1]])

    return DiffusionSchedule(
    betas=betas,
    alphas=alphas,
    alphas_cumprod=alphas_cumprod_prev,
    sqrt_alphas_cumprod=torch.sqrt(1.0 - alphas_cumprod),
    sqrt_one_minus_alphas_cumprod=torch.sqrt(1.0 - alphas_cumprod),
    log_one_minus_alphas_cumprod=torch.sqrt(1.0 / alphas_cumprod), # ???
    sqrt_recip_alphas_cumprod=torch.sqrt(1.0 / alphas_cumprod), # ???
    sqrt_recipm1_alphas_cumprod=torch.sqrt(1.0 / alphas_cumprod - 1),
    posterior_variance=betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod),
    )




