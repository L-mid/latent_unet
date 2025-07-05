
import math
import torch
import torch.nn as nn
from typing import Literal

# === Sinusoidal Embedding ===
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self._dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        # timesteps: [B] or [B, 1] scalar timestep index
        # Returns: [B, dim]

        half_dim = self._dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    

# === Learned Embedding ===
class LearnedTimeEmbedding(nn.Module):
    def __init__(self, dim: int, max_steps: int = 1000):
        super().__init__()
        self.embed = nn.Embedding(max_steps, dim)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        return self.embed(timesteps)
    

# === FiLM-style Modulation Embedding ===
class FiLMTimeEmbedding(nn.Module):
    def __init__(self, dim: int, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim * 2)  # Outputs can scale and shift
        )

    def forward(self, t_emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        t_emb = t_emb.to(self.net[0].weight.dtype)
        out = self.net(t_emb)   # [B, 2 * dim]
        scale, shift = out.chunk(2, dim=-1)
        return scale, shift     # Would be applied in FiLM layers

class FiLMStackedEmbedding(nn.Module):
    # Combines sinusoidal embedding + FiLM projection.
    # Returns (scale, shift)
    def __init__(self, dim: int, hidden_dim: int = 512):
        super().__init__()
        self.sinusoidal = SinusoidalTimeEmbedding(dim)
        self.film = FiLMTimeEmbedding(dim, hidden_dim)

    def forward(self, timesteps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        t_emb = self.sinusoidal(timesteps)
        return self.film(t_emb)

# ==============================================================================

def get_time_embedding(cfg) -> nn.Module:
    # cfg: config.time_embedding
    # Expects cfg.kind in {"sinusoidal", "learned", "film"}

    kind = cfg.kind.lower()
    dim = cfg.dim

    if kind == "sinusoidal":
        return SinusoidalTimeEmbedding(dim)
    if kind == "learned":
        return LearnedTimeEmbedding(dim)
    if kind == "film":
        return FiLMStackedEmbedding(dim)
    else: 
        raise ValueError(f"Unknown time embedding type: {kind}")
    

