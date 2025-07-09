
import torch
import torch.nn as nn
from typing import Callable, Dict
from einops import rearrange
from modules.attention.registry import get_attention
from modules.attention.vanilla_attention import VanillaAttention    # direct import for now
from modules.norm_utils import get_norm_layer

# consider implementing debugging as in inital one

# ----------------------------------------------------------------------------------
# Block Registry
# ----------------------------------------------------------------------------------

BLOCK_REGISTRY: Dict[str, Callable] = {}

def register_block(name: str):
    def decorator(cls):
        BLOCK_REGISTRY[name] = cls
        return cls
    return decorator

# ----------------------------------------------------------------------------------
# Base Block
# ----------------------------------------------------------------------------------

class BaseResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim, norm_type="group", use_attention=False):
        super().__init__()
        self.in_ch, self.out_ch = in_ch, out_ch
        self.use_attention = use_attention

        self.norm1 = get_norm_layer(norm_type, in_ch)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        self.time_proj = nn.Linear(time_dim, out_ch)

        self.norm2 = get_norm_layer(norm_type, out_ch)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.res_conv = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

        if use_attention:
            self.attn = VanillaAttention()


    def forward(self, x, t_emb):
        h = self.conv1(self.act1(self.norm1(x)))
        h += self.time_proj(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(self.act2(self.norm2(h)))

        h += self.res_conv(x)
        if self.use_attention:
            h = self.attn(h)

        return h

# ----------------------------------------------------------------------------------------
# FiLM-style Block
# ----------------------------------------------------------------------------------------

@register_block("film")
class FiLMResBlock(BaseResBlock):
    def __init__(self, in_ch, out_ch, time_dim, norm_type="group", use_attention=False):
        super().__init__(in_ch, out_ch, time_dim, norm_type, use_attention)

        self.time_proj = nn.Linear(time_dim, in_ch * 2)

    def forward(self, x, t_emb):
        h = self.norm1(x)
        scale_shift = self.time_proj(t_emb).unsqueeze(-1).unsqueeze(-1)
        scale, shift = scale_shift.chunk(2, dim=1)

        h = self.act1(h * (1 + scale) + shift)
        h = self.conv1(h)

        h = self.act2(self.norm2(h))
        h = self.conv2(h)

        h += self.res_conv(x)
        if self.use_attention:
            h = self.attn(h)
        return h


# ----------------------------------------------------------------------------------------
# Vanilla Block (same as Base, but registered)
# ----------------------------------------------------------------------------------------

@register_block("vanilla")
class VanillaResBlock(BaseResBlock):
    pass
    
# ----------------------------------------------------------------------------------------
# ResBlock Getter
# ----------------------------------------------------------------------------------------

def get_resblock(kind: str, in_ch: int, out_ch: int, time_dim: int, norm_type: str, use_attention: bool):
    if kind not in BLOCK_REGISTRY:
        raise ValueError(f"Unknowed ResBlock type: {kind}")
    return BLOCK_REGISTRY[kind](in_ch, out_ch, time_dim, norm_type, use_attention)









