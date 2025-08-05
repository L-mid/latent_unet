
from einops import rearrange
import math
from .registry import register_attention
from .base_attention import BaseAttention
import torch.nn as nn
import torch

@register_attention("vanilla")
class VanillaAttention(BaseAttention):
    def __init__(
        self, 
        dim: int, 
        num_heads: int = 8, 
        norm_groups: int = 8,
        dim_head: int = None,
        start_layer: int = 0,
        window_size: int = None,
        backend: str = None, 
        **kwargs
    ):
        super().__init__(
            dim=dim,
            num_heads=num_heads,
            norm_groups=norm_groups,
        )

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0, "Channels must be divisible by number of heads"

        self.norm = nn.GroupNorm(norm_groups, dim)

        self.qkv = nn.Conv1d(dim, dim * 3, kernel_size=1)
        self.proj_out = nn.Conv1d(dim, dim, kernel_size=1)

    def forward(self, x):
        b, c, h, w = x.shape
        x = self.norm(x)
        x = rearrange(x, 'b c h w -> b c (h w)')    # flatten spatial dims [b, c, n]

        # QKV projection
        qkv = self.qkv(x).chunk(3, dim=1)           # 3 x [b, heads*dim, n]

        for t in qkv:
            assert t.shape[1] % self.num_heads == 0, f"Incompatible head split: {t.shape[1]} not divisible by {self.num_heads}"
        q, k, v = map(
            lambda t: rearrange(t, 'b (h d) n -> b h d n', h=self.num_heads),
            qkv
        )

        # Attention
        scale = self.head_dim ** -0.5
        attn = torch.softmax(torch.einsum('b h d i, b h d j -> b h i j', q, k) * scale, dim=-1)
        out = torch.einsum('b h j k, b h d k ->  b h d j', attn, v)

        out = rearrange(out, 'b h d n -> b (h d) n')
        out = self.proj_out(out)

        x = x + out     # Residule connection
        return rearrange(x, 'b c (h w) -> b c h w', h=h, w=w)
    




