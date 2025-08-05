
import torch
import torch.nn as nn
from .base_attention import BaseAttention
from helpers.test_utils import controlled_test
from .registry import register_attention

@register_attention("window") 
class WindowAttention(BaseAttention):
    def __init__(
        self, 
        dim, 
        num_heads = 4, 
        norm_groups = None, 
        dim_head = None, 
        start_layer = 0, 
        window_size = 4, 
        backend = None, 
        **kwargs
    ):
        super().__init__(
        dim=dim, 
        num_heads=num_heads,   
        window_size=window_size, 
    )
        
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):           # x: (B, C, H W)
        B, C, H, W = x.shape
        assert H % self.window_size == 0 and W % self.window_size == 0

        # Rearrange into (B*windows, N, C)
        x = x.view(B, C, H//self.window_size, self.window_size, W//self.window_size, self.window_size)
        x = x.permute(0, 2, 4, 3, 5, 1).reshape(-1, self.window_size**2, C)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(x.shape[0], -1, self.num_heads, C // self.num_heads).transpose(1, 2), qkv)

        attn = torch.softmax((q @ k.transpose(-2, -1)) * self.scale, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(x.shape[0], -1, C)
        out = self.proj(out)

        # Restore
        out = out.view(B, H//self.window_size, W//self.window_size, self.window_size, self.window_size, C)
        out = out.permute(0, 5, 1, 3, 2, 4).reshape(B, C, H, W)
        return out
        

        
