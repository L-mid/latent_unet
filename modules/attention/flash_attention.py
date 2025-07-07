
import torch
import torch.nn as nn
import torch.nn.functional as F
from .registry import register_attention
from .base_attention import BaseAttention

@register_attention("flash")
class FlashAttention(BaseAttention):
    """
    FlashAttention-like block supporting three backend modes:
        - 'auto': Use PyTorch's flash attention if available, else fallback
        - 'flash_only': Require flash attention, raise if not available
        - 'fallback_only': Always use softmax attention
    """

    def __init__(
        self, 
        dim: int,
        num_heads: int = 8, 
        norm_groups: int = None,    # Here but not impimented as cfg controllable
        dim_head: int = None,       
        start_layer: int = 0,
        window_size: int = 0,   
        backend: str = "auto",  # "auto" | "flash_only" | "fallback_only"
        **kwargs
    ):
        super().__init__(
            dim=dim, 
            num_heads=num_heads,
            backend=backend
        )

        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.backend = backend

        self.norm = nn.GroupNorm(8, dim)
        self.qkv_proj = nn.Conv2d(dim, dim * 3, kernel_size=1)
        self.out_proj = nn.Conv2d(dim, dim, kernel_size=1)


    def forward(self, x):
        B, C, H, W = x.shape
        x_norm = self.norm(x)
        qkv = self.qkv_proj(x_norm)
        q, k, v = torch.chunk(qkv, 3, dim=1)

        # Project Q, K, V
        q = q.view(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)
        k = k.view(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)
        v = v.view(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)


        # Determine flash eligibility

        flash_available = (
            hasattr(F, "scaled_dot_product_attention")
            and q.device.type == "cuda"
        )

        # Mode logic
        if self.backend == "flash_only":
            if not flash_available:
                raise RuntimeError("FlashAttention requires CUDA and PyTorch 2.0+")
            out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        
        elif self.backend == "fallback_only":
            out = self._soft_max_attention(q, k, v)
            
        elif self.backend == "auto":
            try:
                if flash_available:
                    out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
                else:
                    raise RuntimeError("Falling back to softmax attention")
            except Exception:
                out = self._soft_max_attention(q, k, v)

        else: raise ValueError(f"Invalid backend mode: {self.backend}")


        # Reshape back: (B, heads, HW, head_dim) -> (B, C, H, W)
        out = out.permute(0, 1, 3, 2).reshape(B, C, H, W)
        return self.out_proj(out) + x   # Residual connection

    def _soft_max_attention(self, q, k, v):
        # Fallback to manual softmax attention
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            attn = torch.softmax(attn_scores, dim=-1)
            return torch.matmul(attn, v)

        

