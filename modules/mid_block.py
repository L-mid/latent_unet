
import torch
import torch.nn as nn
from modules.residual_block import get_resblock
from modules.attention.registry import get_attention

class MidBlock(nn.Module):
    def __init__(
        self, 
        dim: int, 
        time_emb_dim: int, 
        midblock_cfg=None, 
        resblock_cfg=None, 
        attention_cfg=None,
    ):
        super().__init__()

        # Build first ResBlock (no attention)
        self.res1 = get_resblock(
            cfg=resblock_cfg,
            in_ch=dim,
            out_ch=dim,
            time_dim=time_emb_dim,
        )

        # Optional Attention
        if midblock_cfg.use_attention:
            self.attn = get_attention(
                dim=dim,
                cfg=attention_cfg,
            )
        else:
            self.attn = None

        # Build second ResBlock (no attn)
        self.res2 = get_resblock(
            cfg=resblock_cfg,
            in_ch=dim,
            out_ch=dim,
            time_dim=time_emb_dim,
        )

    def forward(self, x, t_emb):
        x = self.res1(x, t_emb)
        if self.attn:
            x = self.attn(x)
        x = self.res2(x, t_emb)
        return x
    
    

        






