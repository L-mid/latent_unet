
import torch
import torch.nn as nn
from modules.residual_block import get_resblock
from modules.attention.registry import get_attention

class MidBlock(nn.Module):
    def __init__(self, dim, time_embed_dim, midblock, resblock, attention):
        super().__init__()

        # Build first ResBlock (no attention)
        self.res1 = get_resblock(
            cfg=resblock,
            in_ch=dim,
            out_ch=dim,
            time_dim=time_embed_dim,
        )

        # Optional Attention
        if midblock.use_attention:
            self.attn = get_attention(
                cfg=attention,
            )
        else:
            self.attn = None

        # Build second ResBlock (no attn)
        self.res2 = get_resblock(
            cfg=resblock,
            in_ch=dim,
            out_ch=dim,
            time_dim=time_embed_dim,
        )

    def forward(self, x, t_emb):
        x = self.res1(x, t_emb)
        if self.attn:
            x = self.attn(x)
        x = self.res2(x, t_emb)
        return x
    
    

        






