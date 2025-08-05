
import torch.nn as nn
from modules.residual_block import get_resblock
from modules.attention.registry import get_attention
from typing import Dict

class DownBlock(nn.Module):
    def __init__(
        self, 
        in_ch: int, 
        out_ch: int, 
        time_emb_dim: int,
        num_layers: int = 1,
        debug_enabled=False,
        use_attention=False,
        resblock_cfg=None, 
        attention_cfg=None
    ):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch

        self.debug_enabled = debug_enabled

        self.resblocks = nn.ModuleList([
            get_resblock(resblock_cfg, in_ch if i == 0 else out_ch, out_ch, time_emb_dim)
            for i in range(num_layers)
        ])
        self.attn = get_attention(out_ch, attention_cfg) if use_attention else None  
        self.downsample = nn.Conv2d(out_ch, out_ch, kernel_size=4, stride=2, padding=1)


    def forward(self, x, t_emb):
        if self.debug_enabled:
            print(f"[DownBlock] input: {x.shape}")

        for block in self.resblocks:
            x = block(x, t_emb)
        if self.attn:
            x = self.attn(x)

        skip = x
        if self.debug_enabled:
            print(f"[DownBlock] skip: {skip.shape}")

        x = self.downsample(x)

        if self.debug_enabled:
            print(f"[DownBlock] after downsamples: {x.shape}")

        return x, skip
        

