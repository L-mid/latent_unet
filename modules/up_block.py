
import torch
import torch.nn as nn
from modules.residual_block import get_resblock
from modules.attention.registry import get_attention

class UpBlock(nn.Module):
    def __init__(
        self, 
        in_ch, 
        out_ch, 
        time_emb_dim, 
        num_layers=1,
        resblock_cfg=None, 
        attention_cfg=None, 
        use_attention=False
    ):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
        self.resblocks = nn.ModuleList([
            get_resblock(resblock_cfg, out_ch * 2 if i == 0 else out_ch, out_ch, time_emb_dim,)
            for i in range(num_layers)    
        ])
        self.attn = get_attention(attention_cfg, out_ch) if use_attention else None

    def forward(self, x, skip, t_emb):
        x = self.upsample(x) 
        x = torch.cat([x, skip], dim=1)
        for block in self.resblocks:
            x = block(x, t_emb)
        if self.attn:
            x = self.attn(x)
        return x
    



