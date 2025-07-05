
import torch.nn as nn

class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_embed_dim, resblock_cfg, attention_cfg, apply_attention=False):
        super().__init__()
        self.layer = nn.Identity()

    def forward(self, x):
        return self.layer
    