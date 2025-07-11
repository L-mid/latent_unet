
import torch.nn as nn

class FinalHead(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        if out_ch <= 0:
            raise ValueError(f"FinalHead: out_ch must be > 0, got {out_ch}")
        if in_ch <= 0:
            raise ValueError(f"FinalHead: in_ch must be > 0, got {in_ch}")
        
        self.block = nn.Sequential(
            nn.GroupNorm(8, in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return self.block(x)
    
