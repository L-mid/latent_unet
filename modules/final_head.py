
import torch.nn as nn

class FinalHead(nn.Module):
    def __init__(self, base, out_ch):
        super().__init__()
        self.layer = nn.Identity()

    def forward(self, x):
        return self.layer    