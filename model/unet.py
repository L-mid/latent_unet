
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, in_channels, base_channels, time_embedding, downs, mid, ups, final_head):
        super().__init__()

    def forward(self, x, time_embedding):
        return x