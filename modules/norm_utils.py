
import torch.nn as nn

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels)

    def forward(self, x):
        # x: [B, C, H, W] -> [B, H, W, C]
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2)


def get_norm_layer(norm_type: str, num_channels: int):
    if norm_type == "group":
        return nn.GroupNorm(8, num_channels)
    elif norm_type == "batch":
        return nn.BatchNorm2d(num_channels)
    elif norm_type == "layer":
        return LayerNorm2d([num_channels])   # usually not recommended for CCNs.
    else:
        raise ValueError(f"Unknown norm type: {norm_type}")