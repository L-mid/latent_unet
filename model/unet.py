
import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(
        self, 
        in_channels: int,
        base_channels: int,
        time_embedding: nn.Module,
        downs: nn.ModuleList,
        mid: nn.Module,
        ups: nn.ModuleList,
        final_head: nn.Module
):
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.time_embedding = time_embedding
        self.downs = nn.ModuleList(downs) # fix for normalization
        self.mid = mid
        self.ups = nn.ModuleList(ups)
        self.final_head = final_head

        self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # Compute time embedding
        temb = self.time_embedding(t)

        # Initial projection
        x = self.init_conv(x).to(x.device) # fixed here

        # Down path with skip connections
        skips = []
        for down in self.downs:
            x, skip = down(x, temb)
            skips.append(skip)

        # Mid block
        x = self.mid(x, temb)

        # Up path with skip connections
        for up, skip in zip(self.ups, reversed(skips)):  
            x = up(x, skip, temb)

        # Final output ptojection
        out = self.final_head(x)
        return out
    


    