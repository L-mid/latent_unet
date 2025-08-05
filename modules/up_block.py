
import torch
import torch.nn as nn
from modules.residual_block import get_resblock
from modules.attention.registry import get_attention

class UpBlock(nn.Module):
    def __init__(
        self, 
        in_ch: int, 
        out_ch: int, 
        time_emb_dim: int, 
        skip_channels: int,
        num_layers: int = 1,
        expect_skip=True,
        debug_enabled: bool=False,
        resblock_cfg=None, 
        attention_cfg=None, 
        use_attention=False
    ):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.skip_channels = skip_channels


        self.expect_skip=expect_skip
        self.debug_enabled = debug_enabled

        upsample_out_ch = out_ch

        self.cat_in_ch = upsample_out_ch + skip_channels  # e.g., 64 + 128 = 192 for example


        self.upsample = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
        self.resblocks = nn.ModuleList([
            get_resblock(resblock_cfg, self.cat_in_ch if i == 0 else out_ch, out_ch, time_emb_dim,)
            for i in range(num_layers)    # i == 0 could be an issue
        ])
        self.attn = get_attention(out_ch, attention_cfg) if use_attention else None


    def forward(self, x, skip, t_emb):
        x = self.upsample(x) 

        if self.debug_enabled:
            print(f"[UpBlock] x: {x.shape}, skip: {skip.shape if skip is not None else 'None'}")

        if self.expect_skip:
            assert skip is not None, f"Expected skip connection but got None."
            assert x.shape[2:] == skip.shape[2:], f"Upsampled x shape {x.shape} does not match skip shape {skip.shape}"
            
            x = torch.cat([x, skip], dim=1)

            if self.debug_enabled:
                print(f"[UpBlock] after concat: {x.shape}, expected: {self.cat_in_ch}")
        else:
            assert skip is None, "Did not expect skip connections but got one"
            if self.debug_enabled:
                print("[UpBlock] no skip provided, skipping concat")

        for block in self.resblocks:
            x = block(x, t_emb)
            if self.debug_enabled:
                print(f"[UpBlock] after resblock: {x.shape}")

        if self.attn:
            x = self.attn(x)

        return x
    



