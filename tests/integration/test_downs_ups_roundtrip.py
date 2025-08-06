
import torch
import pytest
from modules.down_block import DownBlock
from modules.up_block import UpBlock
from omegaconf import OmegaConf

@pytest.mark.parametrize("img_size", [32])     # removed 64 & 128 for speed
@pytest.mark.parametrize("num_layers", [1, 2, 3])
@pytest.mark.parametrize("resblock_cfg", [{"kind": "vanilla"}])
@pytest.mark.parametrize("attention_cfg", [
    OmegaConf.create({"kind": "vanilla", "params": {"num_heads": 4}})
])
def test_downblock_upblock_pair(img_size, num_layers, resblock_cfg, attention_cfg):
    in_ch = 64
    out_ch = 128
    time_emb_dim = 256

    assert (img_size % 2 == 0), "img_size must be divisible by 2 for ConvTranspose2d"

    # Simulate input
    x = torch.randn(1, in_ch, img_size, img_size)
    t_emb = torch.randn(1, time_emb_dim)

    # Creates DownBlock
    down = DownBlock(
        in_ch=in_ch,
        out_ch=out_ch,
        time_emb_dim=time_emb_dim,
        num_layers=num_layers,
        resblock_cfg=resblock_cfg,
        attention_cfg=attention_cfg,
        use_attention=True
    )

    # Forward pass through DownBlock
    x_down, skip = down(x, t_emb)

    # Extract skip channels for UpBlock construction
    skip_channels = skip.shape[1]   


    # Create UpBlock (skip used, so in_ch doubled)
    up = UpBlock(
        in_ch=out_ch,
        out_ch=in_ch,
        time_emb_dim=time_emb_dim,
        skip_channels=skip_channels,
        num_layers=num_layers,
        resblock_cfg=resblock_cfg,
        attention_cfg=attention_cfg,
        use_attention=True,
        debug_enabled=True,
        expect_skip=True,
    )

    # Forward pass through UpBlock
    x_up = up(x_down, skip, t_emb)

    # Check final spatial size
    assert x_up.shape[2:] == (img_size, img_size), "Spatial size mismatch after roundtrip"

    # Check final channel size 
    assert x_up.shape[1] == in_ch, "Final channel mismatch after UpBlock"



