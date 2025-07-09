
from model.unet import UNet
from modules.time_embedding import get_time_embedding
from modules.residual_block import get_resblock
from modules.down_block import DownBlock
from modules.mid_block import MidBlock
from modules.up_block import UpBlock
from modules.final_head import FinalHead
from modules.attention.registry import get_attention

def build_unet_from_config(cfg):
    # Time Embedding
    time_embedding = get_time_embedding(cfg.time_embedding)

    # Down/Up Sampling Channels
    base = cfg.model.base_channels
    ch_mults = cfg.model.channel_multipliers
    in_channels = [base * m for m in ch_mults[:-1]]
    out_channels = [base * m for m in ch_mults]

    # Mid Block
    mid_block = MidBlock(
    dim=out_channels[-1],
    time_embed_dim=cfg.time_embedding.params.dim,
    resblock=cfg.resblock,
    attention=cfg.attention
    )

    # Down Blocks
    downs = [
        DownBlock(
            in_ch=in_c,
            out_ch=out_c,
            time_embed_dim=cfg.time_embedding.params.dim,
            resblock_cfg=cfg.resblock,
            attention_cfg=cfg.attention, #
            apply_attention=(i >= cfg.attention.params.start_layer)
        )
        for i, (in_c, out_c) in enumerate(zip([base] + in_channels, out_channels))
    ]

    # Up Blocks
    ups = [
        UpBlock(
            in_ch=out_c * 2,    # Due to skip connections
            out_ch=in_c,
            time_embed_dim=cfg.time_embedding.params.dim,
            resblock_cfg=cfg.resblock,
            attention_cfg=cfg.attention,
            apply_attention=(i >= cfg.attention.params.start_layer)
        )
        for i, (in_c, out_c) in enumerate(zip(reversed(in_channels), reversed(out_channels)))
    ]
    
    # Final Projection Layer
    final_head = FinalHead(base, cfg.final_head.out_channels)

    return UNet(
        in_channels=cfg.model.in_channels,
        base_channels=base,
        time_embedding=time_embedding,
        downs=downs,
        mid=mid_block,
        ups=ups,
        final_head=final_head
    )
   

