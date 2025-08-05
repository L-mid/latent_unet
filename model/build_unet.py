
from model.unet import UNet
from modules.time_embedding import get_time_embedding
from modules.residual_block import get_resblock
from modules.down_block import DownBlock
from modules.mid_block import MidBlock
from modules.up_block import UpBlock
from modules.final_head import FinalHead
from modules.attention.registry import get_attention

from modules import attention


def build_unet_from_config(cfg):
    # Time Embedding
    time_embedding = get_time_embedding(cfg.time_embedding)

    # Down/Up Sampling Channels
    base = cfg.model.base_channels      # e.g. 128
    ch_mults = cfg.model.channel_multipliers
    in_channels = [base * m for m in ch_mults[:-1]]     # -> [128, 256, 512]
    out_channels = [base * m for m in ch_mults[1:]]         # -> [256, 512, 1024]

    # Mid Block
    mid_block = MidBlock(
    dim=out_channels[-1],
    time_emb_dim=cfg.time_embedding.params.dim,
    midblock_cfg=cfg.midblock,
    resblock_cfg=cfg.resblock,
    attention_cfg=cfg.attention
    )

    # Down Blocks
    downs = []

    downs.extend(
        DownBlock(
            in_ch=cfg.model.in_channels,
            out_ch=base * ch_mults[0],
         time_emb_dim=cfg.time_embedding.params.dim,
            num_layers=cfg.updown.num_layers,
            debug_enabled=cfg.debug.enabled,
            resblock_cfg=cfg.resblock,
            attention_cfg=cfg.attention, #
            use_attention=(i >= cfg.attention.params.start_layer),
        )
        for i in range(0)
    )

    downs.extend([
        DownBlock(
            in_ch=in_c,
            out_ch=out_c,
            time_emb_dim=cfg.time_embedding.params.dim,
            num_layers=cfg.updown.num_layers,
            debug_enabled=cfg.debug.enabled,
            resblock_cfg=cfg.resblock,
            attention_cfg=cfg.attention, #
            use_attention=(i >= cfg.attention.params.start_layer),
        )
        for i, (in_c, out_c) in enumerate(zip(in_channels, out_channels))
    ])


    ups = []
    for i in range(len(in_channels)):   # len(in_channels) == number of UpBlocks
        upsample_in_ch = out_channels[-(i+1)]
        out_ch = in_channels[-(i+1)]

        expect_skip = True
        skip_ch = out_channels[-(i+1)] if expect_skip else 0
        cat_in_ch = upsample_in_ch + skip_ch   # done in up_block

        ups.append(
            UpBlock(
                in_ch=upsample_in_ch,    # Due to skip connections
                out_ch=out_ch,
                time_emb_dim=cfg.time_embedding.params.dim,
                expect_skip=cfg.updown.expect_skip,     # set to True
                skip_channels=skip_ch,
                num_layers=cfg.updown.num_layers,
                debug_enabled=cfg.debug.enabled,       
                resblock_cfg=cfg.resblock,
                attention_cfg=cfg.attention, #
                use_attention=(i >= cfg.attention.params.start_layer)
            )
        )

    
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
   

