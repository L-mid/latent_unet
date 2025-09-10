
import sys
from unittest.mock import MagicMock
from helpers.test_utils import controlled_test
from model.config import load_config
import pytest
import torch.nn as nn


CATAGORY = "subsystems"
MODULE = "config"

@pytest.fixture()
def cfg():
    return load_config("configs/unet_config.yaml")
  
    
@controlled_test(CATAGORY, MODULE)
def test_build_unet_config_with_mocked_modules(cfg, test_config):

    # Save original sys.modules
    original_sys_modules = sys.modules.copy()


    try:
        sys.modules.pop("model.build_unet", None)

        # Strict mocks for each class/function
        mock_time_embedding = MagicMock(return_value="mocked_time_embed")
        mock_midblock = MagicMock(return_value="mocked_midblock")
        mock_downblock = MagicMock(return_value="mocked_downblock")
        mock_upblock = MagicMock(return_value="mocked_upblock")
        mock_final_head = MagicMock(return_value="mocked_final_head")
        mock_unet = MagicMock(return_value="mocked_unet") 

        DummyModule = type("DummyModule", (nn.Module,), {"forward": lambda self, *a, **kw: a[0] if a else None})

        # Inject mocks
        sys.modules.update({
            'model.unet': MagicMock(UNet=mock_unet),
            'modules.time_embedding': MagicMock(get_time_embedding=mock_time_embedding),
            'modules.down_block': MagicMock(DownBlock=mock_downblock),
            'modules.up_block': MagicMock(UpBlock=mock_upblock),
            'modules.mid_block': MagicMock(MidBlock=mock_midblock),
            'modules.final_head': MagicMock(FinalHead=mock_final_head),
            'modules.residual_block': MagicMock(get_resblock=MagicMock(return_value=DummyModule())),
            'modules.attention.resgistry': MagicMock(get_attention=MagicMock(return_value=DummyModule()))
        })

        # Import after mocks are injected
        from model.build_unet import build_unet_from_config 

        model = build_unet_from_config(cfg)
        assert model == "mocked_unet"

        # --- Time Embedding ---
        mock_time_embedding.assert_called_once_with(cfg.time_embedding)

        # --- MidBlock ---
        mock_midblock.assert_called_once()
        mid_args = mock_midblock.call_args.kwargs
        assert mid_args == {
            "dim": cfg.model.base_channels * cfg.model.channel_multipliers[-1],
            "time_emb_dim": cfg.time_embedding.params.dim,
            "midblock_cfg": cfg.midblock,
            "resblock_cfg": cfg.resblock,
            "attention_cfg": cfg.attention,
            }
        
        # --- DownBlocks ---
        expected_down_blocks = len(cfg.model.channel_multipliers) - 1
        assert mock_downblock.call_count == expected_down_blocks

        for i, call in enumerate(mock_downblock.call_args_list):
            args = call.kwargs
            expected_keys = {
                "in_ch", "out_ch", "time_emb_dim", "num_layers", "debug_enabled", "resblock_cfg", "attention_cfg", "use_attention"
            }
            assert set(args.keys()) == expected_keys

            # Check use_attention logic
            assert args["use_attention"] == (i >= cfg.attention.params.start_layer)

            # --- UpBlocks ---
            assert mock_upblock.call_count == expected_down_blocks
            for i, call in enumerate(mock_upblock.call_args_list):
                args = call.kwargs
                expected_keys = {
                    "in_ch", "out_ch", "time_emb_dim", "num_layers", "debug_enabled", 
                    "resblock_cfg", "attention_cfg", "skip_channels", "use_attention"
                } 
                assert set(args.keys()) == expected_keys

                assert args["use_attention"] == (i >= cfg.attention.params.start_layer)

            # --- FinalHead ---
            mock_final_head.assert_called_once_with(cfg.model.base_channels, cfg.final_head.out_channels)

            # --- UNet ---
            mock_unet.assert_called_once()
            unet_args = mock_unet.call_args.kwargs
            assert unet_args == {
                "in_channels": cfg.model.in_channels,
                "base_channels": cfg.model.base_channels,
                "time_embedding": "mocked_time_embed",
                "downs": [mock_downblock.return_value] * expected_down_blocks,
                "mid": "mocked_midblock",
                "ups": [mock_upblock.return_value] * expected_down_blocks,
                "final_head": "mocked_final_head"
            }

    finally:
        # Restore original modules to avoid polluting other tests
        sys.modules.clear()
        sys.modules.update(original_sys_modules)





    