
import sys
from unittest.mock import MagicMock
from helpers.test_utils import controlled_test
from model.config import load_config
from helpers.patch_context import patch_module_attr

import pytest

CATAGORY = "subsystems"
MODULE = "config"

@pytest.fixture()
def cfg():
    return load_config("configs/unet_config.yaml")
  
    
@controlled_test(CATAGORY, MODULE)
def test_build_unet_config_with_mocked_modules(cfg, test_config):

    # Backup:
    original_sys_modules = sys.modules.copy()

    # Strict mocks for each class/function
    mock_time_embedding = MagicMock(return_value="mocked_time_embed")
    mock_midblock = MagicMock(return_value="mocked_midblock")
    mock_downblock = MagicMock(return_value="mocked_downblock")
    mock_upblock = MagicMock(return_value="mocked_upblock")
    mock_final_head = MagicMock(return_value="mocked_final_head")
    mock_unet = MagicMock(return_value="mocked_unet") 

    # Wipe all cached modules that import UNet early
    for mod in ["model.unet", "model.build_unet"]:
        sys.modules.pop(mod, None)

    try:
        with patch_module_attr("modules.time_embedding", "get_time_embedding", mock_time_embedding), \
            patch_module_attr("modules.mid_block", "MidBlock", mock_midblock), \
            patch_module_attr("modules.down_block", "DownBlock", mock_downblock), \
            patch_module_attr("modules.up_block", "UpBlock", mock_upblock), \
            patch_module_attr("modules.final_head", "FinalHead", mock_final_head), \
            patch_module_attr("model.unet", "UNet", mock_unet):

            from model.build_unet import build_unet_from_config

            # Sanity test: config parsing + builder logic runs
            model = build_unet_from_config(cfg)
            assert model == "mocked_unet"

            # ---- Time embedding ----
            mock_time_embedding.assert_called_once_with(cfg.time_embedding)

            # ---- Mid block ---
            mock_midblock.assert_called_once()
            mid_args = mock_midblock.call_args.kwargs
            assert mid_args["dim"] > 0
            assert mid_args["resblock_cfg"] == cfg.resblock
            assert mid_args["attention_cfg"] == cfg.attention

            # ---- Down Blocks ----
            expected_down_blocks = len(cfg.model.channel_multipliers) - 1
            assert mock_downblock.call_count == expected_down_blocks
            for call_args in mock_downblock.call_args_list:
                assert "in_ch" in call_args.kwargs
                assert "out_ch" in call_args.kwargs
                assert "use_attention" in call_args.kwargs

            print("\n[Test Build Unet] --- DownBlock Channel Configurations ---")
            for i, call_args in enumerate(mock_downblock.call_args_list):
                in_ch = call_args.kwargs["in_ch"]
                out_ch = call_args.kwargs["out_ch"]
                print(f"DownBlock {i}: in_ch={in_ch}, out_ch={out_ch}")

            # ---- Up Blocks ----
            assert mock_upblock.call_count == expected_down_blocks 
            for call_args in mock_upblock.call_args_list:
                assert "in_ch" in call_args.kwargs
                assert "out_ch" in call_args.kwargs
                assert "use_attention" in call_args.kwargs

            print("\n[Test Build Unet] --- UpBlock Channel Configureations ---")
            for i, call_args in enumerate(mock_upblock.call_args_list):
                in_ch = call_args.kwargs["in_ch"]
                out_ch =  call_args.kwargs["out_ch"]
                print(f"UpBlock {i}: in_ch={in_ch}, out_ch={out_ch}")    

            # ---- Final Head ----
            mock_final_head.assert_called_once_with(cfg.model.base_channels, cfg.final_head.out_channels)

            # ---- Final UNet ----
            mock_unet.assert_called_once()
            unet_kwargs = mock_unet.call_args.kwargs
            assert unet_kwargs["final_head"] == "mocked_final_head"


    finally:
        for mod in ["model.unet", "model.build_unet"]:
            if mod in original_sys_modules:
                sys.modules[mod] = original_sys_modules[mod]
            else:
                sys.modules.pop(mod, None)



@controlled_test(CATAGORY, MODULE)
def test_build_unet_config_shape(cfg, test_config):

        # Backup:
    original_sys_modules = sys.modules.copy()

    # Strict mocks for each class/function
    mock_downblock = MagicMock(return_value="mocked_downblock")
    mock_upblock = MagicMock(return_value="mocked_upblock")

    try:
        with patch_module_attr("modules.down_block", "DownBlock", mock_downblock), \
            patch_module_attr("modules.up_block", "UpBlock", mock_upblock):

            base = cfg.model.base_channels
            mults = cfg.model.channel_multipliers

            expected_down_channels = list(zip(
                [base] + [base * m for m in mults[:-1]],
                [base * m for m in mults]                                
            ))

            expected_up_channels = list(zip(
                reversed([base * m for m in mults[:-1]]),
                reversed([base * m for m in mults])                        
            ))

            # Assert each DownBlock got the correct channels
            for call, (exp_in, exp_out) in zip(mock_downblock.call_args_list, expected_down_channels):
                assert call.kwargs["in_ch"] == exp_in
                assert call.kwargs["out_ch"] == exp_out


            # Assert each UpBlock got the correct channels
            for call, (exp_in, exp_out) in zip(mock_upblock.call_args_list, expected_up_channels):
                assert call.kwargs["in_ch"] == exp_out * 2
                assert call.kwargs["out_ch"] == exp_in

    finally:
        for mod in ["model.unet", "model.build_unet"]:
            if mod in original_sys_modules:
                sys.modules[mod] = original_sys_modules[mod]
            else:
                sys.modules.pop(mod, None)


    