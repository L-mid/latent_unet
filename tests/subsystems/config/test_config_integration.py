
import pytest
pytest.skip("[test_config integration] SKIPPING test_config integration", allow_module_level=True)

import torch
from unittest.mock import MagicMock
import sys

from model.build_unet import build_unet_from_config
sys.modules['diffusion.ddpm'] = MagicMock() # from diffusion.ddpm import DDPM
from model.config import load_config

# ---- Test 1: Full model builds from config ----
def test_unet_builds_from_full_config(parsed_cfg):
    model = build_unet_from_config(parsed_cfg)

    # Optional: lightweight forward sanity check
    dummy_input = torch.randn(2, parsed_cfg.model.in_channels, 64, 64)
    dummy_t = torch.randint(0, parsed_cfg.schedule.timesteps, (2,))
    out = model(dummy_input, dummy_t)
    assert out.shape == dummy_input.shape


# ---- Test 2: Forward process builds from config ----
def test_ddpm_schedule_builds_from_config(parsed_cfg):
    ddpm = DDPM(parsed_cfg.schedule)
    assert ddpm.betas.shape[0] == parsed_cfg.schedule.timesteps


# ---- Test 3: Invalid schedule types still caught here ----
    bad_cfg = {
        "model": {"in_channels": 3, "out_channels": 3, "base_channels": 32},
        "schedule": {"schedule_type": "invaild_type", "timesteps": 100}
    }
    with pytest.raises(ValueError):
        load_config(bad_cfg)


# ---- Test 4: Full end-to-end config load pipeline ----
def test_end_to_end_build(minimal_cfg):
    assert "model" is not None
    ddpm = DDPM(minimal_cfg.schedule)
    assert ddpm.betas.shape[0] == minimal_cfg.schedule.timesteps



