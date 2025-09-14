
import pytest
import torch
from unittest.mock import MagicMock
import sys
from omegaconf import OmegaConf

from model.build_unet import build_unet_from_config
from model.config import load_config
from diffusion.ddpm import DDPM
from helpers.mock_configs import accurate_cfg

cfg = accurate_cfg(); cfg = OmegaConf.create(cfg)


# ---- Test 1: Full model builds from config ----
def test_unet_builds_from_full_config():
    model = build_unet_from_config(cfg)

    # Optional: lightweight forward sanity check
    dummy_input = torch.randn(2, cfg.model.in_channels, 64, 64)
    dummy_t = torch.randint(0, cfg.schedule.timesteps, (2,))
    out = model(dummy_input, dummy_t)
    assert out.shape == dummy_input.shape


# ---- Test 2: Forward process builds from config ----
def test_ddpm_schedule_builds_from_config():
    model = build_unet_from_config(cfg)
    ddpm = DDPM(model, timesteps=cfg.schedule.timesteps)
    assert ddpm.betas.shape[0] == cfg.schedule.timesteps


# ---- Test 3: Invalid schedule types still caught here ----
    bad_cfg = {
        "model": {"in_channels": 3, "out_channels": 3, "base_channels": 32},
        "schedule": {"schedule_type": "invaild_type", "timesteps": 100}
    }
    with pytest.raises(Exception):
        load_config(bad_cfg)


# ---- Test 4: Full end-to-end config load pipeline ----
def test_end_to_end_build():
    model = build_unet_from_config(cfg)
    assert "model" is not None
    ddpm = DDPM(model, cfg.schedule.timesteps)
    assert ddpm.betas.shape[0] == cfg.schedule.timesteps



