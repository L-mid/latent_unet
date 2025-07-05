
import pytest
from helpers.test_utils import controlled_test
from model.config import config_to_dict
from omegaconf import OmegaConf

CATEGORY = "subsystems"
MODULE = "config"

@pytest.mark.parametrize(
    "cfg_name, cfg_obj", [
        ("unet", "unet_config"),
        ("test_config", "test_config"),
        ("mock_data", "mock_data_config")
    ],
    indirect=["cfg_obj"]
)
@controlled_test(CATEGORY, MODULE)
def test_config_parses_and_has_required_sections(cfg_name, cfg_obj, test_config):
    cfg = cfg_obj

    # Minimal structure checks per config
    if cfg_name == "unet":
        assert "model" in cfg and cfg.model.base_channels > 0
        assert "schedule" in cfg and "schedule" in cfg.schedule

    elif cfg_name == "test_config":
        assert "test_config" in cfg
        assert "unit" in cfg.test_config

    elif cfg_name == "mock_data":
        assert "defaults" in cfg
        assert isinstance(cfg.image_batch.batch_size, int)
    else:
        pytest.fail("Unknown config type")

@pytest.mark.parametrize(
    "cfg_obj", [
        ("unet_config"),
        ("test_config"),
        ("mock_data_config")
    ],
    indirect=["cfg_obj"]
)
@controlled_test(CATEGORY, MODULE)
def test_config_serialization_roundtrip(cfg_obj, test_config):
    cfg = cfg_obj

    # Convert to dict
    as_dict = config_to_dict(cfg)

    # Re-create OmegaCofg object and compare keys
    reloaded = OmegaConf.create(as_dict)
    assert sorted(reloaded.keys()) == sorted(cfg.keys())





