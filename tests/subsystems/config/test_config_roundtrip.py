
import pytest
import os
import tempfile
from omegaconf import OmegaConf
from helpers.test_utils import controlled_test

CATEGORY = "subsystems"
MODULE = "config"

@pytest.mark.parametrize(
    "cfg_obj", [
        ("unet_config"), 
        ("test_config"), 
        ("mock_data_config")
    ],
    indirect=["cfg_obj"]
)
@controlled_test(CATEGORY, MODULE)
def test_yaml_roundtrip(cfg_obj, test_config):
    cfg = cfg_obj
    # Save to temporary file
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "saved_config.yaml")
        OmegaConf.save(cfg, path)

        # Reload
        reloaded_cfg = OmegaConf.load(path)

        # Assert indentical structure
        assert OmegaConf.to_container(cfg, resolve=True) == OmegaConf.to_container(reloaded_cfg, resolve=True)







