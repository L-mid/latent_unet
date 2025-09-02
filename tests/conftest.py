
import sys
import os
import pytest
import importlib

# Ensure root project directory is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
importlib.invalidate_caches()

os.environ["WANDB_SILENT"] = "true"        # Suppress W&B terminal output
# os.environ["WANDB_MODE"] = "disabled"       # Disables W&B entirely for tests (often better for tests)

from model.config import load_config


# Load U-Net config 
@pytest.fixture(scope='session')
def unet_config():
    return load_config("configs/unet_config.yaml")

# Load testing config
@pytest.fixture(scope='session')
def test_config():
    return load_config("configs/test_config.yaml")

# Load mock data config
@pytest.fixture(scope='session')
def mock_data_config():
    return load_config("configs/mock_data_config.yaml")


@pytest.fixture
def cfg_obj(request):
    # request.param will be the *string* like "unet_config"
    return request.getfixturevalue(request.param)
