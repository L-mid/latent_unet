
import sys
import os
import pytest
import importlib


# Ensure root project directory is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
importlib.invalidate_caches()

# WANDB
os.environ["WANDB_SILENT"] = "true"        # Suppress W&B terminal output
os.environ["WANDB_MODE"] = "disabled"       # Disables W&B entirely for tests (often better for tests)

# TENSORFLOW
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # 0=all, 1=INFO off, 2=+WARN, 3=+ERROR
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")  # optional; avoids extra logs on some builds


def pytest_addoption(parser):
    parser.addoption(
        "--exp",
        action="store_true",
        help="Run experimental tests (skipped by default)."
    )

def pytest_collection_modifyitems(config, items):
    run_exp = config.getoption("--exp")
    for item in items:
        if "/experimental/" in str(item.fspath).replace("\\", "/"):
            item.add_marker(pytest.mark.experimental)
            if not run_exp:
                item.add_marker(pytest.mark.skip(reason="Experiemtnal tests are skipped by default. Use --exp."))


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

import os, tqdm, pytest
@pytest.fixture(autouse=True, scope="session")
def _kill_tqdm_moniter():
    os.environ["TQDM_DISABLE"] = "1"    # no bars in CI
    tqdm.tqdm.monitor_interval = 0      # disabled moniter thread
    yield


@pytest.fixture
def fp():
    from utils.failure_injection_utils.failpoints import enable_failpoints
    with enable_failpoints() as fp:
        yield fp


