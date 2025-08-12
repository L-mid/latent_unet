
import torch
import pytest
from trainer.losses import LOSS_REGISTRY, DiffusionLoss

class DummyConfig:
    class Training:
        num_epochs = 1
        steps_per_epoch = 100
    training = Training()


# ---------------------------
# Fixtures
# ---------------------------

@pytest.fixture
def dummy_inputs():
    pred = torch.randn(4, 3, 64, 64)
    target = torch.randn(4, 3, 64, 64)
    return pred, target

@pytest.fixture
def step():
    return 50   # Mid-training


# -----------------------------
# Loss Registry Tests
# -----------------------------

def test_loss_registry_includes_bacis_losses():
    assert "mse" in LOSS_REGISTRY
    assert "huber" in LOSS_REGISTRY

def test_loss_registry_all_callable():
    for name, fn in LOSS_REGISTRY.items():
        assert callable(fn), f"{name} is not callable"


# -------------------------------
# Basic Loss Tests 
# -------------------------------

def test_mse_loss(dummy_inputs):
    pred, target = dummy_inputs
    loss = LOSS_REGISTRY["mse"](pred, target)
    assert loss > 0     # not supported between instances of 'tuple' and 'int'?
    assert loss.dim() == 0


# -------------------------------
# Optional Losses
# -------------------------------

def test_lpips_loss(dummy_inputs):
    if "lpips" not in LOSS_REGISTRY:
        pytest.skip("LPIPS not installed")
    pred, target = dummy_inputs
    loss = LOSS_REGISTRY["lpips"](pred, target)
    assert loss > 0
    assert loss.dim() == 0

def test_vgg_loss(dummy_inputs):
    if "vgg" not in LOSS_REGISTRY:
        pytest.skip("VGG not available")
    pred, target = dummy_inputs
    loss = LOSS_REGISTRY["vgg"](pred, target)
    assert loss > 0
    assert loss.dim() == 0



# ---------------------------------
# Weighting / Schedule Logic
# ---------------------------------

def test_weighted_loss_combination(dummy_inputs, step):
    pred, target = dummy_inputs

    config = DummyConfig()
    config.losses = [
        {"type": "mse", "weight": 1.0},
        {"type": "huber", "start_weight": 0.0, "end_weight": 2.0, "weight": 0.5, "schedule": "linear"},
    ]
    loss_fn = DiffusionLoss(config)
    out = loss_fn(pred, target, step=step)

    assert "total_loss" in out
    assert out["mse_loss"].dim() == 0
    assert out["huber_loss"].dim() == 0
    assert isinstance(out["huber_weight"], float)


# -------------------------------
# Visual Mode 
# -------------------------------

def visualize_output(dummy_inputs, step):
    pred, target = dummy_inputs
    config = DummyConfig()
    config.losses = [{"type": "mse", "weight": 1.0}]

    loss_fn = DiffusionLoss(config)
    out = loss_fn(pred, target, step=step, visualize=True)

    assert "output_sample" in out
    assert "target_sample" in out
    assert out["output_sample"].shape[0] == 4



