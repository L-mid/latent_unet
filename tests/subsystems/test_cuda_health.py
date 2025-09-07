
import pytest
import torch
from helpers.test_utils import controlled_test  # removed
from model.build_unet import build_unet_from_config

# === NOTES:
"""
UNTESTED.

"""


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_unet_cuda_forward_no_nan(unet_config): # where's unet config from?
    # Basic CUDA smoke test: forward pass runs, no NaNs/Infs.
    model = build_unet_from_config(unet_config).cuda()
    model.eval()

    dummy_input = torch.randn(2, unet_config.model.in_channels, 64, 64, device="cuda")
    t = torch.randint(low=0, high=1000, size=(dummy_input.size(0),), dtype=torch.long)

    with torch.no_grad():
        out = model(dummy_input, t)

    assert torch.isfinite(out).all(), "Non-finite values in output (NaN/Inf)"

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_unet_amp_autocast_forward(unet_config):
    model = build_unet_from_config(unet_config).cuda()
    model.eval()

    dummy_input = torch.randn(2, unet_config.model.in_channels, 64, 64, device="cuda")
    t = torch.randint(low=0, high=1000, size=(dummy_input.size(0),), dtype=torch.long)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        with torch.no_grad():
            out = model(dummy_input, t)

    assert torch.isfinite(out).all(), "AMP output contains NaN/Inf"

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_unet_output_consistency_vs_cpu(unet_config):
    # Compare CPU and CUDA outputs (sanity, not exact match).
    torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    model = build_unet_from_config(unet_config).eval()
    
    x = torch.randn(1, unet_config.model.in_channels, 64, 64).float()
    t = torch.randint(low=0, high=1000, size=(x.size(0),), dtype=torch.long)

    with torch.no_grad():
        y_cpu = model(x.cpu(), t.cpu()).float()
        y_gpu = model.to("cuda")(x.cuda(), t.cuda()).float().cpu()
    torch.testing.assert_close(y_cpu, y_gpu, rtol=1e-3, atol=1e-3)