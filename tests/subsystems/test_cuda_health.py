
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

    t = torch.randint(low=0, high=1000, size=(dummy_input.size(0),), dtype=torch.long)
    dummy_input = torch.randn(2, unet_config.model.in_channels, 64, 64, device="cuda")
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        with torch.no_grad():
            out = model(dummy_input, t)

    assert torch.isfinite(out).all(), "AMP output contains NaN/Inf"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_unet_output_consistency_vs_cpu(unet_config):
    # Compare CPU and CUDA outputs (sanity, not exact match).
    model_cpu = build_unet_from_config(unet_config).cpu().eval()
    model_gpu = build_unet_from_config(unet_config).cuda().eval()

    input_cpu = torch.randn(1, unet_config.model.in_channels, 64, 64)
    t_cpu = torch.randint(low=0, high=1000, size=(input_cpu.size(0),), dtype=torch.long)

    input_gpu = input_cpu.clone().cuda() # uh
    t_gpu = torch.randint(low=0, high=1000, size=(input_gpu.size(0),), dtype=torch.long)

    with torch.no_grad():
        out_cpu = model_cpu(input_cpu, t_cpu)
        out_gpu = model_gpu(input_gpu, t_gpu).cpu()

    torch.testing.assert_close(out_cpu.float(), out_gpu.float(), rtol=1e-3, atol=1e-3)

