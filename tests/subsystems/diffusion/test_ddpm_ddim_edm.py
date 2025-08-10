
import torch
import pytest

from diffusion.ddpm import DDPM
from diffusion.ddim import ddim_sample
from diffusion.edm import edm_preprocess, edm_sample, get_edm_schedule

from model.build_unet import build_unet_from_config
from model.config import load_config


@pytest.fixture(scope="module")
def dummy_inputs():
    batch_size = 2
    image_size = 32
    channels = 3
    device = "cuda" if torch.cuda.is_available() else "cpu"

    x = torch.randn(batch_size, channels, image_size, image_size).to(device)
    t = torch.randint(0, 1000, (batch_size,), device=device)
    return x, t, device

@pytest.fixture(scope="module")
def model_and_config(unet_config):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = build_unet_from_config(unet_config).to(device) 
    model.eval()
    return model, unet_config 


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Samplers take too long on cpu")      # set to cuda only if curious otherwise leave off
@pytest.mark.parametrize("sampler_type", ["ddpm", "edm", "ddim"])
def test_sampler_forward_pass(dummy_inputs, model_and_config, sampler_type):
    x, t, device = dummy_inputs
    model, cfg = model_and_config

    if sampler_type == "ddpm":
        sampler = DDPM(model=model, device=device)
    elif sampler_type == "ddim":
        sampler = ddim_sample(model=model, shape=(2, 3, 32, 32), num_steps=cfg.sampler.num_steps, device=device)
    elif sampler_type == "edm":
        sampler = edm_sample(model=model, shape=(2, 3, 32, 32), cfg=cfg.sampler, device=device)
    else:
        raise ValueError("Unknown sampler type")

    with torch.no_grad():
        if sampler_type == "ddpm":
            result = sampler.sample(shape=(2, 3, 32, 32))
        else:
            result = sampler

    assert result.shape == (2, 3, 32, 32), f"{sampler_type} output shape mismatch"
    assert result.isfinite().all(), f"{sampler_type} output contains NaNs"

    print({sampler_type}, "passed.")



@pytest.mark.skipif(not torch.cuda.is_available(), reason="Samplers take too long on cpu")      # set to cuda only if curious otherwise leave off
def test_ddim_consistency(dummy_inputs, model_and_config):
    x, t, device = dummy_inputs
    model, cfg = model_and_config

    out1 = ddim_sample(model=model, shape=(2, 3, 32, 32), num_steps=cfg.sampler.num_steps, device=device)
    out2 = ddim_sample(model=model, shape=(2, 3, 32, 32), num_steps=cfg.sampler.num_steps, device=device)

    assert (out1 != out2).any(), "DDIM sampling is not stochasic by default. Unexpected repeat?"


@pytest.mark.parametrize("sampler_class", [DDPM, ddim_sample, edm_sample])
def test_sampler_instantiation(dummy_inputs, sampler_class, model_and_config):
    model, cfg = model_and_config
    _, _, device = dummy_inputs
    
    if sampler_class == DDPM:
        sampler = sampler_class(model=model, device=device)
    elif sampler_class == ddim_sample:
        sampler = ddim_sample(model=model, shape=(2, 3, 32, 32), num_steps=1, device=device)
    elif sampler_class == edm_sample:
        sampler = edm_sample(model=model, shape=(2, 3, 32, 32), cfg=cfg, device=device)

    assert sampler is not None, f"{sampler_class.__name__} failed to instantiate"






