
import torch
import pytest
from diffusion.forward_process import ForwardProcess
from diffusion.schedule import get_diffusion_schedule

BATCH = 4
CHANNELS = 3
HEIGHT = WIDTH = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@pytest.mark.parametrize("schedule", ["linear", "cosine"])
def test_beta_schedule_shapes(schedule):
    fp = ForwardProcess(schedule=schedule, timesteps=1000)

    schedule = get_diffusion_schedule("linear", 1000)
    print(schedule)

    assert fp.betas.shape[0] == 1000
    assert fp.alphas_cumprod.shape == (1000,)
    assert torch.all(fp.alphas_cumprod > 0) and torch.all(fp.alphas_cumprod <= 1)

def test_q_sample_shape_consistency():
    fp = ForwardProcess(schedule="linear", timesteps=1000)
    x_start = torch.randn(BATCH, CHANNELS, HEIGHT, WIDTH).to(DEVICE)
    t = torch.randint(0, 1000, (BATCH,), device=DEVICE)
    x_noised, noise = fp.q_sample(x_start, t, return_noise=True)
    assert x_noised.shape == x_start.shape
    assert noise.shape == x_start.shape

def test_noise_determinisum_seeded():
    fp = ForwardProcess(schedule="linear", timesteps=1000)
    x_start = torch.randn(1, 3, 32, 32)
    t = torch.tensor([100])
    torch.manual_seed(42)
    x1 = fp.q_sample(x_start, t)
    torch.manual_seed(42)
    x2 = fp.q_sample(x_start, t)
    assert torch.allclose(x1, x2), "Noised sampled aren't deterministic with fixed seed"

def test_alpha_product_monotonicity():
    fp = ForwardProcess(schedule="linear", timesteps=1000)
    assert torch.all(fp.alphas_cumprod[1:] < fp.alphas_cumprod[:-1]).item(), \
    "Alphas should decrease over time"


def test_returned_noise_stats():
    fp = ForwardProcess(schedule="linear", timesteps=1000)
    x_start = torch.randn(BATCH, 3, 32, 32)
    t = torch.randint(0, 1000, (BATCH,))
    x_t, noise = fp.q_sample(x_start, t, return_noise=True)

    assert noise.mean().abs().item() < 0.1, "Noise mean to high"
    assert 0.8 < noise.std().item() < 1.2, "Noise std suspicious"

@pytest.mark.skip(reason="Visualizer not yet implimented")
def test_visual_debug_compatibitly():
    # Ensure this is hook compatible for visualizations.
    #from utils.visualizer import visualize_noising_process
    fp = ForwardProcess(schedule="cosine", timesteps=1000)
    x_start = torch.randn(1, 3, 32, 32)
    #visualize_noising_process(fp, x_start, save_path="tmp/test_noising.gif") 