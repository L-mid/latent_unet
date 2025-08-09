
import pytest 
import torch
from diffusion.schedule import make_beta_schedule

# 1. Validate that all supported types return tensors of correct shape and values
@pytest.mark.parametrize("schedule_type", ["linear", "cosine", "quadratic"])
def test_schedule_output_shape_and_range(schedule_type):
    timesteps = 1000
    betas = make_beta_schedule(schedule_type, timesteps)

    assert isinstance(betas, torch.Tensor)
    assert betas.shape == (timesteps,)
    assert torch.all(betas > 0), "All betas values must be positive"
    assert torch.all(betas < 1), "All betas values must be < 1"
    assert betas.isfinite().all(), "Beta schedule contains or Infs"

# 2. Chech monotonic increase for certain schedule types
def test_linear_schedule_is_monotonic():
    betas = make_beta_schedule("linear", 1000)
    assert torch.all(betas[1:] >= betas[:-1]), "Linear schedule should be increasing"

# 3. Edge case: single timestep
def test_single_step_schedule():
    betas = make_beta_schedule("cosine", 1)
    assert betas.shape == (1,)
    assert 0 < betas[0] < 1

# 4. Check known fixed value at start/end for cosine schedule
def test_cosine_schedule_endpoints():
    betas = make_beta_schedule("cosine", 1000)
    assert betas[0].item() < betas[-1].item(), "Cosine schedule should increase"

# 5. Fuzz test for numeric stability
@pytest.mark.parametrize("schedule_type", ["linear", "cosine", "quadratic"])
@pytest.mark.parametrize("timesteps", [10, 100, 1000, 5000])
def test_schedule_stability(schedule_type, timesteps):
    betas = make_beta_schedule(schedule_type, timesteps)
    assert betas.isfinite().all(), f"{schedule_type} with {timesteps} steps has unstable values"