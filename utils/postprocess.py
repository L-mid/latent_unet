
import torch
from diffusion.ddim import ddim_sample


def to_display(x: torch.Tensor, in_range=(-1, 1)):
    lo, hi = in_range 
    x = (x - lo) / (hi - lo)
    return x.clamp_(0, 1)


def make_grid(x: torch.Tensor, nrow=8):
    # x: (B, C, H, W) in [0, 1]
    from torchvision.utils import make_grid as _make_grid
    return _make_grid(x, nrow=nrow)



# vizualizer example helper
def sample_grid(
        pipeline, *, shape, num_steps=50, nrow=8, seed=None,
        guidance_scale=None, cond=None, decode=False, log_fn=None
):
    imgs, extras = pipeline.generate(
        shape=shape, num_steps=num_steps, seed=seed,
        guidance_scale=guidance_scale, cond=cond, decode=decode
    ) 
    disp = to_display(imgs, in_range=(-1, 1))
    grid = make_grid(disp, nrow=nrow)
    if log_fn:
        log_fn("samples/grid", grid)
    return grid, extras







