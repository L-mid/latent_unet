
import os
import sys
import types
import math
import random
import numpy as np
import torch
import pytest

# Force non-interactive backend for headless runs
import matplotlib
matplotlib.use("Agg")

# Import the module under test (assumes repo root / PYTHONGPATH already set)
import importlib

# --- Utilities & fixtures ------------------------------------------------------

def _mk_cfg(tmp_path, max_images=4, use=None):
    # Minimal config shim to satisfy Visualizer/visualize_everything access pattern:
    #    cfg.visualizer.{enabled, save_dir, max_images, use, latent_channels}

    cfg = types.SimpleNamespace()
    vis = types.SimpleNamespace()
    vis.enabled = True
    vis.save_path = str(tmp_path)
    vis.max_images = max_images
    vis.use = use or []
    vis.latent_channels = 8
    vis.betas = None
    cfg.visualizer = vis
    return cfg

@pytest.fixture(autouse=True)
def _seed_everything():
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    yield

@pytest.fixture
def tmpdir(tmp_path):
    return tmp_path

@pytest.fixture
def viz():
    # Fresh import each time (handy while iterating)
    sys.modules.pop("latent_unet_v1.utils.visualizer", None)
    return importlib.import_module("latent_unet_v1.utils.visualizer")

# --- Synthetic tensors ------------------------------------------------------------------------

def _rgb_batch(b=6, c=3, h=32, w=32, device="cpu"):
    return torch.clamp(torch.randn(b, c, h, w, device=device), -2, 2)

def _attn_map_list(num_layers=2, b=3, heads=4, h=16, w=16, device="cpu"):
    return [torch.rand(b, heads, h, w, device=device) for _ in range(num_layers)]

# --- Tests for class methods -------------------------------------------------------------------

def test_save_grid_helper(tmpdir, viz):
    cfg = _mk_cfg(tmpdir, max_images=4)
    x = _rgb_batch(b=8)
    from utils.visualizer import _save_image_grid
    _save_image_grid(x, save_dir=tmpdir / "grid.png", filename=f"grid.png", nrow=4)
    assert os.path.exists(tmpdir / "grid.png")

def test_compare_noising(tmpdir, viz):
    fn = viz.VISUALIZER_REGISTRY.get("noising")
    save_dir = tmpdir / "noising"
    b = 8 
    x0 = _rgb_batch(b=b)
    xt = _rgb_batch(b=b)
    xhat = _rgb_batch(b=b)
    fn(x0=x0, xt=xt, xhat=xhat, step=12, save_path=str(save_dir))
    out = save_dir / "denoising_step000012.png"
    assert out.exists()

def test_visualize_attention(tmpdir, viz):
    maps = _attn_map_list(num_layers=3, b=2, heads=4, h=12, w=12)
    fn = viz.VISUALIZER_REGISTRY.get("attention")
    fn(attn_maps=maps, step=7, save_path=str(tmpdir)) 
    # One file per layer 
    for i in range(3): 
        assert (tmpdir / f"attn_layer{i}_step000007.png").exists()

@pytest.mark.skipif(importlib.util.find_spec("sklearn") is None, reason="scikit-learn not installed")
def test_plot_timestep_embedding(tmpdir, viz):
    t_embeds = torch.randn(128, 64) # [T, D]
    fn = viz.VISUALIZER_REGISTRY.get("time_embeddings")
    fn(t_embeds=t_embeds, step=5, save_path=str(tmpdir))
    assert (tmpdir / "time_embedding_pca_000005.png").exists()

def test_plot_beta_schedule(tmpdir, viz):
    betas = torch.linspace(1e-4, 0.02, 1000)
    fn = viz.VISUALIZER_REGISTRY.get("schedule")
    fn(betas=betas, name="beta_sched", save_path=str(tmpdir))
    assert (tmpdir / "beta_sched.png").exists()


def test_plot_guidance_effects(tmpdir, viz):
    g = 4   # guidance scales
    b = 2
    samples = torch.randn(g, b, 3, 32, 32)
    guidance_scales = [0.0, 1.0, 3.0, 7.5]
    fn = viz.VISUALIZER_REGISTRY.get("guidance")
    fn(samples=samples, guidance_scales=guidance_scales, step=3, save_path=str(tmpdir))
    assert (tmpdir / "guidance_grid_000003.png").exists()

def test_visualize_latents(tmpdir, viz):
    # Note: function is registered as a free function via decorator, but the class method also exists.
    # We call the registered function explicitly to ensure it tolerates raw latents.
    assert hasattr(viz, "visualize_everything")
    latents = torch.rand(2, 6, 16, 16)  # [B, C, H, W]
    save_path = tmpdir / "latents"
    # Call the free function directly from registry since that's what's decorated
    # (visualize_everything drives this too, but keep the call surface minimal here)
    fn = viz.VISUALIZER_REGISTRY.get("latents")
    assert fn is not None, "latents visualizer not registered"
    fn(latents=latents, step=1, save_path=str(save_path))
    # Saves one image per channel up to min(B, C)
    for i in range(min(2, 6)):
        assert (save_path / f"latent_ch_{i}_step1.png").exists()

# --- Tests for registry & top-level orchestrator -----------------------------------------------------

def test_registry_contains_expected(viz):
    # Smoke check that registration happened
    keys = set(viz.VISUALIZER_REGISTRY.keys())
    # You can expand this list as you stabalize more entries
    expected = {"latents", "param_norms", "grad_flow", "schedule", "time_embeddings", "attention", "noising", "guidance"}
    # Don't require all right away; just ensure core ones 
    missing = expected - keys
    assert not missing, f"Missing visualizers in registry: {missing}"

def test_visualize_everything_minimal(tmpdir, viz):
    # Build a tiny dummy "model" and call visualize_everything with latents only
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 3, kernel_size=1)

        def forward(self, x):
            return self.conv(x)
        
    model = Tiny()
    latents = torch.randn(2, 4, 16, 16)

    cfg = _mk_cfg(tmpdir, use=["latents", "schedule"])
    # Prepare kwargs to match visualize_everything signature
    betas = torch.linspace(1e-4, 0.02, 64)

    # Monkey-patch the schedule visualizer to accept (model, latents, params, step, save_path, n_channels)
    # by wrapping the original with a shim that pulls out 'betas' from closure.
    
    viz.visualize_everything(
        model=model,
        latents=latents,
        step=42,
        betas=betas,
        named_parameters=model.named_parameters(),
        cfg=cfg.visualizer,
    )

    print (tmpdir)
    assert (tmpdir / "latent_ch_0_step42.png").exists()
    assert (tmpdir / f"schedule.png").exists()

# --- Grad/param plots ---------------------------------------------------------------------

def test_grad_flow(tmpdir, viz):
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.seq = torch.nn.Sequential(
                torch.nn.Conv2d(3, 8, 3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(8, 3, 3, padding=1),
            )
        def forward(self, x):
            return self.seq(x)
        
    model = Tiny()
    x = torch.randn(4, 3, 16, 16, requires_grad=True)
    out = model(x).sum()
    out.backward()
    viz.VISUALIZER_REGISTRY["grad_flow"](named_parameters=model.named_parameters(), 
                                          save_path=str(tmpdir / "grad_flow.png"))
    assert (tmpdir / "grad_flow.png").exists()

def test_param_norms(tmpdir, viz):
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.seq = torch.nn.Sequential(
                torch.nn.Conv2d(3, 8, 3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(8, 3, 3, padding=1),
            )
        def forward(self, x):
            return self.seq(x)
        
    model = Tiny()
    viz.VISUALIZER_REGISTRY["param_norms"](model=model, save_path=str(tmpdir / "param_norms.png"))
    assert (tmpdir / "param_norms.png").exists()

# --- Shape/contract checks (cheap invariants) ------------------------------------------------------

def test_attention_contract_shapes(viz):
    # Ensure the attention visualizer expects list of [B, H, W] per head aggregated to [B, H, W]
    # Our contract here: attn[j].mean(0) should yield [H, W]
    maps = _attn_map_list(num_layers=1, b=2, heads=3, h=7, w=9)
    attn = maps[0]          # [B, Heads, H, W]
    assert attn.ndim == 4
    B, Heads, H, W = attn.shape
    reduced = attn[0].mean(0)   # -> [H, W]
    assert tuple(reduced.shape) == (H, W)







