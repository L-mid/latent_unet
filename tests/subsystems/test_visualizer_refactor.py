
"""Test for the refactored Visualizer.

These are fast, CPU-only, headless tests that verify each build-in plugin:
- Creates an image files at the expected path
- Produces non-empty, non-uniform pixels (basic sanity)

Optional knobs:
- Set env VIZ_SHOW=1 to open saved images for manual inspection (skipped in CI).
- Set env VIZ_WRITE_GOLDEN=1 to (re)wrtie goldens when running snapshot tests.

Run:
    pytest -q tests -k visualizer -m visual

Add to your pytest.ini:

[pytest]
markers =
    visual: tests that generate visual output and write files

    
$env:VIZ_SHOW=1; pytest -q -m visual -k visualizer
# (optional) unset after:
Remove-Item Env:VIZ_SHOW

$env:VIZ_WRITE_GOLDEN=1; pytest -q -m visual -k visualizer  # only for rewriting the golden.

"""

from __future__ import annotations
import os
import io
import sys
import math
import pathlib
from typing import Iterable, List

import numpy as np
import torch
import PIL.Image as Image
import matplotlib
matplotlib.use("Agg")   # headless
import pytest

# --- import the refactor ---
try:    # if the module is importable as a package/module
    from utils.visualizer_refactor import build_visualizer, VisualizerConfig
except Exception:
    # Fallback: load from sibling file next to this test
    import importlib.util
    here = pathlib.Path(__file__).resolve().parent
    ref = here / "visualizer_refactor.py"
    spec = importlib.util.spec_from_file_location("visualizer_refactor", str(ref))
    mod = importlib.util.module_from_spec(spec) # type: ignore
    assert spec is not None and spec.loader is not None, "Cannot import visual visualizer_refactor.py"
    spec.loader.exec_module(mod) # type: ignore
    build_visualizer = mod.build_visualizer # type: ignore
    VisualizerConfig = mod.VisualizerConfig 


# --------------------------------
# Helpers
# --------------------------------

def _nontrivial_image(path: pathlib.Path) -> None:
    assert path.exists(), f"missing image: {path}"
    with Image.open(path) as im:
        arr = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
    assert arr.size > 0, "empty image array"
    assert np.isfinite(arr).all(), "non-finite pixels present"
    # Not all pixels indentical 
    assert float(arr.std()) > 1e-6, f"image appears uniform: std={arr.std()}"


def _maybe_show(path: pathlib.Path, show_any=False) -> None:
    if show_any == True:
        if os.environ.get("VIZ_SHOW") == "1":
            try:
                with Image.open(path) as im:
                    im.show()
            except Exception:
                pass



# --------------------------- 
# Fixtures
# ---------------------------

@pytest.fixture(scope="function")
def outdir(tmp_path, request) -> pathlib.Path:
    d = tmp_path / f"viz_{request.node.name}"
    d.mkdir()
    return d


@pytest.fixture()
def viz(outdir: pathlib.Path):
    cfg = VisualizerConfig(enabled=True, output_dir=str(outdir), use=[], image_format="png")
    return build_visualizer(cfg)

@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)
    np.random.seed(0)


# ------------------------------------
# Each plugin test
# ------------------------------------

@pytest.mark.visual     # looks odd
def test_noising_saves_grid(viz, outdir: pathlib.Path):
    B, C, H, W = 6, 3, 16, 16 
    x0 = torch.rand(B, C, H, W)
    xt = (x0 + 0.25*torch.randn_like(x0)).clamp(0, 1)
    xhat = torch.rand(B, C, H, W)
    viz.log("noising", step=0, x0=x0, xt=xt, xhat=xhat)
    path = outdir / "denoising" / "denoising_step000000.png"
    _nontrivial_image(path)
    _maybe_show(path)


@pytest.mark.visual
def test_attention_saves_layers(viz, outdir: pathlib.Path): 
    B, Ht, Wt, = 5, 8, 8
    # two layers: one [B, H, W], one [B, Heads, H, W]
    attn1 = torch.rand(B, Ht, Wt)
    attn2 = torch.rand(B, 4, Ht, Wt)
    viz.log("attention", step=1, attn_maps={attn1, attn2})
    p0 = outdir / "attention" / "attention_layer0_step000001.png"
    p1 = outdir / "attention" / "attention_layer1_step000001.png"
    _nontrivial_image(p0)
    _nontrivial_image(p1)
    _maybe_show(p0)


@pytest.mark.visual
def test_time_embeddings_pca(viz, outdir: pathlib.Path):
    t_embeds = torch.randn(64, 32)
    viz.log("time_embeddings", step=2, t_embeds=t_embeds)
    path = outdir / "time_embeddings" / "time_embeddings_pca_000002.png"
    _nontrivial_image(path)
    _maybe_show(path)


@pytest.mark.visual
def test_schedule_plot(viz, outdir: pathlib.Path):
    betas = torch.linspace(1e-4, 0.02, steps=128)
    viz.log("schedule", step=3, betas=betas, name="beta_schedule")
    path = outdir / "schedule" / "beta_schedule_000003.png"
    _nontrivial_image(path)
    _maybe_show(path)


@pytest.mark.visual                                 #
def test_guidance_grid(viz, outdir: pathlib.Path):
    G, B, C, H, W = 3, 4, 3, 16, 16
    samples = torch.rand(G, B, C, H, W)
    viz.log("guidance", step=4, samples=samples, guidance_scales=[0.0, 1.0, 2.0])
    path = outdir / "guidance" / "guidance_grid_000004.png"
    _nontrivial_image(path)
    _maybe_show(path)


@pytest.mark.visual
def test_grad_flow(viz, outdir: pathlib.Path):
    import torch.nn as nn
    model = nn.Sequential(
        nn.Conv2d(3, 4, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(4, 3, 3, padding=1),
    )
    x = torch.randn(2, 3, 16, 16)
    y = model(x).sum()
    y.backward()    # produce grads
    viz.log("grad_flow", step=5, named_parameters=model.named_parameters())
    path = outdir / "grad_flow" / "grad_flow_000005.png"
    _nontrivial_image(path)
    _maybe_show(path)


@pytest.mark.visual
def test_param_norms(viz, outdir: pathlib.Path):
    import torch.nn as nn
    model = nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 3, 3, padding=1),
    )
    viz.log("param_norms", step=6, model=model)
    path = outdir / "param_norms" / "param_norms_000006.png"
    _nontrivial_image(path)
    _maybe_show(path)


@pytest.mark.visual
def test_latents(viz, outdir: pathlib.Path):
    z = torch.randn(2, 12, 12, 12)  # [B, C, H, W]
    viz.log("latents", step=7, latents=z)
    path = outdir / "latents" / "latents_000007.png"
    _nontrivial_image(path)
    _maybe_show(path)


@pytest.mark.visual
def test_run_all_respects_use(viz, outdir: pathlib.Path):
    # run all should only execute specified plugins
    viz.cfg.use = ["schedule", "latents"]
    betas = torch.linspace(1e-4, 0.01, steps=32)
    z = torch.randn(1, 4, 8, 8)
    viz.run_all(step=8, betas=betas, latents=z)
    expect1 = outdir / "schedule" / "beta_schedule_000008.png"
    expect2 = outdir / "latents" / "latents_000008.png"
    _nontrivial_image(expect1)
    _nontrivial_image(expect2) 
    _maybe_show(expect1); _maybe_show(expect2)


@pytest.mark.visual
def test_disabled_is_noop(outdir: pathlib.Path):
    cfg = VisualizerConfig(enabled=False, output_dir=str(outdir))
    viz = build_visualizer(cfg)
    # nothing should be created even if we call plugins
    betas = torch.linspace(1e-4, 0.01, steps=8)
    try:
        viz.log("schedule", step=9, betas=betas)
    except Exception:
        # No-op visualizer should not throw if disabled
        pass 
    # directory remains empty
    assert list(outdir.rglob("*.png")) == []


# ---------------------------------
# Optional: simple image snapshot test (goldens)
# ---------------------------------

def _mse(a: np.ndarray, b: np.array) -> float:
    a = a.astype(np.float32) / 255.0
    b = b.astype(np.float32) / 255.0
    return float(np.mean((a - b) ** 2))

@pytest.mark.visual
def test_noising_snapshot(viz, outdir: pathlib.Path):
    """
    Compare against a golden with a loose MSE threshold.
    Create/update the golden by setting VIZ_WRITE_GOLDEN=1.
    """

    B, C, H, W = 4, 3, 16, 16
    x0 = torch.rand(B, C, H, W)
    xt = (x0 + 0.25*torch.randn_like(x0)).clamp(0, 1)
    xhat = torch.rand(B, C, H, W)
    viz.log("noising", step=10, x0=x0, xt=xt, xhat=xhat)
    out_path = outdir / "denoising" / "denoising_step000010.png"
    _nontrivial_image(out_path)

    # golden path under repo (adjust if needed)
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    golden_dir = repo_root / "test" / "goldens" 
    golden_dir.mkdir(parents=True, exist_ok=True)
    golden_path = golden_dir / "noising_step000010.png"

    with Image.open(out_path) as im:
        arr = np.asarray(im.convert("RGB"))

    if os.environ.get("VIZ_WRITE_GOLDEN") == "1":
        Image.fromarray(arr).save(golden_path)
        pytest.skip("Golden (re)written; skipping comparasion")

    assert golden_path.exists(), "Golden does not exist. Run with VIZ_WRITE_GOLDEN=1 to create it."

    with Image.open(golden_path) as gm:
        garr = np.asarray(gm.convert("RGB"))

    # Loose threshold since stochasic inputs may vary; seed helps keep stable
    mse = _mse(arr, garr)
    assert mse < 0.005, f"visual drift too high (mse={mse})"
