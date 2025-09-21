
# if cpu only: drop batch_size to 16/32 and vis_every to 1000 to keep it snappy.

"""
Refactored Visualizer
---------------------
A single, production-oriented Visualizer class with a tiny builder.

Design goals
* One entry point kept around the whole run. 
* Consistent output directory & naming.
* Registry for build-in and user-defined visualizers (plugins).
* Safe, atomic-ish saves; always closes figures.
* Works with plain dict or OmegaConf configs.
* Graceful fallbacks (e.g., PCA without scikit-learn).

Usage (example)
---------------
from visualizer_refactor import build_visualizer

viz = build_visualizer(cfg)     # reads cfg.visualization

# ... inside training loop ...

# call multiple presets selected in cfg.visualization.use
viz.run_all(step=step, model=model, latents=latents, betas=betas, 
            attn_maps=attn_maps, samples=samples, guidance_scales=g_scales)

"""

from __future__ import annotations

import os
import io
import math
import tempfile
import numbers
import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional
from collections.abc import Mapping
import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from einops import rearrange

try:
    from utils.failure_injection_utils.failpoints import failpoints # optional
except Exception:   # pragma: no cover - make optional in case project doesn't include it in some envs
    class _NoFailpoints:
        def should_raise(self, *_args, **_kwargs):
            return None
    failpoints = _NoFailpoints()   

# -------------------------------
# Config
# -------------------------------
@dataclass
class VisualizerConfig:
    enabled: bool = True
    output_dir: str = "outputs/visualizations"
    use: List[str] = field(default_factory=list)
    image_format: str = "png"
    nrow: Any = 4   # Accepts int | str | list | tuple
    normalize: bool = True
    scale_each: bool = True
    failpoint_name: Optional[str] = "visualizer.run_all"
    # How to treat extra kwargs passed to plugins
    strict_kwargs: bool = False     # if true, raise on unknown kwargs instead of dropping
    warn_dropped_kwargs: bool = False   # if true, print a warning wehn dropping extras
    

    @staticmethod
    def from_any(cfg: Any) -> "VisualizerConfig":
        """Best-effort extraction from dict/OmegaConf/object with .visualization."""
        # Accept both cfg and cfg.visualization
        v = getattr(cfg, "visualization", cfg)
        def get(name, default):
            if isinstance(v, Mapping):
                return v.get(name, default)
            return getattr(v, name, default)
        return VisualizerConfig(
            enabled=bool(get("enabled", True)),
            output_dir=str(get("output_dir", "outputs/visualizations")),
            use=list(get("use", [])),
            image_format=str(get("image_format", "png")),
            nrow=int(get("nrow", 4)),
            normalize=bool(get("normalize", True)),
            scale_each=bool(get("scale_each", True)),
            failpoint_name=get("failpoint_name", "visualizer.run_all"),
            strict_kwargs=bool(get("strict_kwargs", False)),
            warn_dropped_kwargs=bool(get("warn_dropped_kwargs", False))
        )
    
# --------------------------------
# Utility helpers
# --------------------------------

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _atomic_save_bytes(data: bytes, dst_path: str) -> None:
    _ensure_dir(os.path.dirname(dst_path))
    with tempfile.NamedTemporaryFile(delete=False, dir=os.path.dirname(dst_path)) as tmp:
        tmp.write(data)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = tmp.name
    os.replace(tmp_path, dst_path)

def _save_figure(fig: plt.Figure, path: str, dpi: int = 120) -> None:
    buf = io.BytesIO()
    fig.savefig(buf, format=os.path.splitext(path)[1].lstrip("."), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    _atomic_save_bytes(buf.getvalue(), path)

def _save_image_grid(
        tensors: torch.Tensor,
        save_path: str,
        *,
        nrow: int = 4,
        normalize: bool = True,
        scale_each: bool = True,
) -> None:
    _ensure_dir(os.path.dirname(save_path))
    grid = vutils.make_grid(tensors, nrow=nrow, normalize=normalize, scale_each=scale_each)
    # save_image does not offer atomic by default; wrap similarly via buffer
    buf = io.BytesIO()
    vutils.save_image(grid, buf, format=os.path.splitext(save_path)[1].lstrip("."))
    _atomic_save_bytes(buf.getvalue(), save_path)



def _coerce_nrow(v: Any, default: int = 4) -> int:
    # OmegaConf ListConfig -> list
    try:
        from omegaconf import ListConfig    # type: ignore
        if isinstance(v, ListConfig):
            v = list(v)
    except Exception:
        pass

    if isinstance(v, numbers.Integral):
        return int(v)
    
    if isinstance(v, str):
        try:
            return int(v),
        except ValueError:
            return default
    if isinstance(v, (list, tuple)):
        if len(v) == 0:
            return default
        
        if len(v) >= 2:
            # Interpret (rows, cols) -> torchvision.make_grid expects columns
            try:
                return int(v[1])
            except Exception:
                try:
                    return int(v[0])
                except Exception:
                    return default
        try: 
            return int(v[0])
        except Exception:
            return default
    return default


def _filter_kwargs(fn: Callable[..., Any], cfg: VisualizerConfig, given: dict) -> dict:
    """
    Filter kwargs by the plugin's signature.
    - If the fn has **kwargs, pass everything.
    - Else pass only parameters present in its signature.
    - If strict_kwargs=True, raise on unknowns; otherwise drop them (optionally warn)
    """

    def _ensure_mapping(given):
        # Coerce 'given' to a mapping
        if isinstance(given, Mapping):
            return dict(given)
        if given is None:
            return {}
        if isinstance(given, (list, tuple, set)):
            # allow list/tuples/set of (k, v) pairs
            return dict(given)
    
    given = _ensure_mapping(given)

    sig = inspect.signature(fn)
    params = sig.parameters

    # Fast path: function accepts **kwargs
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return dict(given)
    
    accepted = set(params.keys())
    to_pass = {k: given[k] for k in given if k in accepted}

    if cfg.strict_kwargs:
        unknown = [k for k in given.keys() if k not in accepted]
        if unknown:
            raise TypeError(f"{fn.__name__} got unexpected kwargs: {unknown}")
    elif cfg.warn_dropped_kwargs:
        unknown = [k for k in given.keys() if k not in accepted]
        if unknown:
            print(f"[Visualizer] Dropping extra kwargs for {fn.__name__}: {unknown}")

    return to_pass      # always a dict


# ---------------------------------
# Visualizer (with plugin registry)
# ---------------------------------

PluginFn = Callable[..., None]

class Visualizer:
    """
    Visualizer with a registry of plotting/logging plugins.
    """

    _RESISTRY: Dict[str, PluginFn] = {}

    def __init__(self, cfg: VisualizerConfig):
        self.cfg = cfg
        self.enabled = bool(cfg.enabled)
        self.root = os.path.abspath(cfg.output_dir)
        _ensure_dir(self.root)
        self.cfg.nrow = _coerce_nrow(self.cfg.nrow)

    # --- registry managment ---
    @classmethod
    def register(cls, name: str) -> Callable[[PluginFn], PluginFn]:
        def deco(func: PluginFn) -> PluginFn:
            cls._RESISTRY[name] = func
            return func
        return deco
    
    def available(self) -> List[str]:
        return sorted(self._RESISTRY.keys())       
    
    # --- convenience path helpers ---
    def _fmt_step(self, step: Optional[int]) -> str:
        return f"{int(step):06d}" if step is not None else "no_step"
    
    def _path(self, *parts: str) -> str:
        path = os.path.join(self.root, *parts)
        _ensure_dir(os.path.dirname(path))
        return path
    
    # --- public API ---
    def log(self, name: str, /, **kwargs: Any) -> None:
        if not self.enabled:
            return
        fn = self._RESISTRY.get(name)
        if fn is None:
            raise KeyError(f"Visualizer '{name}' not found. Available: {self.available()}")
        try:
            fn(self, **kwargs)
        except Exception as e:
            raise RuntimeError(f"Visualizer '{name}' failed {e}") from e
        
    def run_all(self, *, step: Optional[int] = None, **kwargs: Any) -> None:
        """Run all plugins listed in cfg.use. Any missing kwargs are ingored by each plugin."""
        if not self.enabled:
            return
        exc = failpoints.should_raise(self.cfg.failpoint_name) if self.cfg.failpoint_name else None
        if exc: 
            raise exc
        # Combine step into kwargs just once; filter per-plugin
        combined = dict(kwargs)
        if step is not None:
            combined.setdefault("step", step)       # don't understand this step
        for name in self.cfg.use:
            fn = self._RESISTRY.get(name)
            if fn is None:
                continue
            try:
                self_kwargs = _filter_kwargs(fn, self.cfg, combined)
                fn(self, **self_kwargs)
            except Exception as e:
                raise RuntimeError(f"Failed visualizer '{name}': {e}") from e


# -------------------------------
# Built-in plugins
# -------------------------------

# --- NOSING / DENOISING --- 
@Visualizer.register("noising")
def _viz_noising(self: Visualizer, *, x0: torch.Tensor, xt: torch.Tensor, xhat: torch.Tensor,
                    step: Optional[int] = None, tag: str = "denoising") -> None:
    m = min(x0.size(0), xt.size(0), xhat.size(0))
    imgs = torch.cat([x0[:m], xt[:m], xhat[:m]], dim=0)
    out = self._path(tag, f"{tag}_step{self._fmt_step(step)}.{self.cfg.image_format}")
    _save_image_grid(imgs, out, nrow=self.cfg.nrow, normalize=self.cfg.normalize, scale_each=self.cfg.scale_each) # row 4


@Visualizer.register("attention")
def _viz_attention(self: Visualizer, *, attn_maps: Iterable[torch.Tensor], step: Optional[int] = None, 
                   tag: str = "attention") -> None:
    # attn_maps: list/iter of tensors per layer, shaped [B, Heads, H, W] or [B, H, W]
    for layer_idx, attn in enumerate(attn_maps): 
        attn = attn.detach().float().cpu()
        if attn.dim() == 4:     # [B, Heads, H, W] -> mean over heads
            attn = attn.mean(dim=1)
        # now [B, H, W] 
        B = attn.shape[0]
        cols = min(8, B)
        rows = int(math.ceil(B / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(2*cols, 2*rows))
        axes = np.array(axes).reshape(rows, cols)
        for j in range(B):
            r, c = divmod(j, cols)
            ax = axes[r, c]
            ax.imshow(attn[j].numpy(), cmap="viridis")
            ax.axis("off")
        for j in range(B, rows*cols):
            r, c = divmod(j, cols)
            axes[r, c].axis("off")
        fig.suptitle(f"Attention layer {layer_idx}")
        out = self._path(tag, f"{tag}_layer{layer_idx}_step{self._fmt_step(step)}.{self.cfg.image_format}")
        _save_figure(fig, out) 


# --- TIMESTEP EMBEDDINGS (PCA) --- 
@Visualizer.register("time_embeddings")
def _viz_time_embeddings(self: Visualizer, *, t_embeds: torch.Tensor, step: Optional[int] = None,
                         tag: str = "time_embeddings") -> None:
    x = t_embeds.detach().cpu().numpy()
    # Try sklearn PCA, fall back to SVD if unavailable
    try:
        from sklearn.decomposition import PCA   # type: ignore
        proj = PCA(n_components=2).fit_transform(x)
    except Exception:
        # simple 2D via SVD
        x0 = x - x.mean(axis=0, keepdims=True)
        U, S, Vt = np.linalg.svd(x0, full_matrices=False)
        proj = x0 @ Vt[:2].T
    fig, ax = plt.subplots(figsize=(6, 6))
    sc = ax.scatter(proj[:, 0], proj[:, 1], c=np.arange(len(proj)), s=10)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("Timestep")
    ax.set_title("Time Embedding PCA")
    out = self._path(tag, f"{tag}_pca_{self._fmt_step(step)}.{self.cfg.image_format}")
    _save_figure(fig, out)


# --- Schedule (e.g., betas) ---
@Visualizer.register("schedule")    # only does linear beta
def _viz_schedule(self: Visualizer, *, betas: torch.Tensor, name: str = "beta_schedule",
                  step: Optional[int] = None, tag: str = "schedule") -> None:
    b = betas.detach().cpu().numpy()
    fig, ax = plt.subplots()
    ax.plot(b, label="beta")
    ax.set_title("Beta Schedule")
    ax.set_xlabel("timestep")
    ax.set_ylabel("beta")
    ax.legend()
    out = self._path(tag, f"{name}_{self._fmt_step(step)}.{self.cfg.image_format}")
    _save_figure(fig, out)


# --- Guidance Effects ---
@Visualizer.register("guidance")
def _viz_guidance(self: Visualizer, *, samples: torch.Tensor, guidance_scales: Iterable[float],
                  step: Optional[int] = None, tag: str = "guidance") -> None:
    # samples: [G, B, C, H, W] -> grid with nrow=len(scales)
    imgs = rearrange(samples, 'g b c h w -> (g b) c h w')
    out = self._path(tag, f"{tag}_grid_{self._fmt_step(step)}.{self.cfg.image_format}")
    _save_image_grid(imgs, out, nrow=max(1, len(list(guidance_scales))),
                     normalize=self.cfg.normalize, scale_each=self.cfg.scale_each)


# --- Gradient Flow ---
@Visualizer.register("grad_flow")
def _viz_grad_flow(self: Visualizer, *, named_parameters: Iterable, step: Optional[int] = None,
                   tag: str = "grad_flow") -> None:
    ave_grads, max_grads, layers = [], [], []
    for n, p in named_parameters:
        if p is not None and p.requires_grad and p.grad is not None:
            layers.append(n)
            g = p.grad.detach()
            ave_grads.append(float(g.abs().mean()))
            max_grads.append(float(g.abs().max()))
    if not layers:
        return  # nothing to plot
    idx = np.arange(len(layers))
    fig, ax = plt.subplots(figsize=(max(12, len(layers)*0.3), 6))
    ax.bar(idx, max_grads, alpha=0.5, label='max')
    ax.bar(idx, ave_grads, alpha=0.5, label='mean')
    ax.axhline(0, color='k', linewidth=1)
    ax.set_xticks(idx)
    ax.set_title("Gradient flow per layer")
    ax.set_xlabel("layers")
    ax.set_ylabel("|grad|")
    ax.legend()
    ax.grid(True)
    out = self._path(tag, f"{tag}_{self._fmt_step(step)}.{self.cfg.image_format}")      
    _save_figure(fig, out) 


# --- Prameter Norms ---
@Visualizer.register("param_norms")
def _viz_param_norms(self: Visualizer, *, model: torch.nn.Module, step: Optional[int] = None,
                     tag: str = "param_norms") -> None:
    names, norms = [], []
    for name, param in model.named_parameters():
        if param is not None and param.requires_grad:
            try:
                norms.append(float(param.detach().data.norm()))
            except Exception:
                norms.append(float(torch.nan))
            names.append(name)
    if not names:
        return
    idx = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(max(12, len(names)*0.3), 6))
    ax.bar(idx, norms)
    ax.set_xticks(idx)
    ax.set_xticklabels(names, rotation=90, fontsize=8)
    ax.set_title("Weight L2 norms per layer")
    ax.set_title("layer")
    ax.set_ylabel("L2 norm")
    out = self._path(tag, f"{tag}_{self._fmt_step(step)}.{self.cfg.image_format}")
    _save_figure(fig, out)


# --- Latents ---
@Visualizer.register("latents")
def _viz_latents(self: Visualizer, *, latents: torch.Tensor, step: Optional[int] = None,
                 tag: str = "latents", max_channels: int = 16) -> None:
    # latents: [B, C, H, W]
    z = latents.detach().float().cpu()
    b, c, h, w = z.shape
    show_c = min(c, max_channels)
    cols = min(8, show_c)
    rows = int(math.ceil(show_c / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2*cols, 2*rows))
    axes = np.array(axes).reshape(rows, cols)
    for i in range(show_c):
        r, c_ = divmod(i, cols)
        axes[r, c_].imshow(z[0, i].numpy(), cmap='viridis')
        axes[r, c_].set_title(f"ch {i}", fontsize=8)
        axes[r, c_].axis("off")
    for i in range(show_c, rows*cols):
        r, c_ = divmod(i, cols)
        axes[r, c_].axis("off")
    fig.suptitle("Latent channels (sample 0)")
    out = self._path(tag, f"{tag}_{self._fmt_step(step)}.{self.cfg.image_format}")
    _save_figure(fig, out)


# -----------------------
# Builder & No-op
# -----------------------

class _NoOpVisualizer(Visualizer):
    def __init__(self, cfg: VisualizerConfig):  # pragma: no cover
        super().__init__(cfg)        
        self.enabled = False
    def log(self, *_, **__):
       return
    def run_all(self, *_, **__):
        return 
    
def build_visualizer(cfg_like: Any) -> Visualizer:
    cfg = VisualizerConfig.from_any(cfg_like)
    if not cfg.enabled:
        return _NoOpVisualizer(cfg)
    return Visualizer(cfg)


__all__ = [
    "Visualizer",
    "VisualizerConfig",
    "build_visualizer",
]
    

def test_usage(load_yaml=False, ymal_path=None):
    """Usage test."""

    ymal_path = str(ymal_path)

    from model.config import load_config
    if load_yaml: 
        yaml_cfg = load_config(ymal_path)

    def _make_viz_cfg():
        from omegaconf import OmegaConf

        cfg = OmegaConf.create({
            "viz": {
                "enabled": True,
                "output_dir": "outputs/viz_tiny",
                "use": ["schedule", "noising", "grad_flow", "param_norms"],
                "image_format": "png",
                "nrow": 8,
            },
        },
        )

    cfg = yaml_cfg if load_yaml else _make_viz_cfg()

    # from visualizer_refactor import build_visualizer

    viz = build_visualizer(cfg)


    # Example calls in a training loop: (with cfg shims)
    viz.log("noising", step=cfg.viz.step, x0=cfg.viz.x0, xt=cfg.viz.xt, xhat=cfg.viz.xhat)
    viz.log("grad_flow", step=cfg.viz.step, named_parameters=cfg.viz.params)
    viz.run_all(step=cfg.viz.step, model=cfg.viz.model, latents=cfg.viz.latents, betas=cfg.viz.betas,
                attn_maps=cfg.viz.attn_maps, samples=cfg.viz.samples, guidance_scales=cfg.viz.g_scales)

