
# Going to train a model.
# This model will be designed for high speed inference with as decent of quality as possible.
# Model will use latent codes (e.g. from VQGAN/autoencoder).

# The main 5 goals to remeber to incorperate are (treat as through a production scale model): 
# ✅ Professionalism 
# ✅ Scalability
# ✅ Modularity
# ✅ Optimization Compatibility
# ✅ Debuggability


# What each priority means in code before building:
"""
# ✅ 1. Professionalism 
#   Standerized pratices for complex teams working on production scale models.
#   Each module does one thing — no mixing U-Net and attention in one file.
#   No magic constants in-line: all hyperparams from config or constructor.
#   Use forward(self, x, t, context=None) instead of 50 args.

# ✅ 2. Scalability
#   Channel multpliers and attention levels are defined per-resolution.
#   U-Net layers are build in loops from configs, not hardcoded.
#   Modular attention/resblock choices (e.g. "attn_type: flash")

# ✅ 3. Modularity 
#   ResBlock/Attention/ConvBlock all separate modules.
#   Drop-in replacements: e.g. you can swap SelfAttention for LinearAttention by config.
#   All forward paths return hooks for optional debug extraction.

# ✅ 4. Optimization Compatibility
#   Every block optionally wrapped in torch.utils.checkpoint.
#   AMP-friendly: only sensitive ops excluded from autocast. 
#   register_buffer used for non-leanrable constants (alphas, sigmas, etc.)
#   Time embeddings precomputed once per batch and passed. 

# ✅ 5. Debuggability
#   Shape tracing (if self.debug: print(x.shape, "after layer X"))
#   Assert blocks check input shape validity if debug is on. 
#   Visualizer scripts plot skip connections, latent activations.
#   Fully pluggable hooke(layername) system for inspecting internals.
"""

# FILE STRUCTURE: (needs revision)
"""
latent_unet/
│
├── run.py                          # Entrypoint: training, testing, sampling
├── README.md                       # How to use the repo
├── pyproject.toml  
├── .gitignore  
│
├── etc...

"""

## --- Overview ---
# Full Example files below (keep in mind may have errors/not integrate with one another, they are samples from a few prev UNet & file):
# Every relevent file should be given a 'testing kit' to ENSURE it's debugged in isolation. 
# I'll send completed files as we go to ensure cross compatibility (very important!). 


# File: run.py --------------------------------------------------------------------
# Example:      # empty
"""

if __name__ == "__main__":
    main()

"""

# File: README.md -----------------------------------------------------------------
# Example:
"""
# Latent U-Net Diffusion

A modular, scalable, and aggressively optimized U-Net framework for training denoising diffusion models (DDPm, DDIM, etc.) on 256x256+ resolution images, even on limited compute.

Supports plug-and-play experimentation with residual blocks, time embeddings, attention mechanisms (flash, windowed), midblocks, schedulers, samplers, and more.

---

## 🚀 Features

- Fully config-driven with 'OmegaConf'
- Gradient checkpointing, AMP, EMA
- Modular U-Net blocks (residuals, attention, etc.)
- Flash attention / Swin-style window attention support
- Supports DDPM and DDIM samplers
- Scalable to latent diffusion or larger image sizes
- Plug-in loss functions (p2, MSE, etc.)
- Sample visualizations and training logging built in

---

## 📦 Installation

```bash
git clone https://github.com/yourname/latent-unet.git
cd latent-unet
pip install -e .


## Configuration

Edit configs/unet_config.yaml to control:
- Model structure (ch_mults, attention kind, block types)
- Training hyperparameters (batch_size, EMA, checkpointing)
- Sampler type (ddpm, ddim)
- Optimization details (lr, amp, grad_clip)


## Training

Training the model (from scratch):
python run.py --config configs/unet_config.yaml 

Training the model (resuming training from checkpoint):
python run.py --config configs/unet_config.yaml --resume


## Sampling

Generated images will appear in outputs/ :
python run.py --config configs/unet_config.yaml --sample_only


## Logging
- TensorBoard logging to ./logs/
- Add your wandb API key and enable via logger.py if needed.


## TODO
- Add LoRA for lightweight fine-tuning
- Integrate VAE encoder/decoder for latent diffusion
- Add DDP support for multi-GPU scaling
- Add more attention types (linear, Performer, etc.)


## License
MIT License


## Citations
This repo draws inspiration from:
- [Ho et al., 2020] Denoising Diffusion Probabilitic Models (DDPM)
- [Song et al., 2021] DDIM
- [Rombach et al., 2022] Latent Diffusion Models


-----------------------
# Project structure:

latent_unet/
│
├── tbd


## For mapping tree keys:

indentations, 
or use (copy pase:)

├──

└──

│
"""

# File: pyproject.toml ------------------------------------------------------------
# Example:
"""
[project]
name = "latnet-unet"
version = "0.1.0"
description = "Modular diffusion model framework with UNet"
authors = [{ name = "Your Name" }]
readme = "README.md"
requires-python = ">=3.9"
license = { text = "MIT" }

dependencies = [
    "torch>=2.0",
    "omegaconf",
    "einops",
    "wandb",
    "tqdm",
    "matplotlib",
    "numpy",
    "scikit-learn",
]

[project.optional-dependencies]
dev = [
    "black",        
    "ruff",   
    "mypy",        
    "pytest",     
    "pytest-cov",
    "pytest-mock",
    "xformers",
    "tensorstore",   
    "jupyter",    
]

[build-system]
requires = ["setuptools>=61.0]
build-backend = "setuptools.build_meta"

[tool.black]
line-length = 88

[tool.mypy]
strict = true
ingore_missing_imports = true


# Install in editable mode: pip install -e ".[dev]"

# Start using the repo like a real Python module: 
# from latent_unet.model.unet import UNet.
"""

# File: .gitingore ----------------------------------------------------------------
# Example:
"""
# --- Python Cache ---
__pycache__/
*.py[cod]
*.so

# --- Virtual Environments ---
.venv/
env/
venv/
*.egg-info/

# --- Jupyter Notebook ---
.ipynb_checkpoints/

# --- Logs, Outputs, Checkpoints ---
logs/
checkpoints/
outputs/
samples/
*.log
*.out
*.err

# --- Model Artifacts --- 
*.pt
*.pth
*.ckpt
*.onnx

# --- Training Tools ---
wandb/
tensorboard/
lightning_logs/
mlruns/

# --- Dataset Artifacts ---
*.zip
*.tar
*.tar.gz
*.npz
*.npy
*.h5

# --- IDEs & Editor Configs ---
.vscode/
.idea/
*.swp

# --- OS & Misc ---
.DS_Store
Thumbs.db
*.bak
*.tmp

# --- Configs and Overrides (Optional) --- 
config.yaml
*.local.yaml
"""

# File: .github/workflows/ci.yaml
"""
name: Diffusion Model CI

on:
    push:
        branches: [ main ]
    pull_request:
        branches: [ main ]

jobs:
    test:
        run-on: ubuntu-latest
        strategy:
        matrix:
            python-version: ["3.10"]
        steps:
        - uses: actions/checkout@v3

        - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v4
        with: 
            python-version: ${{ matrix.python-version }}

        - name: Install dependencies
        run: |
            python -m pip install --upgrade pip
            pip install -e .[dev]

        - name: Run unit tests
        run: | 
            pytest tests/unit --maxfail=5 --disable-warnings --durations=10

        - name: Run integration tests
        run: |
        pytest tests/integration --maxfail=3 --disable-warnings

        - name: Run subsystem tests
        run: |
            pytest tests/subsystems --maxfail=3 --disable-warnings

"""

#   configs/

# File: unet_config.yaml ----------------------------------------------------------
# Notes:
"""
Doesn't match the U-Net.
"""
# unet_config.yaml sample:
"""
# ---- General Settings ----
seed: 1337
device: "cuda"
resume_path: null

model:
    name: "Unet"
    image_size: 3
    in_channels: 3
    base_channels: 128
    channel_multipliers: [1, 2, 4, 8]
    num_res_blocks: 2

attention:
    kind: "vanilla"       # "vanilla", "linear", "flash"
    num_heads: 8
    start_layer: 2     # apply at downsample levels 1 and 2
    window_size: 7

resblock: 
    kind: "default"     # could be "convnext", "film", etc.
    norm: "group"
    use_scale_shift_norm: false

time_embedding:
    dim: 512
    type: "sinusoidal"      # "sinusoidal", "fourier", learned 

midblock:
    use_attention: true
    num_layers: 2

updown: 
    conv_type: "standard"       # could be "resample, "convnext_down", etc.
    use_skip_connections: true

final_head:
    use_norm: true
    activation: "SiLU"
    kernal_size: 3

ema:
    use_ema: true
    decay: 0.9999
    update_every: 1

schedule:
    schedule: "cosine"
    beta_start: 0.0001
    beta_end: 0.02
    timesteps: 1000    

optim:
    optimizer: "adamw"
    lr: 3e-4
    betas: [0.9, 0.999]
    weight_decay: 0.01
    scheduler: "cosine"         # Or "linear", "constant"
    step_size: 1000
    gamma: 0.95

    
# ---- Losses -----    
losses:
    - type: "mse"
    weight: 1.0

    - type: "lpips"
    start_weight: 0.0
    end_weight: 1.0
    schedule: "linear"

    - type: "vgg"
    weight: 0.5
    schedule: "cosine"
    
# ---- Training Hyperparameters ----    

training: 
    batch_size: 32
    num_epochs: 300
    grad_clip: 1.0
    grad_accum: 1
    amp: true
    gradient_checkpointing: true 
    vis_interval: 10        # visualize every N epochs
    ckpt_interval: 25       # checkpoint every N epochs
    num_workers: 8


checkpoint:
    backend: "zarr"         # or "vanilla", or "tensorstore"
    save_every: 1000
    out_dir: "checkpoints/baseline_flash"

    
# ---- Sampling ----

sampler:
    type: "ddim"    # Options: ddpm | ddim | edm

    # Shared sampling config
    num_steps: 50
    guidance_scale: 3.5
    classifier_free: true

    # DDPM-specific (ancestral sampling)
    eta: 1.0

    # DDIM-specific
    ddim_eta: 0.0 # Usually 0 for determinisitc, >0 for stochastic

    # EDM-specific
    sigma_min: 0.002
    sigma_max: 80.0
    rho: 7.0
    solver: "heun" # heun | euler

# ---- Data ----

data:
    name: laion # (or others custom tbd)
    root: /data/laion
    batch_size: 64
    num_workers: 8
    shuffle: true
    tokenizer: clip
    prefetch: true 
    transform_type: strong
    use_webdataset: true
    

# ---- Logging ----

logging:
    use_wandb: true
    project: "latent-unet"
    run-name: "baseline_128ch_flash"    

    
# ---- Visualization ----
    

visualizer:
    enable: true
    output_dir: ./visuals
    save_every: 500     # Save every N steps
    max_images: v       # Number of images per visualization
    steps_to_plot: [0, 50, 100, 250, 500, 750, 999]
    plot_attention: true
    plot_latents: true
    plot_timesteps: true
    use: ["grad_flow", "param_norms", "latents"]
    latent_channels: 8
    

# --- Debugging ---

debug:
    enable_verbose_logs: true
    show_latent_heatmaps: false
    inspect_attention_weights: true

"""

# File: test_config.ymal
"""
test_config:
    run_all: false      # Set to true to override and run everything

    modules:
        sampler_utils:
            enabled: true
            tests:
            -   dynamic_thresholding
            -   cfg_guidance
            -   scheduled_guidance
            -   self_conditioning
            -   snr
            -   snr_weight
            -   curriculum_sampling
            -   debug_hooks

        residual_block:
            enabled: true
            tests:
            -   shape_consistency
            -   backprop_gradients
            -   variant_compatibility
            -   normalization_stability
            -   gradient_checkpointing
            -   film_modulation

        attention:
            enabled: true
            tests:
            -   vanilla
            -   flash
            -   window
            -   device_support
            - attention_shapes

        down_block:
            enabled: true
            tests:
            -   output_size_check
            -   residual_and_attention_check
            -   downsampling_path_consistency

        up_block:
            enabled: true
            tests:
            -   skip_connection_check
            -   output_size_check
            -   upsampling_path_consistency

        mid_block:
            enabled: true
            tests:
            -   attention_effectiveness
            -   bottleneck_integrity

        final_head:
            enabled: true
            tests:
            -   projection_shape
            -   device_transfer
            - initalization_properties

        unet:
            enabled: true
            tests:
            -   integration_shape
            -   device_transfer
            -   module_wiring
            -   full_pass_through

        diffusion:
            enabled: true
            tests:
            -   ddpm
            -   ddim
            -   edm
            -   forward_process
            -   sampler_utils

        logging:
            verbose: true
            colored_output: true
            log_to_file: true
            file_path: logs/test_summary.txt

"""

# File: patch_config.yaml
"""
enabled_patch_groups:
    - torch
    - deepspeed
    - xformers

patch_overrides:
    torch:
        inference_mode_logger: true
        dummy_bugfix: false

    deepspeed:
        dummy_patch: true

    xformers:
        flash_attn_patch: false

"""

# File: mock_data_config.ymal
"""
defaults:
    seed: 42
    device: cpu
    dtype: float32

image_batch:
    batch_size: 8
    channels: 3
    height: 256
    width: 256
    value_range: [0.0, 1.0]

latent_batch:
    batch_size: 8
    latent_dim: 4
    height: 32
    width: 32

text_embedding:
    batch_size: 8
    embedding_dim: 768

unet_inputs:
    image_size: 256
    latent_dim: 4
    embedding_dim: 768

"""

#   model/

# File: build_unet.py -------------------------------------------------------------
# Notes:
"""
Builds U-Net from configs.
"""
# build_unet_from_config sample:
"""
from model.unet import UNet
from modules.time_embedding import get_time_embedding
from modules.residual_block import get_res_block
from modules.mid_block import MidBlock
from modules.down_block import DownBlock
from modules.up_block import UpBlock
from modules.final_head import FinalHead
from modules.attention import get_attention_block

def build_unet_from_config(cfg):
    # Time Embedding
    time_embedding = get_time_embedding(cfg.time_embedding)

    # Down/Up Sampling Channels
    base = cfg.model.base_channels
    ch_mults = cfg.model.ch_mults
    in_channels = [base * m for m in ch_mults[:-1]]
    out_channels = [base * m for m in ch_mults]

    # Mid Block
    mid_block = MidBlock(
        dim=out_channels[-1],
        time_embed_dim=cfg.time_embedding.dim,
        resblock=cfg.resblock,
        attention=cfg.attention
    )

    # Down Blocks
    downs = [
        DownBlock(
            in_ch=in_c,
            out_ch=out_c,
            time_embed_dim=cfg.time_embedding.dim,
            resblock_cfg=cfg.resblock,
            apply_attention=(i >= cfg.attention.start_layer)
        )
        for i, (in_c, out_c) in enumerate(zip([base] + in_channels, out_channels))
    ]

    # Up Blocks
    ups = [
        UpBlock(
            in_ch=out_c * 2, # Due do skip connections
            out_ch=in_c,
            time_embed_dim=cfg.time_embedding.dim,
            resblock_cfg=cfg.resblock,
            attention_cfg.attention,
            apply_attention=(i >= cfg.attention.start_layer)
        )
        for i, (in_c, out_c) in enumerate(zip(reversed(in_channels), reversed(out_channels)))
    ]

        # Final Projection Layer
        final_head = FinalHead(base, cfg.model.out_channels)

    return UNet(
        in_channels=cfg.model.in_channels,
        base_channels=base,
        time_embedding=time_embedding,
        downs=downs,
        mid=mid_block,
        ups=ups,
        final_head=final_head
    )

"""

# File: unet.py -------------------------------------------------------------------
# Notes:
"""
3 downsampling stages (32x32 -> 16x16 -> 4x4). Less if we can make it good. (or built for latents)
"""
# U-Net example:
"""
import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        base_channels: int,
        time_embedding: nn.Module,
        downs: nn.ModuleList,
        mid: nn.Module,
        ups: nn.ModuleList,
        final_head: nn.Module
    ):
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.time_embedding = time_embedding
        self.downs = downs
        self.mid = mid
        self.ups = ups
        self.final_head = final_head

        self.init_conv = nn.Conv2d(in_channels, base_channels, kernal_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # Compute time embedding
        temb = self.time_embedding(t)

        # Initial projection
        x = self.init_conv(x)

        # Down path with skip connections
        skips = []
        for down in self.downs:
            x, skip = down(x, temb)
            skips.append(skip)

        # Mid block
        x = self.mid(x, temb)

        # Up path with skip connections
        for up, skip in zip(self.ups, reversed(skips[:-1])):    # last skip not resued
            x = up(x, skip, temb)

        # Final output projection
        out = self.final_head(x)
        return out


"""

# File: config.py ----------------------------------------------------------------
# Notes:
"""
Not compatible, just an example. 
"""
# config.py sample: (advanced)
"""
from dataclasses import dataclass, field
from typing import List, Optional
from omegaconf import OmegaConf

# ---- Model Configs ----
@dataclass
class TimeEmbeddingConfig:
    kind: str = "sinusoidal"    # sinusoidal | learned | fourier
    dim: int = 512 

@dataclass
class ResBlockConfig:
    kind: str = "vanilla"       # vanilla | FiLM | convnet-style

@dataclass
class AttentionConfig:
    kind: str = "vanilla"       # vanilla | flash | window
    num_heads: int = 4
    start_layer: int = 2        # Apply attention at deeper layers only

@dataclass
class ModelConfig:
    in_channels: int = 3
    out_channels: int = 3
    base_channels: int = 128
    ch_mults: List[int] = field(default_factory=lambda: [1, 2, 4, 8])

# ---- Optimizer, EMA, Scheduler ----

@dataclass
class NoiseScheduleConfig:
    schedule: str = "linear" # linear | cosine | sigmoid
    beta_start: float = 1e-4
    beta_end: float = 0.02
    timesteps: int = 1000

@dataclass
class OptimConfig:
    lr: float = 1e-4
    betas: tuple = (0.9, 0.999)
    weight_decay: float = 0.01

@dataclass
class EMAConfig:
    use_ema: bool = True
    decay: float = 0.9999
    update_every: int = 1

@dataclass
class TrainingConfig:
    batch_size: int = 16
    num_epochs: int = 100
    grad_clip: float = 1.0
    gradient_checkpointing: bool = False
    amp: bool = True

    
@dataclass
class SamplerConfig:
    type: str = "ddim"      # ddpm | ddim | edm
    num_steps: int = 50
    guidance_scale: float = 0.0
    classifier_free: bool = False

    # DDPM / DDIM
    eta: float = 1.0
    ddim_eta: float = 0.0

    # EDM-specific
    sigma_min: float = 0.002
    sigma_max: float = 80.0
    rho: float = 7.0
    solver: str = "heun"    # huen | euler

@dataclass
class VisualizerConfig:
    enabled: bool = True
    save_dir: str = "./vis"
    frequency: int = 100
    grad_flow: bool = True
    param_norms: bool = True
    latent_channels: int = 8

    

# ---- Global Config ----
@dataclass
class Config:
    seed: int = 42
    device: str = "cuda"

    # Modules
    model: ModelConfig = ModelConfig()
    time_embedding: TimeEmbeddingConfig = TimeEmbeddingConfig()
    resblock: ResBlockConfig = ResBlockConfig()
    attention: AttentionConfig = AttentionConfig()
    visualizer: VisualizerConfig = VisualizerConfig()

    # Optimization
    optim: OptimConfig = OptimConfig()
    ema: EMAConfig = EMAConfig()
    training: TrainingConfig = TrainingConfig()

    # Sampler
    sampler: SampleConfig = SamplerConfig()

    # Misc
    log_dir: str = "./logs"
    save_dir: str = "./checkpoints"
    project_name: str = "unet_diffusion"

# ---- YAML Loading Utility ----
def load_config(path_or_dict) -> Config:
    # Load a config from YAML file or dict.
    # Supports CLI overrides via OmegaConf.

    if isinstance(path_or_dict, str):
        cfg = OmegaConf.load(path_or_dict)
    else:
        cfg = OmegaConf.create(path_or_dict)

    merged = OmegaConf.merge(OmegaConf.structured(Config), cfg)
    return OmegaConf.to_container(merged, resolve=True, structured=True)


# An example useage from run.py:

from configs.config import load_config
from model.build_unet import build_unet_from_config

cfg = load_config("configs/unet_config.yaml")
model = build_unet_from_config(cfg)

# Example configs/unet_config.yaml

seed: 1337
device: "cuda"

model:
    base_channels
    ch_mults: [1, 2, 4, 8]

time_embedding:
    kind: "sinusoidal"
    dim: 512

resblock:
    kind: "vanilla"

attention:
    kind: "flash"
    start_layer: 2

training:
    batch_size: 32
    gradient_checkpointing: true

ema: 
    use_ema: true
    decay: 0.9999
"""

#   modules/

# File: time_embbeding.py ---------------------------------------------------------
"""
import math
import torch
import torch.nn as nn
form typing import Litteral

# === Sinusoidal Embedding ===
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        # timesteps: [B] or [B, 1] scalar timestep index
        # Returns: [B, dim]

        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

# === Learned Embedding ===
class LearnedTimeEmbedding(nn.Module):
    def __init__(self, dim: int, max_steps: int = 1000):
        super().__init__()
        self.embed = nn.Embedding(max_steps, dim)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        return self.embed(timesteps)

# === FiLM-style Modulation Embedding === 
class FiLMTimeEmbedding(nn.Module):
    def __init__(self, dim: int, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Seqential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim * 2) # Outputs scale and shift
        )

    def forward(self, t_emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.net(t_emb) # [B, 2 * dim]
        scale, shift = out.chunk(2, dim=-1)
        return scale, shift # Would be applied in FiLM layers

def build_time_embedding(cfg) -> nn.Module:
    # cfg: config.time_embedding
    # Expects cfg.kind in {"sinusoidal", "learned", "film"}

    kind = cfg.kind.lower()
    dim = cfg.dim

    if kind == "sinusoidal":
        return SinusoidalTimeEmbedding(dim)
    elif kind == "learned":
        return LearnedTimeEmbedding(dim)
    elif kind == "film":
        return FiLMTimeEmbedding(dim)
    else:
        raise ValueError(f"Unknown time embbeding type: {kind}")
"""

# File: residual_block.py ---------------------------------------------------------
# Notes:
"""
1. Add scale + shift FiLM (not just bias).
2. ENSURE Resblock is dealing with changing channel dims, NOT a conv layer
"""
# ResBlock example: 
"""
import torch
import torch.nn as nn
from typing import Callable, Dict
from einops import rearrange
from modules.attention_block import AttentionBlock
from Modules.norm_utils import get_norm_layer
from utils.debug import debug_log, debug_section, attach_debug_hooks

# ----------------------------------------------------------------------------
# Block Registery
# -------------------------------------------------------------------------------------

BLOCK_REGISTRY: Dict[str, Callable] = {}

def register_block(name: str):
    def decorator(cls):
        BLOCK_REGISTRY[name] = cls
        return cls
    return decorator

# --------------------------------------------------------------------------------------
# Base Block
# --------------------------------------------------------------------------------------

class BaseResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim, norm_type="group", use_attention=False):
        super().__init__()
        self.in_ch, self.out_ch = in_ch, out_ch
        self.use_attention = use_attention

        self.norm1 = get_norm_layer(norm_type, out_ch)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.res_conv = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

        if use_attention:
            self.attn = AttnentionBlock(out_ch)

        attach_debug_hooks(self, "resblock")

    def forward(self, x, t_emb):
        with debug_section("resblock"):
            h = self.conv1(self.act1(self.norm1(x)))
            h += self.time_proj(t_emb).unsqueeze(-1).unsqueeze(-1)
            h = self.conv2(self.act2(self.norm2(h)))

            h += self.res_conv(x)
            if self.use_attention:
                h = self.attn(h)
            debug_log(f"Resblock output: {out.shape}", name=resblock)
            return h

# --------------------------------------------------------------------------------
# FiLM-style Block
# ---------------------------------------------------------------------------------------

@register_block("film")
class FiLMResBlock(BaseResBlock):
    def forward(self, x, t_emb):
        h = self.norm1(x)
        scale_shift = self.time_proj(t_emb).unsqueeze(-1).unsqueeze(-1)
        scale, shift = scale_shift.chunck(2, dim=1)

        h = self.act1(h * (1 + scale) + shift)
        h = self.conv1(h)

        h = self.act2(self.norm2(h))
        h = self.conv2(h)

        h += self.res_conv(x)
        if self.use_attention:
            h = self.attn(h)
        return h

# ---------------------------------------------------------------------------------------
# Vannila Block (same as Base, but registered)
# ---------------------------------------------------------------------------------------

@register_block("vannilla")
class VanillaResBlock(BaseResBlock):
    pass
    

# ----------------------------------------------------------------------------------------
# Builder
# ------------------------------------------------------------------------------------------

def build_res_block(kind: str, in_ch: int, out_ch: int, time_dim: int, norm_type: str, use_attention: bool):
    if kind not in BLOCK_REGISTRY:
        raise ValueError(f"Unknown ResBlock type: {kind}")
    return BLOCK_REGISTRY[kind](in_ch, out_ch, time_dim, norm_type, use_attention)

"""

# File: norm_utils.py --------------------------------------------------------------------------
"""
import torch.nn as nn

def get_norm_layer(norm_type: str, num_channels: int):
    if norm_type == "group":
        return nn.GroupNorm(8, num_channels)
    elif norm_type == "batch":
        return nn.BatchNorm2d(num_channels)
    elif norm_type == "layer":
        return nn.LayerNorm([num_channels, 1, 1]) # usually not recommended for CNNs.
    else:
    raise ValueError(f"Unknown norm type: {norm_type}")
"""

#   modules/    attention/

# File: base_attention.py -----------------------------------------------------------
"""
import torch.nn as nn
from abc import ABC, abstractmethod

class BaseAttention(nn.Module, ABC):
    # Abstract base class for attention blocks.
    # All attention subclasses (Vannilla, flash, windowed, etc.) should inherit from this.

    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

    @absractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward pass for attention layer.

        # Args: 
            x (Tensor): shape (B, C, H, W)
        Returns:
            Tensor: shape (B, C, H, W)

        raise NotImplimentedError("All attention modules must implement forward())

    def extra_repr(self):
        return f"dim={self.dim}, heads={self.num_heads}"
        
"""

# File: vanilla_attention.py --------------------------------------------------------
# Notes:
"""
1. Use window attention for 32x32, 16x16, 8x8, and default attention for 4x4. (or revise for latent integration)
"""
# AttentionBlock sample:
"""
from einops import rearrange
import math

class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=4, norm_groups=8):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        assert channels % num_heads == 0, "Channels must be divisible by number of heads"

        self.norm = nn.GroupNorm(norm_groups, channels)

        self.qkv = nn.Conv1d(channels, channels * 3, kernal_size=1)
        self.proj_out = nn.Conv1d(channels, channels, kernal_size=1)

    def forward(self, x):
        b, c, h, w = x.shape
        x = self.norm(x)
        x = rearrange(x, 'b c h w -> b c (h w)') # Flatten spatial dims

        # QKV projection
        qkv = self.qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, 'b (h d) n -> b h d n', h=self.num_heads),
            qkv    
        )

        # Attention
        scale = self.head_dim ** -0.5
        attn = torch.softmax(torch.einsum('b h d i, b h d j -> b h i j', q k) * scale, dim=-1)
        out = torch.einsum('b h j k, b h d k -> b h d j', attn, v)

        out = rearrange(out, 'b h d n -> b (h d) n')
        out = self.proj_out(out)

        x = x + out # Residula connection
        return rearrange(x, 'b c (h w) -> b c h w', h=h, w=w)
""" 

# File: window_attention.py --------------------------------------------------------
"""
for attention at higher resolutions (16x16+).
"""
# WindowAttenton sample: 
"""
class WindowAttention(nn.Module):
    def __init__(self, dim, window_size=8, heads=4):
        super().__init__()
        self.window_size = window_size
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x): # x: (B, C, H, W)
        B, C, H, W = x.shape
        assert H % self.window_size == 0 and W % self.window_size == 0

        # Rearrange into (B*windows, N, C)
        x = x.view(B, C, H//self.window_size, self.window_size, W//self.window_size, self.window_size)
        x = x.permute(0, 2, 4, 3, 5, 1).reshape(-1, self.window_size**2, C)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(x.shape[0], -1, self.heads, C // self.heads).transpose(1, 2), qkv)

        attn = torch.softmax((q @ k.transpose(-2, -1)) * self.scale, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(x.shape[0], -1, C)
        out = self.proj(out)

        # Restore
        out = out.view(B, H//self.window_size, W//self.window_size, self.window_size, self.window_size, C)
        out = out.permute(0, 5, 1, 3, 2, 4).reshape(B, C, H, W)
        return out
"""

# File: flash_attention.py ---------------------------------------------------------
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_attention import BaseAttention

class FlashAttention(BaseAttention):
    # FlashAttention-like block using PyTorch 2.0+ scaled_dot_product_attention if avaliable.
    # Fallbacks to standard attention otherwise. 

    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__(dim, num_heads)

        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.head_dim = dim // num_heads
        self.scale = self.head ** -0.5

        self.norm = nn.GroupNorm(8, dim)

        self.qkv_proj = nn.Conv2d(dim, dim * 3, kernel_size=1)
        self.out_proj = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.norm(x)

        # Project Q, K, V
        q = q.view(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)
        k = k.view(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)
        v = v.view(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)

        # PyTorch 2.0+ built in flash attention
        try:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        except Exception:
        # Fallback to manual softmax attention
            attn_scores = torch.matmal(q, k.transpose(-2, -1)) * self.scale
            attn = torch.softmax(attn_scores, dim=-1)
            out = torch.matmal(attn, v)

        # Reshape back: (B, heads, HW, head_dim) -> (B, C, H, W)
        out = out.permute(0, 1, 3, 2).reshape(B, C, H, W)
        return self.out_proj(out) + x # Residual connection



"""

#   modules/

# File: mid_block.py ---------------------------------------------------------------
# Notes:
"""
1. Very cheap version. Considered transformer layer decided against it
"""
# MidBlock Sample:
"""
import torch
import torch.nn as nn
from modules.residual_block import get_resblock
from modules.attention.registry import get_attention

class MidBlock(nn.Module):
    def __init__(self, dim, time_embed_dim, resblock, attention):
        super().__init__()

        # Build first ResBlock (no attention)
        self.res1 = get_resblock(
            kind=resblock.kind,
            in_ch=dim,
            out_ch=dim,
            time_dim=time_embed_dim,
            norm_type=resblock.norm_type,
            attention_layer=None
        )

        # Optional Attention
        if attention.use_attention:
            self.attn = get_attention(
                kind=attention.kind,
                dim=dim,
                heads=attention.heads,
                dim_head=attention.dim_head
            )
        else:
            self.attn = None

        # Build second ResBlock (no attn)
        self.res2 = get_resblock(
            kind=resblock.kind
            in_ch=dim,
            out_ch=dim,
            time_dim=time_embed_dim,
            norm_type=resblock.norm_type,
            attention_layer=None
        )

    def forward(self, x, t_emb):
        x = self.res1(x, t_emb)
        if self.attn:
            x = self.attn(x)
        x = self.res2(x, t_emb)
        return x

"""


# File: down_block.py -----------------------------------------------------------
# Notes:
"""
Designed with using latents in mind.
"""
# Downblock Sample:
"""
import torch.nn as nn
from modules.residual_block import get_resblock
from modules.attention.base_attention import get_attention

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim,
                num_layers=1, use_attention=False, resblock_cfg=None, attn_cfg=None):
        super().__init__()
        self.resblocks = nn.ModuleList([
            get_resblock(in_channels if i == 0 else out_channels, out_channels, time_emb_dim, resblock_cfg)
            for i in range(num_layers)
        ])
        self.attn = get_attention(out_channels, attn_cfg) if use_attention else None
        self.downsample = nn.Conv2d(out_channels, out_channels, kernal_size=4, stride=2, padding=1)

    def forward(self, x, t_emb):
        for block in self.resblocks:
            x = block(x, t_emb)
        if self.attn:
            x = self.attn(x)
        skip = x
        x = self.downsample(x)
        return x, skip
"""

# File: up_block.py ---------------------------------------------------------------
# Notes:
"""
Designed with using latents in mind.
"""
# UpBlock Sample:
"""
import torch.nn as nn
from modules.residual_block import get_resblock
from modules.attention.base_attention import get_attention

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim,
                num_layers=1, use_attention=False, resblock_cfg=None, attn_cfg=None):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernal_size=4, stride=2, padding=1)
        self.resblocks = nn.ModuleList([
            get_resblock(out_channels * 2 if i == 0 else out_channels, out_channels, time_emb_dim, resblock_cfg)
            for i in range(num_layers)
        ])
        self.attn = get_attention(out_channels, attn_cfg) if use_attention else None

    def forward(self, x, skip, t_emb):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        for block in self.resblocks:
            x = block(x, t_emb)
        if self.attn:
            x = self.attn(x)
        return x
"""

# File: final_head.py ------------------------------------------------------------
# Notes: 
"""
Might switch to 1x1 conv with no activation
"""
# FinalHead sample:
"""
class FinalBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(8, in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, kernal_size=3, padding=1)
        )

    def forward(self, x):
        return self.block
"""


#   Diffusion/

# File: ddpm.py --------------------------------------------------------------------
"""
import torch
import torch.nn.functional as F
from typing import Tuple, Optional

from model.schedule import get_beta_schedule
from diffusion.sampler_resistry import register_sampler

@register_samper("ddpm")
class DDPM:
    def __init__(
        self,
        model,
        timesteps: int = 1000,
        beta_schedule: str = "cosine",
        prediction_type: str = "epsilon", # or "x0"
        device: str = "cuda"
    ):
        self.model = model
        self.timesteps = timesteps
        self.device = device
        self.prediction_type = prediction_type

        # Load beta and precompute schedules
        betas = get_beta_schedule(beta_schedule, timesteps).to(device)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
        self.posterior_variance = betas * (1. - alphas_cumprod[:-1]) / (1. - alphas_cumprod[1:])

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None):
        # Forward diffusion: sample x_t from x_0 with q(x_t | x_0)

        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alpha = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alpha = self.extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alpha * x_start + sqrt_one_minus_alpha * noise

    def p_losses(self, x_0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None):
        # Compute model loss for given x_0 and time t.

        if noise is None:
            noise = torch.randn_like(x_0)
        x_noisy = self.q_sample(x_0, t, noise)
        pred = self.model(x_noisy, t)

        if self.prediction_type == "epslion":
            target = noise
        elif self.prediction_type == "x0":
            target = x_0
        else: 
            raise ValueError(f"Invalid prediction_type: {self.prediction_type}")

        return F.mse_loss(pred, target)

    @torch.no_grad()
    def p_sample(self, x_t: torch.Tensor, t: torch.Tensor):
        # One reverse sampling step: p(x_{t-1} | x_t)

        model_out = self.model(x_t, t)

        model_out = self.model(x_t, t)
    
        if self.prediction_type == "epsilon":
            x_0_pred = self.predict_x0_from_eps(x_t, t, model_out)
        elif self.prediction_type == "x0":
            x_0_pred = model_out
        else:
            raise ValueError("Invalid prediction type")

        coef1 = self._extract(self.betas, t, x_t.shape)
        coef2 = self._extract(self.alphas, t, x_t.shape)
        posterior_mean = (
            self._extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_0_pred +
            self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * model_out
        )

        if (t == 0).all():
            return x_0_pred
        else:
            noise = torch.randn_like(x_t)
            var = self._extract(self.posterior_varience, t - 1, x_t.shape)
            return posterior_mean + torch.sqrt(var) * noise

    @torch.no_grad()
    def sample(self, shape: Tuple[int], return_all=False):
    # Generate a full denoised image from pure noise using ancestral sampling.

    x = torch.randn(shape, device=self.device)
    xs = [x.clone()] if return_all else None

    for t in reversed(range(self.timesteps)):
        t_tensor = torch.full((shape[0],), t, device=self.device, dtype=torch.long)
        x = self.p_sample(x, t_tensor)
        if return_all
            xs.append(x.clone())

    return x if not return_all else torch.stack(xs, dim=1)

    def _extract(self, a: torch.Tensor, t: torch.Tensor, shape: time.Size):
        # Helper to match tensor dimensions
        out = a.gather(-1, t)
        return out.view(-1, *((1,) * (len(shape) -1))

    def _predict_x0_from_eps(self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor):
        # Convert predicted noise into predicted x_0.
        return (x_t - self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * eps) / \
                self.extract(self.sqrt_alphas_cumprod, t, x_t.shape)
   

"""

# File: ddim.py --------------------------------------------------------------------
"""
import torch
import.nn.functional as F
from tqdm import tqdm
from diffusion.sampler_registry import register_sampler

from .schedule import get_alpha_schedule
from .sampler_utils import apply_classifier_free_guidance

@register_sampler("ddim") 
@torch.no_grad()
    def ddim_sample(
        model: torch.nn.Module,
        shape: tuple,
        num_steps: int,
        eta: float = 0.0,
        guidance_scale: float = 1.0,
        cond: Optional[dict] = None,
        unconditional_cond: Optional[dict] = None,
        cfg_fn: Callable = apply_classifier_free_guidance,
        noise_schedule: str = "cosine",
        device: str = "cuda"
    ):

        B, C, H, W = shape
        x = torch.randn(shape, device=device)

        # Schedule
        timesteps, alphas, alphas_prev, sigmas = get_ddim_schedule(num_steps, schedule=noise_schedule, eta=eta, device=device)

        for i, t in enumerate(timesteps):
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)

            # Classifier-Free Guidance
            model_input = cfg_fn(model, x, t_batch, cond=cond, uncond=unconditional_cond, scale=guidance_scale)

            alpha = alphas[i]
            alpha_prev = alphas_prev[i]
            sigma = sigmas[i]
            sqrt_one_minus_alpha = torch.sqrt(1. - alpha)

            # Predict noise e_theta
            eps = model_input

            # Predict x_0
            x0_pred = (x - sqrt_one_minus_alpha * eps) / torch.sqrt(alpha)
            x0_pred = torch.clamp(x0_pred, -1.0, 1.0)

            # DDIM Step
            dir_xt = torch.sqrt(1. - alpha_prev - sigma**2) * eps
            noise = sigma * torch.randn_like(x)
            x = torch.sqrt(alpha_prev) * x0_pred + dir_xt + noise

        return x


    def get_ddim_schedule(num_steps, schedule="cosine", eta=0.0, device="cuda"):
        # Compute the DDIM alpha, sigma, and timestep schedules.

        alphas_cumprod = get_alpha_schedule(schedule, device=device)
        total_steps = len(alphas_cumprod)

        step_indices = torch.linspace(0, total_steps - 1, num_steps, device=device).long()

        alphas = alphas_cumprod[step_indices]
        alphas_prev = torch.cat([alphas[:1], alphas[:-1]])
        sigmas = eta * torch.sqrt((1 - alphas_prev) / (1 - alphas) * (1 - alphas / alphas_prev))

        return step_indices, alphas, alphas_prev, sigmas
"""

# File: edm.py ---------------------------------------------------------------------
"""
import torch
import torch.nn.functional as F
from tqdm import tqdm
from diffusion.sampler_resistry import register_sampler

# Flexible schedule based on Karras et al. (rho-schedule)
def get_edm_schedule(num_steps, sigma_min, sigma_max, rho, device):
    steps = torch.linspace(0, 1, num_steps, device=device)
    sigmas = (sigma_max ** (1 / rho) + steps * (sigma_min ** (1 / rho) - sigma_max (1 / rho))) ** rho

# Input normalization constants
def edm_preprocess(x, sigma):
    c_skip = 1 / torch.sqrt(1 + sigma ** 2)
    c_out = sigma / torch.sqrt(1 + sigma ** 2)
    c_in = 1 / torch.sqrt(sigma ** 2 + 1)
    return c_skip, c_out, c_in

# Classifier-free guidance application
def apply_cfg(x, pred, guidance_scale, cond_pred=None):
    return pred if cond_pred is None else pred + guidance_scale * (cond_pred - pred)



@register_samper("ddpm")    
# Full EDM sampling procedure
@torch.no_grad()
def edm_sample(
    model,
    shape,
    cfg,
    condition_fn=None,
    device="cuda"
):

    # Load from YAML config
    num_steps   = cfg.sampler.num_steps
    sigma_min   = cfg.sampler.sigma_min
    sigma_max   = cfg.sampler.sigma_max
    rho         = cfg.sampler.rho
    solver      = cfg.sampler.solver
    guidance_scale = getattr(cfg.sampler, "guidance_scale", 0.0)
    classifier_free = cfg.sampler.get("classifier_free", False)

    sigmas = get_edm_schedule(num_steps, sigma_min, sigma_max, rho, device)
    x = torch.randn(shape, device=device) * sigmas[0] # Sample from largest noise

    for i in tqdm(range(ken(sigmas) - 1)):
        sigma = sigmas[i]
        sigma_nest = sigmas[i + 1]
        t = sigma.expand(shape[0])

        if classifier_free:
            x_in = torch.cat([x] * 2, dim=0)
            t_in = torch.cat([t] * 2, dim=0)
            out = model(x_in, t_in)
            pred, cond_pred = out.chunk(2, dim=0)
            out = apply_cfg(x, pred, guidance_scale, cond_pred)
        else:
            out = model(x, t)

        d = (x - out) / sigma
        dt = sigma_next - sigma

        if solver == "euler":
            x = x + d * dt
        elif solver == "heun":
            x_pred = x + d * dt
            t_next = sigma_next.expand(shape[0])
            if classifier_free:
                x_in_next = torch.cat([x_pred] * 2, dim=0)
                t_in_next = torch.cat([t_next] * 2, dim=0)
                out_next = model(x_in_next, t_in_next)
                pred_next, cond_pred_next = out_next.chunk(2, dim=0)
                out_next = apply_cfg(x_pred, pred_next, guidance_scale, cond_pred_next)
            else:
                out_next = model(x_pred, t_next)
            d_next = (x_pred - out_next) / sigma_next
            x = x + 0.5 * (d + d_next) * dt
        else:
            raise ValueError(f"Unknown solver type: {solver})

    return x

"""

# File: sampler_registry.py ----------------------------------------------------------------
"""
SAMPLER_REGISTERY = {}

def register_sampler(name):
    def wrapper(fn):
        SAMPLER_REGISTRY[name.lower()] = fn
        return fn
    return wrapper

def get_sampler(name):
    name = name.lower()
    if name not in SAMPLER_RESISTRY:
        raise ValueError(f"Sampler '{name}' is not registerted.")
    return SAMPLER_RESGISTRY[name]

"""

# File: sampler_utils.py -----------------------------------------------------------
"""
import torch
import torch.nn.functional as F
from typing import Optional, Tuple

# === Dynamic Threshholding (Imagen-style) ===
@torch.no_grad()
def dynamic_thresholding(x0: torch.Tensor, percentile: float = 0.995) -> torch.Tensor:
    # Clamp and normalize outputs based on per-sample dynamic range.
    # Helps prevent oversaturated reconstructions.

    s = torch.quantile(x0.abs().flatten(1), percentile, dim=1).clamp(min=1.0)
    s = s.view(-1, 1, 1, 1)
    return x0.clamp(-s, s) / s

# === Classifier-Free Guidance (with scheduled scale) ===
@torch.no_grad()
def cfg_sample(model, x, t, cond, uncond, guidance_scale: float = 5.0) -> torch.Tensor:
    # Classic classifier-free guidance (cond + uncond interpolation).
    eps_uncond = model(x, t, uncond)
    eps_cond = model(x, t, cond)
    return eps_uncond + guidance_scale * (eps_cond - eps_uncond)

def scheduled_guidance_scale(t: torch.Tensor, max_scale: float = 5.0) -> torch.Tensor:
    # Returns a timestep-dependent guidance scale. Useful in CFG where 
    # you want to suppress scale at small t (late steps).

    return max_scale * (1.0 - t.float() / 1000.0)

@torch.no_grad()
def apply_self_conditioning(model, x, t, cond=None, x_prev=None):
    # Self conditioning wrapper: pass previous x0 as extra input.
    # Assumes model accepts extra channel input (e.g. 6 channels).
    if x_prev is None:
        x_prev = torch.zeros_like(x)
    x_input = torch.cat([x, x_prev], dim=1)
    return model(x_input, t, cond)


# === SNR-Based Sample Weighting (for losses) ===
def compute_snr(t: torch.Tensor, alphas_cumprod: torch.Tensor) -> torch.Tensor:
    # Compute signal-to-noise ratio for each timestep.
    alpha_bar = alphas_cumprod[t.long()]
    return alpha_bar / (1 - alpha_bar + 1e-8)


def snr_weight(t: torch.Tensor, alphas_cumprod: torch.Tensor, gamma: float = 5.0) -> torch.Tensor:
    snr = compute_snr(t, alphas_cumprod)
    return (snr / snr.mean()).pow(gamma)

    
# === Timestep Curriculm Sampling ===
def importance_sample_timesteps(batch_size: int, epoch: int, max_epoch: int,
                                t_min: int = 20, t_max: int = 999) -> torch.Tensor:
    # Prefer smaller timesteps in early epochs, for curriculum training.
    max_t = int(t_min + (t_max - t_min) * min(epoch / max_epoch, 1.0))
    return torch.randint(t_min, max_t + 1, (batch_size,))


# === Debugging Hooks ===
@torch.no_grad()
def debug_denoising(model, x, t, cond=None):
    # Logs mean/std of prediction and runtime. Can be used for profiling.
    import time
    start = time.time()
    pred = model(x, t, cond)
    elapsed = time.time() - start
    print(f"[Step {t.item()}]") μ: {pred.mean.item():3f} σ: {pred.std().item():.3f} | Time: {elapsed:.3f}s")
    return pred

"""

# File: forward_process.py ---------------------------------------------------------
"""
import torch
from model.schedule import get_diffusion_schedule

class ForwardProcess:
    def __init__(self, schedule="cosine", timesteps=1000, device="cpu")
        self.timesteps = timesteps
        self.device = device

        # === 1. Cache the full noise schedule ===
        self.betas, self.alphas, self.alphas_cumprod = get_alpha_schedule(schedule, timesteps, device)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def extract(self, tensor, t, shape):
        # Grab per-timestep values and broadcast to input shape
        out = tensor.gather(0, t).float()
        return out.reshape(-1, 1, 1, 1).expand(shape)

    def q_sample(self, x_start, t, noise=None, return_noise=False):
        # Sample from q(x_t | x_0)
        if noise is None
            noise = torch.randn_like(x_start)
        sqrt_alpha = self.extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus = self.extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        x_t = sqrt_alpha * x_start + sqrt_one_minus * noise
        return (x_t, noise) if return_noise else x_t

    def get_snr_weights(self, t):
        # Compute SNR = alpha / (1 - alpha) for loss weighting.
        alpha_bar = self.extract(self.alphas_cumprod, t, (t.shape[0], 1, 1, 1))
        snr = alpha_bar / (1 - alpha_bar)
        return snr

    def q_posterior(self, x_start, x_t, t):
        # (Optional) q(x_{t-1} | x_t, x_0) for ELBO / DDPM++ loss.
        raise NotImplementedError("Posterior logic needed only for log-likelihood modeling.")
    
    def visualize_trajectory(self, x0, steps=[0, 250, 500, 750, 999]):
        # Return noise samples for multiple timesteps (for GIFs or debugging)
        x_t_series = []
        for t_scalar in steps:
            t = torch.full((x0.shape[0],), t_scalar, dtype=torch.long, device=x0.device)
            x_t = self.q_sample(x0, t)
            x_t_series.append(x_t)
        return x_t_series

"""

# File: schedule.py ---------------------------------------------------------------
"""
import torch
import math
from dataclasses import dataclass
from typing import Literal, Optional


# ---- BETA SCHEDUELE ----

def make_beta_schedule(
    schedule_type: Literal["linear", "cosine", "quadratic"],
    timesteps: int,
    beta_start: float = 1e-4,
    beta_end: float = 0.02, 
) -> torch.Tensor:
    if schedule_type = "linear":
        return torch.linspace(beta_start, ebta_end, timesteps)

    elif schedule_type == "cosine":
        return betas_for_alpha_bar(timesteps)

    elif schedule_type == "quadratic":
        return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2

    else:
        raise ValueError(f"Unknown beta schedule type: {schedule_type}")


def betas_for_alpha_bar(timesteps: int, max_beta=0.999) -> torch.Tensor:
    def alpha_bar(t):
        return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2

    betas = []
    for i in range(timesteps):
        t1 = i / timesteps
        t2 = (i + 1) / timesteps
        beta = min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta)
    return torch.tensor(betas, dtype=torch.float32)


# ---- SCHEDULE WRAPPER ----

@dataclass
class DiffusionShcedule:
    betas: torch.Tensor
    alphas: torch.Tensor
    alphas_cumprod: torch.Tensor
    alphas_cumprod_prev: torch.Tensor
    sqrt_alphas_cumprod: torch.Tensor
    sqrt_one_minus_alphas_cumprod: torch.Tensor
    log_one_minus_alphas_cumprod: torch.Tensor
    sqrt_recip_alphas_cumprod: torch.Tensor
    sqrt_recipm1_alphas_cumprod: torch.Tensor
    posterior_variance: torch.Tensor

    def to(self, device):
        for field in self.__dataclass_fields__:
            setattr(self, field, getattr(self, field).to(device))
        return self


def get_diffusion_schedule(
    schedule_type: str,
    timesteps: int,
    beta_start: float = 1e-4.
    beta_end: float = 0.02,
) -> DiffusionSchedule:
    betas = make_beta_scheudle(schedule_type, timesteps, beta_start, beta_end)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alpha_cumprod_prev = torch.cat([torch.tensor([1.0], dtype=alphas.dtype), alphas_cumprod[:-1]])

    return DiffusionSchedule(
        betas=betas,
        alphas=alphas,
        alphas_cumprod=alphas_cumprod,
        alphas_cumprod_prev=alphas_cumprod_prev,
        sqrt_alphas_cumprod=torch.sqrt(alphas_cumprod),
        sqrt_one_minus_alphas_cumprod=torch.sqrt(1.0 - alphas_cumprod),
        log_one_minus_alphas_cumprod=torch.sqrt(1.0 / alphas_cumprod),
        sqrt_recip_alphas_cumprod=torch.sqrt(1.0 / alphas_cumprod),
        sqrt_recipm1_alphas_cumprod=torch.sqrt(1.0 / alphas_cumprod - 1),
        posterior_variance=betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod),
    )

"""


#    trainer/

# File: train_loop.py --------------------------------------------------------------
# Notes:
"""
Should call for utils/faliure_injection during training.
"""
# Sample:
"""
import os, torch, time
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from utils.debug import debug_log, debug_section

from utils.checkpointing import save_checkpoint, load_checkpoint
from utils.visualizer import build_visualizer
from trainer.logger import build_logger
from trainer.optim_utils import build_optimizer, build_scheduler, build_ema
from diffsuion.sampler_registry import build_diffusion
from trainer.losses import get_loss_fn
from tqdm import tqdm


def train_loop(cfg, model, dataset):
    # -------- 1. Setup -----------
    device = torch.device(cfg.device)
    model.to(device)

    diffusion = build_diffusion(cfg.diffusion)
    visualizer = build_visualizer(cfg.visualizer)
    logger = build_logger(cfg.logger)

    dataloader =  Dataloader(dataset, batch_size=cfg.training.batch_size, shuffle=True, num_workers=cfg.training.num_workers)
    optimizer = build_optimizer(model, cfg.optim)
    scheduler = build_scheduler(optimizer, cfg.training)
    ema = build_ema(model, cfg.ema)

    loss_fn = get_loss_fn(cfg.training.loss_type)
    scaler = GradScaler(enabled=cfg.training.amp)

    start_epoch = 0
    if cfg.resume_path:
        start_epoch = load_checkpoint(cfg.resume_path, model, optimizer, ema)

        
    # -------------- 2. Training Loop -----------------
    for epoch in range(start_epoch, cfg.training.num_epochs):
        model.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

        for step, batch in enumerate(pbar):
            with debug_section("train_step"):
                x = batch["image"].to(device)
                t = diffusion.sample_timesteps(x.shape[0]).to(device)
                noise = torch.randn_like(x)
                x_noisy = diffusion.q_sample(x, t, noise=noise)

                with autocast(enabled=cfg.training.amp):
                    noise_pred = model(x_noisy, t)
                    loss = loss_fn(noise_pred, noise, t)

                debug_log(f"Step {step} | Loss: {loss.item():.4f}", name="train_loop")

                scaler.scale(loss).backward()

                if cfg.training.grad_clip:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                if ema: ema.update(model)

                # Log to console and optionally W&B/tensorboard
                if logger: logger.log_metric("loss", loss.item(), step=epoch * len(dataloader) + step)
            
            if scheduler: scheduler.step()

        
        # ------------ 3. Evaluation & Logging ----------------
        if (epoch + 1) % cfg.training.vis_interval == 0:
            model.eval()
            vis_imgs visualizer.visualize(model, diffusion, device)
            if logger: logger.log_images("samples", vis_imgs, step=epoch)

            
        # ----------- 4. Checkpointing --------------
        if (epoch + 1) % cfg.training.ckpt_interval == 0:
            save_checkpoint(cfg.save_dir, model, optimizer, ema, epoch)

            
    logger.finish()

"""

# File: cluster_utils.py
"""
import os
import logging
import torch
import torch.distributed as dist
import time

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------------------
# Enviroment dectection
# --------------------------------------------------------------------------------------

def discover_cluster_env() -> dict:
    # Detects cluster environment configuration from known variables.

    env = {}

    # Distributed backend
    env["backend"] = os.environ.get("DIST_BACKEND", "nccl")

    # Master node address/port
    env["master_addr"] = os.environ.get("MASTER_ADDR", "127.0.0.1")
    env["master_port"] = os.environ.get("MASTER_PORT", "29500")

    # World info
    env["world_size"] = int(os.environ.get("WORLD_SIZE", 1))
    env["rank"] = int(os.environ.get("RANK", 0))
    env["local_rank"] = int(os.environ.get("LOCAL_RANK", 0))

    # Optional SLURM support
    if "SLURM_PROCID" in os.environ:
        env["rank"] = int(os.environ["SLURM_PROCID"])
    if "SLURM_NTASKS" in os.environ:
        env["world_size"] = int(os.environ["SLURM_NTASKS"])
    if "SLURM_LOCALID" in os.environ:
        env["local_rank"] = int(os.environ["SLURM_LOCALID"])

    return env


# -------------------------------------------------------------------------
# Distributed init (cluster-safe)
# -------------------------------------------------------------------------

def initialize_distributed(timeout_seconds: int = 1800):
    # Safe, idempotent distributed initialization using cluster environment.

    if not torch.distributed.is_available():
        logger.warning("[CLUSTER] torch.distributed not available.")

    if dist.is_initialized():
        logger.info("[CLUSTER] torch.distributed already initialized.")
        return

    env = discover_cluster_env()

    os.environ["MASTER_ADDR"] = env["master_addr"]
    os.environ["MASTER_PORT"] = env["master_port"]

    logger.info(f"[CLUSTER] Initializing distributed:")
    logger.info(f"  - backend:      {env['backend']}")
    logger.info(f"  - world_size:   {env['world_size']}")
    logger.info(f"  - rank:         {env['rank']}")
    logger.info(f"  - local_rank:   {env['local_rank']}")
    logger.info(f"  - master_addr:  {env['master_addr']}")
    logger.info(f"  - master_port:  {env['master_port']}")

    torch.cuda.set_device(env["local_rank"])

    dist.init_process_group(
        backend=env["backend"],
        rank=env["rank"],
        world_size=env["world_size"],
        timeout=torch.distributed.timedelta(seconds=timeout_seconds),
        init_method="env://"
    )


# ------------------------------------------------------------------------------
# Rank utilities
# ------------------------------------------------------------------------------

def get_rank() -> int:
    if is_distributed():
        return dist.get_rank()
    return 0

def get_world_size() -> int:
    if is_distributed():
        return dist.get_world_size()
    return 1

def is_distributed() -> bool:
    return dist.is_available() and dist.is_initalized()

def is_rank0() -> bool:
    return get_rank() == 0


# ------------------------------------------------------------------------------
# Synchronization helpers
# ------------------------------------------------------------------------------

def global_barrier_sync():
    # Force all processes to sync (e.g., after loading model weights).

    if is_distributed():
        dist.barrier()
        logger.info(f"[CLUSTER] Barrier sync complete (rank {get_rank}).")


        
# ------------------------------------------------------------------------------
# Rank-aware printing
# ------------------------------------------------------------------------------

def rank_print(*args, force: bool = False, **kwargs):
    # Print only from rank 0 by default.

    if is_rank0() or force:
        print(*args, **kwargs)


# --------------------------------------------------------------------------------
# Optional: Retry-safe init wrapper (e.g., for flaky jobs)
# --------------------------------------------------------------------------------

def retry_distributed_init(
    retries: int = 3,
    wait_seconds: int = 5,
    timeout_seconds: int = 1800
):
    for attempt in range(retries):
        try: 
            initialize_distributed(timeout_seconds=timeout_seconds)
            return
        except Exception as e:
            logger.warning(f"[CLUSTER] Init failed (attempt {attempt+1}/{retries}): {e}")
            time.sleep(wait_seconds)
        raise RuntimeError("[CLUSTER] Failed to initialize distributed after multiple attempts.")

"""

# File: losses.py ------------------------------------------------------------------
"""
import torch
import torch.nn.functional as F
from torch import nn
from typing import Dict, Optional
from eniops import rearrange


# Optional perceptual loss libraries
try:
    import lpips
except ImportError:
    lpips = None

try:
    from torchvision.models import vgg16
except ImportError:
    vgg16 = None

    
# -----------------------------
# Loss Registry
# -----------------------------

LOSS_REGISTRY = {}

def register_loss(name):
    def decorator(fn):
        LOSS_REGISTRY[name] = fn
        return fn
    return decorator

    
# -----------------
Basic Losses
# -----------------

@register_loss("mse")
def mse_loss(pred, targetm weight=None):
    loss = F.mse_loss(pred, target, reduction='none')
    if weight is not None:
        loss = loss * weight.view(-1, 1, 1, 1)
    return loss.mean(), loss.detach()

@resister_loss("huber")
def huber_loss(pred, target, weight=None, delta=1.0)
    loss = F.smooth_l1_loss(pred, target, beta=delta, reduction='none')
    if weight is not None:
        loss = loss * weight.view(-1, 1, 1, 1)
    return loss.mean(), loss.detach()

@register_loss("p2")
def p2_loss(pred, target, weight=None):
    loss = (pred - target) ** 2
    if weight is not None:
        loss = loss * weight.view(-1, 1, 1, 1)
    return loss.mean(), loss.detach()


# -----------------------------
# Perceptual Losses
# -----------------------------

@register_loss("lipis")
def lipis_loss(pred, target, **kwargs):
    assert lipis is not None, "LPIPS not installed!"
    loss_fn = lipis(net='vgg').to(pred.device)
    return loss_fn(pred, target).mean()

@register_loss("vgg")
def vgg_feature_loss(pred, target, **kwargs):
    assert vgg16 is not None, "torchvision VGG16 not available"
    vgg = vgg16(pretrained=True).features[:16].eval().to(pred.device)
    for param in vgg.parameters():
        param.requires_grad = False

    def extract_features(x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        return vgg(x)

    f_pred = extract_features(pred)
    f_target = extract_features(target)
    return F.l1_loss(f_pred, f_target)


# -------------------------------
# Dynamic Weight Scheudles
# -------------------------------

def linear_ramp_weight(start, end, max_steps, step):
    alpha = min(step / max_steps, 1.0)
    return start * (1 - alpha) + end * alpha

def cosine_schedule_weight(start, end, max_steps, step):
    import math
    cos_val = (1 - math.cos(min(step / max_steps, 1.0) * math.pi)) / 2
    return start * (1 - cos_val) + end * cos_val


# ------------------------------
# Combined Loss Module
# ------------------------------

class DiffusionLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.losses = config.losses # List of dicts with {type, weight, schedule}
        self.max_steps = config.training.num_epochs * config.training.steps_per_epoch

    def get_weight(self, loss_cfg, step):
        sched = loss_cfg.get("schedule", "constant")
        start = loss_cfg.get("start_weight", loss_cfg["weight"])
        end = loss_cfg.get("end_weight", loss_cfg["weight"])

        if sched == "linear":
            return linear_ramp_weight(start, end, self.max_steps, step)
        elif sched == "cosine":
            return cosine_schedule_weight(start, end, self.max_steps, step)
        else: 
            return loss_cfg["weight"]


    def forward(self, pred, target, step=0, visualize=False) -> Dict[str, torch.Tensor]:
        total_loss = 0.0
        loss_dict = {}

        for loss_cfg in self.losses:
            kind = loss_cfg["type"]
            weight = self.get_weight(loss_cfg, step)

            loss_fn = LOSS_REGISTRY[kind]
            loss_val = loss_fn(pred, target)

            weighted_loss = weight * loss_val
            total_loss += weighted_loss

            loss_dict[f"{kind}_loss"] = loss_val.detach()
            loss_dict[f"{kind}_weight"] = weight
            loss_dict[f{kind}_weighted"] = weighted_loss.detach()

        loss_dict["total_loss"] = total_loss
        if visualize:
            loss_dict["output_sample"] = pred[:4].detach()
            loss_dict["target_sample"] = target[:4].detach()

        return loss_dict


# -----------------
# Dynamic Weight Schedules
-------------------

@register_schedule("constant")
def constant_schedule(t, alpha_t=None):
    return torch.ones_like(t).float()

@register_schedule("inverse_alpha_squared")
def inverse_alpha_squared(t, alpha_t):
    return 1.0 / (alpha_t ** 2 + 1e-7)

@register_schedule("linear_decay")
def linear_decay(t, T=1000):
    return 1.0 - t.float() / T

@register_schedule("cosine_decay")
def cosine_decay(t, T=1000):
    return 0.5 * (1 + torch.cos(torch.pi * t.float() / T))


# -------------------
# CONFIG-COMPATIBLE LOSS HANDLER
# -------------------

def get_loss_fn(cfg):
    loss_type = cfg.training.loss_type.lower()
    schedule_type = cfg.training.loss_schedule.lower()

    loss_fn = LOSS_REGISTRY[loss_type]
    weight_fn = SCHEDULE_REGISTRY[schedule_type]

    def wrapped(pred, target, t, alpha_t=None):
        weight = weight_fn(t, alpha_t)
        loss, raw = loss_fn(pred, target, weight=weight)
        return loss, raw, weight

    return wrapped

    
# ---------------------------------
# DEBUG VISUALIZER
# ---------------------------------

def visualize_loss(loss_tensor, title="Loss Heatmap"):
    if loss_tensor.dim() == 4:
        heatmap = loss_tensor.mean(dim=1).detach().cpu().numpy()
        for i in range(min(4, heatmap.shape[0])): # Show up to 4 samples
            plt.imshow(heatmap[i], cmap='viridis')
            plt.title(f"{title} [sample {i}]")
            plt.colorbar()
            plt.show()

"""

# File: optim_utils.py -------------------------------------------------------------
"""
import torch
from torch import optim
from torch.cuda.amp import GradScaler
from typing import Any, Dict


# ------------------------------
# EMA
# ------------------------------

class EMA:
    def __init__(self, model, decay=0.9999, device=None):
        self.model = model
        self.decay = decay
        self.ema_model = self.clone_model()
        self.ema_model.eval()

    def _clone_model(self):
        import copy
        ema = copy.deepcopy(self.model)
        for p in ema.parameters():
            p.requires_grad = False
        return ema

    @torch.no_grad()
    def update(self):
        msd = self.model.state_dict()
        for k, v in self.ema_model.state_dict().items()
            if k in msd:
                msd_k = msd[k].detach()
                v.copy_(v * self.decay + (1. - self.decay) * msd_k)

    def state_dict(self):
        return self.ema_model.state_dict()


# ------------------------
# Optimizer Builder
# ------------------------

def build_optimizer(params, cfg):
    if cfg.optim.optimizer == "adam"
        return optim.Adam(params, lr=cfg.optim.lr, betas=cfg.optim.betas)
    elif cfg.optim.optimizer == "adamw"
        return optim.AdamW(params, lr=cfg.optim.lr, betas=cfg.optim.betas, weight_decay=cfg.optim.weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {cfg.optim.optimizer}")


# ------------------------
# Scheduler Builder
# ------------------------

def build_scheduler(optimizer, cfg, total_steps):
    if cfg.optim.scheduler == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    elif cfg.optim.lr_scheduler == "linear":
        return optim.lr_scheduler.LinearLR(optimizer, total_inters=total_steps)
    elif cfg.optim.scheduler == "step":
        return.lr_scheduler.StepLR(optimizer, step_size=cfg.optim.step_size, gamma=cfg.optim.gamma)
    else:
        return None


# ------------------------
# AMP Support
# ------------------------

def build_amp(cfg):
    if cfg.training.amp:
        return GradScaler()
    return None


# ------------------------
# Grad Clipping Utility
# ------------------------

def apply_grad_clipping(model, clip_value):
    if clip_vale > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

"""

# File: logger.py ------------------------------------------------------------------
"""
import os
import json
import logging
import datetime 
from pathlib import Path
from typing import Dict, Any, Optional

from torch.utils.tensorboard import SummaryWriter

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

class ExperimentLogger:
    def __init__(self, cfg, output_dir"logs", use_wandb=True, debug_mode=False):
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.log_dir = Path(output_dir) / f"{cfg.project_name}_{timestamp}"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.debug_mode = debug_mode
        self.writer = SummaryWriter(log_dir=str(self.log_dir / "tensorboard"))

        self.use_wandb = use_wandb and WANDB_AVALIABLE and not cfg.debug.disable_wandb
        self.wandb_run = None

        if self.use_wandb:
            self.wandb_run = wandb.init(
                project=cfg.project_name,
                name=cfg.run_name,
                config=cfg,
                dir=str(self.log_dir / "wandb"),
                resume="allow"
            )

        self.setup_python_logger()

    def _setup_python_logger(self):
        self.logger = logging.getLogger("trainer_logger")
        self.logger.setLevel(logging.DEBUG if self.debug_mode else logging.INFO)
        fh = logging.FileHander(self.log_dir / "log.txt")
        fh.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO if not self.debug_mode else logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def log_scalar(self, tag: str, value: float, step: int):
        self.writer.add_scalar(tag, value, step)
        if self.use_wandb:
            wandb.log({tag: value}, step=step)

    def log_image(self, tag: str, image, step: int):
        self.writer.add_image(tag, image, step)
        if self.use_wandb:
            wandb.log({tag: [wandb.Image(image, caption=tag)]}, step=step)

    def log_dict(self, metrics: Dict[str, float], step: int):
        for k, v in metrics.items():
            self.log_scalar(k, v, step)

    def log_config(self, cfg):
        config_path = self.log_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(cfg, f, indent=4)
        if self.use_wandb:
            wandb.config.update(cfg)

    def print(self, msg: str, level: str = "info"):
        if level == "debug" and not self.debug_mode:
            return
        getattr(self.logger, level)(msg)

    def finish(self):
        self.writer.close()
        if self.use_wandb and self.wandb_run:
            self.wandb_run.finish()

    def log_model_graph(self, model, example_input):
        self.writer.add_graph(model, example_input)

"""


#    utils/

# File: debug.py -------------------------------------------------------------------
"""
import contextlib
import time
import logging
from typing import Optional, Dict

class DebugManager:
# Central debug manager to control verbosity and logging across modules.

    def __init__(self):
        self.enabled = False
        self.verbose = False
        self.tracked_modules = set()
        self.log_timestamps = False
        self.log_namespace = True

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False

    def toggle(self):
        self.enabled = not self.enabled

    def track(self, module_name: str):
        self.tracked_modules.add(module_name)

    def untrack(self, module_name: str):
        self.tracked_modules.discard(module_name)

    def reset(self):
        self.tracked_modules.clear()

    def is_tracked(self, name: str):
        return name in self.tracked_modules or not self.tracked_modules

    def set_verbose(self, flag: bool):
        self.log_timesteps = flag

    def set_namespace(self, flag: bool):
        self.log_namespace = flag


# Singleton debug manager
debug = DebugManager()

def debug_log(msg: str, name: Optional[str] = None):
    # Central logging helper
    if not debug.enabled
        return

    timestamp = f"[{time.strftime('%H:%M:%S')}]" if debug.log_timestamps else ""
    namespace = f"[{name}]" if name and debug.log_namespace else ""
    print(f"{timestamp}{namespace} {msg}")

@contextlib.contextmanager
def debug_section(name: str, print_enter_exit: bool = True):
    # Context manager for scoped debug logs.
    if debug.enabled and debug.is_tracked(name):
        if print_enter_exit:
            debug_log(">> Enter", name=name)
        start = time.time()
        yield
        elapsed = time.time - start
        debug_log(f"<< Exit ({elapsed:.3f}s)"), nane=name)
    else:
        yield

def attach_debug_hooks(module, module_name="")
    # Optional: Register forward/backward hooks for module.

    if not debug.enabled:
        return

    def forward_hook(mod, inp, outp):
        debug_log(f"Forward: {mod.__class__.__name__}, Input shape: {inp[0].shape}, Output shape: {outp.shape}, name=module_name")

    module.register_forward_hook(forward_hook)

"""


#    utils/data/

# File: __init__.py         # Entry point (build_dataset, build_loader)
"""
from .registry import build_dataset_from_registry
from .dataloader import create_dataloader
from .transforms import build_transforms
from .sampler import build_sampler

def build_dataset(cfg, mode="train"):
    # Build dataset object from config.

    transformes = build_transforms(cfg, mode)
    dataset = build_dataset_from_registry(cfg, transforms, mode)
    return dataset

def build_loader(cfg, mode="train"):
    # Build DataLoader object from config.

    dataset = build_dataset(cfg, mode)
    sampler = build_sampler(cfg, dataset, is_train=(mode == "train"))

    loader = create_dataloader(
        dataset=dataset,
        batch_size=cfg[mode]["batch_size"],
        shuffle=(sampler is None and mode == "train"),
        sampler=sampler,
        num_workers=cfg[mode].get"num_workers", 4),
        collate_fn=None     # Optional: add your custom collate_fn logic here.
    )
    return loader

"""

# File: dataloader.py       # Wraps dataset, collate_fn, and builds DataLoader
"""
import torch
import numpy as np
import random
import logging

logger = logging.getLogger(__name__)


def worker_init_fn(worker_id):
    # Safe random seed initialization per dataloader worker.
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)
    logger.info(f"[DATALOADER] Worker {worker_id} seeded with {seed}")


def create_dataloader(dataset, batch_size, shuffle, sampler, num_workers, collate_fn):
    # Centralized DataLoader builder.

    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True,
        worker_init_fn=worker_init_fn
    )
    return loader

"""

# File: transforms.py       # Training/test transforms (with config control)
"""
import torchvision.transforms as T


def build_transforms(cfg, mode="train"):
    # Build image transforms based on config.

    if mode == "train":
        augmentations = [
            T.RandomResizedCrop(cfg.data.image_size, scale=(0.8, 1.0)),
            T.RandomHorizonalFlip(),
        ]
    else:   # validation / test transforms
        augmentations = [
            T.Resize(cfg.data.image_size + 32),
            T.CenterCrop(cfg.data.image_size),
        ]

    augmentations.append(T.ToTensor())
    augmentations.append(T.Normalize(mean=[0.5], std=[0.5]))

    transform = T.Compose(augmentations)
    return transform

    # Very simple, can be extended










"""

# File: sampler.py          # Custom samplers: curriculum, class-balanced, distributed
"""
import torch
import logging

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Central builder
# -----------------------------------------------------------------------------

def build_sampler(cfg, dataset, is_train: bool): 
    # Build appropriate sampler based on config.

    if cfg.data.distributed:
        logger.info("[SAMPLER] Using DistributedSampler")
        return DistributedSampler(dataset, shuffle=is_train)

    if getattr(cfg.data, "class_balanced", False):
        logger.info("[SAMPLER] Using BalancedClassSampler")
        return BalancedClassSampler(dataset)

    # Add any other samplers here (e.g., curriculm, weighted)

    return None     # Use default shuffle inside dataloader.


# -----------------------------------------------------------------------------
# Distributed Sampler (standard torch)
# -----------------------------------------------------------------------------

class DistributedSampler(torch.utils.data.DistributedSampler):
    # Wraps torch DistributedSampler to unify config.

    def __init__(self, dataset, shuffle: bool = True):
        super().__init__(dataset, shuffle=shuffle)


# -----------------------------------------------------------------------------
# Example: Balanced Class Sampler (optional)
# -----------------------------------------------------------------------------

class BalancedClassSampler(torch.utils.data.Sampler):
    # Oversamples classes to balance training. Expects dataset.class_labels attribute to exist (list of int class IDs).

    def __init__(self, dataset):
        self.labels = dataset.class_labels  # [int per sample]
        self.num_samples = len(self.labels)

        class_counts = {}
        for label in self.labels:
            class_counts[label] = class_counts.get(label, 0) + 1

        weights = []
        for label in self.labels:
            weights.append(1.0 / class_counts.get[label])
        self.weights = torch.DoubleTensor(weights)

    def __iter__(self):
        return iter(torch.multinomial(self.weights, self.num_samples, replacement=True).tolist())

    def __len__(self):
        return self.num_samples

"""

# File: registry.py         # Maps cfg.data.name -> dataset class (come back here to register all datasets)
"""
import logging

logger = logging.getLogger(__name__)


# Temporary placeholder, we will fully fill this once datasets are written

DATASET_REGISTRY = {}


def register_dataset(name):
    # Decorator for registering new datasets.

    def decorator(cls):
        DATASET_REGISTRY[name] = cls
        logger.info(f"[DATASET] Registered dataset: {name}")
        return cls
    return decorator


def build_dataset_from_registry(cfg, transforms, mode="train"):
    # Instantiate dataset from registry.

    name = cfg.data.name
    dataset_cls = DATASET_REGISTRY.get(name)

    if dataset_cls is None:
        raise ValueError(f"Unknown dataset name: {name}")

    dataset = dataset_cls(cfg, transforms, mode)
    return dataset

"""

# File: common.py           # Shared helpers (e.g. seed fixing, split loaders)
"""
import torch
import numpy as np
import random
import os
from PIL import Image
import logging

logger = logging.getLogger(__name__)

def worker_init_fn(worker_id):
    # Reproducibility: seed all dataloaderworkers.
    seed = torch.initial_seed() % 2**32
    np.randon.seed(seed)
    random.seed(seed)
    logger.info(f"[COMMON] Worker {worker_id} seeded with {seed}")


def fix_numpy_seed(seed: int):
    # Globally fix all random seeds.

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    logger.info(f"[COMMON] Global seed fixed to {seed}")


def split_dataset(dataset, val_fractrion=0.1):
    # Split dataset into train/val subsets.

    total = len(dataset)
    indices = list(range(total))
    random.shuffle(indices)
    val_size = int(total * val_fraction)
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]

    train_subset = torch.utils.data.Subset(dataset, train_indices)
    val_subset = torch.utils.data.Subset(dataset, val_indices)

    logger.info(f"[COMMON] Dataset split: {len(train_subset)} train / {len(val_subset)} val")
    return train_subset, val_subset


def sanity_check_image(path):
    # Simple image file integrity check.

    try: 
        with Image.open(path) as img:
            img.verify()
        expect Exception as e:
            logger.warning(f"[COMMON] Corrupted image file path detected: {path} - {e}")
            return False
        return True

"""

# File: basic_dataset.py    # ImageFolder-style local dataset
"""
import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset
from .registry import register_dataset
import logging

logger = logging.getLogger(__name__)


@register_dataset("BasicImageFolder")
class BasicImageFolderDataset(Dataset):
    # Standard ImageFolder-style dataset:
    # root/class_name/image.jpg

    def __init__(self, cfg, transforms, mode="train"):
        self.root = cfg.data.path
        self.transforms = transforms


        # Build full file list
        self.samples = []
        self.class_to_idx = self._discover_classes()
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

        for class_name, class_idx in self.class_to_idx.items():
            class_dir = os.path.join(self.root, class_name)
            files = glob.glob(os.path.join(class_dir, "*.jpg")) + \
                    glob.glob(os.path.join(class_dir, "*.png")) + \
                    glob.glob(os.path.join(class_dir, "*.jpeg"))
            for f in files:
                self.samples.append((f, class_idx))

                
        logger.info(f"[DATASET] Loaded {len(self.samples)} samples across {len(self.class_to_idx) classes}")

        # For samplers that expect full label list
        self.class_labels = [label for _, label in self.samples]

    def _discover_classes(self):
        classes = [d for d in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, d))]
        classes.sort()
        class_to_idx = {class_name: idx for idx, class_name in enumerate(classes)}
        logger.info(f"[DATASET] Discovered classes: {class_to_idx}")
        return class_to_idx

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, class_idx = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transforms:
            img = self.transforms(img)
        return {
            "image": img,
            "label": class_idx,
            "path": img_path
        }

"""

# File: webdataset_loader.py # Streaming tar loaders (e.g. LAION)
"""
import os
import logging
import webdataset as wds
from torch.utils.data import IterableDataset
from .registry import register_dataset
from .tokenizer import build_tokenizer

logger = logger.getLogger(__name__)


@register_dataset("WebDataset")
class WebDatasetLoader(IterableDataset):
    # Streaming WebDataset loader for large-scale tar archives.

    def __init__(self, cfg, transforms, mode="train"):
        self.cfg = cfg
        self.transforms = transforms
        self.mode = mode

        self.tokenizer = build_tokenizer(cfg)

        self.shards = self._build_shard_list()

        logger.info(f"[WEBDATASET] Loaded {len(self.shards)} shards for {mode} mode")

        # Distributed-safe sharding handled automatically by WebDataset
        self.pipeline = (
            wds.WebDataset(self.shards, shardshuffle=True, resampled=True)
            .shuffle(cfg.data.shuffle_buffer)
            .decode("pil")
            .rename(image="jpg;png", caption=txt)
            .map(self._preprocess_sample)
            .to_tuple("image", "input_ids")
        )

    def _build_shard_list(self):
        path = self.cfg.data.path
        pattern = self.cfg.data.shard_pattern   # e.g."shard-{00000..09999}.tar"
        full_pattern = os.path.join(path.pattern)
        return full_pattern

    def _preprocess_sample(self, sample):
        image = sample["image"]
        caption = sample["caption"]

        if self.transforms:
            image = self.transforms(image)

        tokenized = self.tokenizer.encode(caption)
        input_ids = tokenized["input_ids"].squeeze(0)

        return {"image": image, "input_ids": input_ids}

    def __iter__(self):
        return iter(self.pipeline)

"""

# File: caption_dataset.py  # COCO / JSON/TSV caption-based datasets
"""
import os
import json
import csv
import logging
from PIL import Image
from torch.utils.data import Dataset
from .registry import register_dataset
from .tokenizer import build_tokenizer

logger = logging.getLogger(__name__)


@register_dataset("CaptionDataset")
class CaptionDataset(Dataset):
    # Paired image-caption(Dataset):
    # Supports JSONL, TSV, or CSV metadata files.

    def __init__(self, cfg, transforms, mode="train"):
        self.cfg = cfg
        self.transforms = transforms
        self.mode = mode

        self.metadata_path = cfg.data.metadata_file
        self.root = cfg.data.image_root
        self.format = cfg.data.metadata_format.lower()

        self.entries = self._load_metadata()

        # Tokenizer for captions
        self.tokenizer = build_tokenizer(cfg)

        logger.info(f"[DATASET] Loaded {len(self.entries)} entries for {mode} mode")

    def _load_metadata(self):
        entries = []
        if self.format == "jsonl":
            with open(self.metadata_path, "r") as f:
                for line in f:
                    obj = json.loads(line)
                    entries.append({"image"L obj["image"], "caption": obj["caption"]})
        elif self.format == "tsv":
            with open(self.metadata_path, "r") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    entries.append({"image": parts[0], "caption": parts[1]})
        elif self.format == "csv":
            with open(self.metadata_path, "r") as f:
                reader = csv.reader(f)
                for row in reader:
                    entries.append({"image": row[0], "caption": row[1]})
        else:
            raise ValueError(f"Unsupported metadata format: {self.format}")
        return entries

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        img_path = os.path.join(self.root, entry["image"])
        caption = entry["caption"]

        image = Image.open(img_path).convert("RGB")
        if self.transforms:
            image = self.transforms(image)

        tokenized = self.tokenizer.encode(caption)
        input_ids = tokenized["input_ids"].squeeze(0)   # remove batch dim

        return {
            "image": image,
            "caption": caption,
            "input_ids": input_ids,
            "path": img_path
        }

"""

# File: tokenizer.py        # For text-conditioning (CLIP, BERT)
"""
import logging
from transformers import AutoTokenizer
import open_clip

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# Tokenizer Wrapper Interface
# ------------------------------------------------------------------------------

class Tokenizer:
    # Unified tokenizer interface.

    def __init__(self, tokenizer, max_length: int, pad_token: str = None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token = pad_token

    def encode(self, text: str):
        return self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

    def encode_batch(self, texts):
        return self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size



# -----------------------------------------------------------------------------
# Tokenizer Builders
# -----------------------------------------------------------------------------

def build_tokenizer(cfg) -> Tokenizer:
    # Build tokenizer from config.

    name = cfg.tokenizer.name.lower()
    max_len = cfg.tokenizer.max_length

    if name == "bert":
        logger.info("[TOKENIZER] Using HuggingFace BERT tokenizer")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        return Tokenizer(tokenizer, max_length=max_len)

    if name == "clip":
        logger.info("[TOKENIZER] Using OpenCLIP tokenizer")
        tokenizer = open_clip.get_tokenizer("ViT-B/32")
        # OpenClip tokenizer has different interface - wrap it manually:
        return OpenCLIPTokenizerWrapper(tokenizer, max_length=max_len)

    if name == "t5":
        logger.info("[TOKENIZER] Using HuggingFace T5 tokenizer")
        tokenizer = AutoTokenizer.from_pretrained("t5-base")
        return Tokenizer(tokenizer, max_length=max_len)

    raise ValueError(f"Unknown tokenizer: {name}")


# --------------------------------------------------------------------------
# OpenCLIP Tokenizer Special Wrapper
# --------------------------------------------------------------------------

class OpenCLIPTokenizerWrapper:
    # Special wrapper for OpenCLIP tokenizer interface. 

    def __init__(self, clip_tokenizer, max_length: int):
        self.tokenizer = clip_tokenizer
        self.max_length = max_length

    def encode(self, text: str):
        tokens = self.tokenizer([text], context_length=self.max_length)
        return {"input_ids": tokens}

    def encode_batch(self, texts):
        tokens = self.tokenizer(texts, context_length=self.max_length)
        return {"input_ids": tokens}

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size    

"""

# File: prefetcher.py       # (Optional) Fast GPU prefetch pipeline
"""
import torch
import logging

logger = logging.getLogger(__name__)


class PerfectchLoader:
    # Wraps a standard DataLoader to enable GPU-side perfecting using CUDA streams.

    def __init__(self, loader, device=None):
        self.loader = loader
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __iter__(self):
        self.stream = torch.cuda.Stream(device=self.device)
        self.loader_iter = iter(self.loader)
        self.preload()
        return self

    def __next__(self):
        torch.cuda.current_stream(self.device).wait_stream(self.stream)
        batch = self.next_batch
        self.preload()
        return batch

    def preload(self):
        try:
            next_batch = next(self.loader_iter)
        except StopInteration:
            self.next_batch = None
            raise StopIteration

        with torch.cuda.stream(self.stream):
            self.next_batch = self._move_to_device(next_batch)

    def _move_to_device(self, batch):
        # Recursively move batch to target device.

        if isinstance(batch, torch.Tensor):
            return batch.to(self.device, non_blocking=True)
        elif isinstance(batch, dict):
            return {k: self._move_to_device(v) for k, v in batch.items()}
        elif isinstance(batch, list):
            return [self._move_to_device(v) for v in batch]
        elif isinstance(batch, tuple):
            return tuple(self._move_to_device(v) for v in batch)
        else:
            return batch    # leave non-tensor objects alone

"""


#   utils/

# File: metrics.py -----------------------------------------------------------------
"""
import torch
import torch.nn.functional as F
import logging
from typing import Callable, Dict, Any, Optional

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------------------
# Metric Registry System
# --------------------------------------------------------------------------------------

class MetricRegistry:
    # Centralized metric manager for batch-wize metric evaluation.

    def __init__(self):
        self.metrics: Dict[str, Callable] = {}

    def register(self, name: str, func: Callable):
        # Register a new metric function.

        self.metrics[name] = func
        logger.info(f"[METRICS] Registered metric: {name}")

    def compute(self, outputs: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        # Compute all registered metrics for given outputs and targets.

        results = {}
        for name, func in self.metrics.items():
            value = func(outputs, targets)
            results[name] = value
        return results

    def list_metrics(self):
        return list(self.metrics.keys())

    
# -------------------------------------------------------------------------------
# Individual metric implementations
# -------------------------------------------------------------------------------

# Mean Squared Error (MSE)
def compute_mse(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    return F.mse_loss(outputs, targets).item()

# Mean Absolute Error (MAE)
def compute_mae(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    return F.l1_loss(outputs, targets).item()
    if mse == 0:
        return float("inf")
    psnr = 20 * torch.log10(torch.tensor(max_val)) - 10 * torch.log10(torch.tensor(mse))
    return psnr.item()

# SSIM placeholder (real SSIM requires external library like pytorch-image-quality)
def compute_ssim(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    logger.warning("[METRICS] SSIM not implemented (requires external library)")
    return -1.0

# LPIPS placeholder (external library)
def compute_lpips(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    logger.warning("[METRICS] LPIPS not implemented (requires external library)")
    return -1.0

# Gradient norm utility (optional for stability diagnostics)
def compute_grad_norm(model: torch.nn.Module) -> float:
    total_norm = 0.0
    for p in model.parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    return total_norm ** 0.5


# -------------------------------------------------------------------------------------
# Registry construction helper
# -------------------------------------------------------------------------------------

def build_default_metric_registry() -> MetricRegistry:
    registry = Metric_Registry()
    registry.register("mse", compute_mse)
    registry.register("mae", compute_mae)
    registry.register("psnr", compute_psnr)
    # Future real image metricsL
    registry.register("ssim", compute_ssim)
    registry.register("lpips", compute_lpips)
    return registry

"""

# File: vanilla_checkpointing.py
"""
import torch
import os
import ymal
from pathlib import Path

def save_checkpoint(model, optimizer, ema=None, epoch=0, step=0, loss=None,
                    path="checkpoint.pt", config=None, scaler=None):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "step": step,
        "loss": loss,
    }

    if ema is not None:
        checkpoint["ema"] = ema.state_dict()

    if scaler is not None:
        checkpoint["scaler"] = scaler.state_dict()

    torch.save(checkpoint, path) # Or rewrite into atomic write for safety


    # Optionally save YAML config snapshot for reproducibility
    if config:
        config_path = str(Path(path).with_suffix("yaml"))
        with open(config_path, "w") as f:
            yaml.dump(config, f)

def load_checkpoint(path, model, optimizer=None, ema=None, scaler=None, device="cpu"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"])

    if optimizer and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    if ema and "ema" in checkpoint:
        ema.load_state_dict(checkpoint["scaler"])

    return {
        "epoch": checkpoint.get("epoch", 0)
        "step": checkpoint.get("step", 0)
        "loss": checkpoint.get("loss", None)
    }

def auto_resume(path_or_dir):
    # Automatically find latest checkpoint in a directory.
    ckpt = sorted(Path(path_or_dir).glop("*.pt"), key=os.path.getmtime)
    return str(ckpts[-1]) if ckpts else None

def rotate_checkpoints(dir_path, keep_last_n=3):
    # Keep only the last N checkpoints in a directory.
    ckpts = sorted(Path(dir_path).glob("*.pt"), key=os.path.getmtime)
    for ckpts in ckpts[:-keep_last_n]:
        os.remove(ckpt)

"""

# File: snapshot_consistency_tools.py
"""
import os
import torch
import numpy as np
import json
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------
# Snapshot saving: first-run capture
# -------------------------------------------------------------------------

def save_snapshot(tensors: Dict[str, torch.Tensor], snapshot_dir: str):

    os.makedirs(snapshot_dir, exist_ok=True)

    for name, tensor in tensors.items():
        path = os.path.join(snapshot_dir, f""{name}.pt)
        torch.save(tensor.cpu(), path)
        logger.info(f"[SNAPSHOT] Saved snapshot: {name}")

    meta_path = os.path.join(snapshot_dir, "snapshot_meta.json")
    meta = {
        "entries": list(tensors.keys()),
        "format": "torch.pt",
        "framework": "pytorch",
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)


# ----------------------------------------------------------------------------
# Snapshot loading
# ----------------------------------------------------------------------------

def load_snapshot(snapshot_dir: str) -> Dict[str, torch.Tensor]:
    # Load reference snapshot

    meta_path = os.path.join(snapshot_dir, "snapshot_meta.json")
    if not os.path.exists(meta_path):
        raise RuntimeError(f"[SNAPSHOT] No metadata found in {snapshot_dir}")

    with open(meta_path, "r") as f:
        meta = json.load(f)

    ref_tensors = {}
    for name in meta["entries"]:
        path = os.path.join(snapshot_dir, f"{name}.pt")
        tensor = torch.load(path)
        ref_tensors[name] = tensor
        logger.info(f"[SNAPSHOT] Loaded reference: {name}")

    return ref_tensors

    
# ------------------------------------------------------------------------------------
# Snapshot comparison
# ------------------------------------------------------------------------------------

def compare_to_snapshot(
    tensors: Dict[str, torch.Tensor],
    snapshot_dir: str,
    atol: float = 1e-6,
    rtol: float = 1e-5,
):
    # Compare current tensors to reference snapshot.

    reference = load_snapshot(snapshot_dir)

    for name, ref_tensor in reference.items():
        if name not in tensors:
            raise AssertionError(f"[SNAPSHOT] Missing tensor in current run: {name}")

        test_tensor = tensors[name].cpu()

        if ref_tensor.shape != test_tensor.shape:
            raise AssertionError(
                f"[SNAPSHOT] Shape mismatch for '{name}': "
                f"expected {ref_tensor.shape}, got {test_tensor.shape}"
            )

        if not torch.allclose(ref_tensor, test_tensor, atol=atol, rtol=rtol):
            diff = (ref_tensor - test_tensor).abs().max().item()
            raise AssertionError(
                f"[SNAPSHOT] Tensor mismatch for '{name}': max diff={diff:.6g} exeeds tolerances"
            )

        logger.info(f"[SNAPSHOT] Match passed: {name}")

    logger.info("[SNAPSHOT] All tensors match ✅")

# ---------------------------------------------------------------------------
# Convience full-cycle one-liner
# ---------------------------------------------------------------------------

def check_or_create_snapshot(
    tensors: Dict[str, torch.Tensor],
    snapshot_dir: str,
    create_if_missing: bool = True,
    atol: float = 1e-6,
    rtol: float = 1e-5
):
    # one call handles both first-run creation and comparison.

    if not os.path.exists(snapshot_dir):
        if create_if_missing:
            logger.info("[SNAPSHOT] No snapshot found - creating new baseline")
            save_snapshot(tensors, snapshot_dir)
            logger.info("[SNAPSHOT] Baseline snapshot created ✅")
        else:
            raise RuntimeError(f"[SNAPSHOT] Snapshot directory not found: {snapshot_dir}")
    else:
        compare_to_snapshot(tensors, snapshot_dir, atol=atol, rtol=rtol)

"""


#    utils/zarr_checkpointing/

# zarr_core.py          #Direct Zarr array open/create logic
"""
import zarr
import numpy as np
import torch
import os
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Store open/create
# ---------------------------------------------------------------------------

def open_store(path: str, mode: str = "a", storage_options: dict = None) -> zarr.Group:
    # Opens (or creates) a Zarr store.

    # Args:
        # path (str): Filesystem path or cloud URL
        # mode (str): "a" (default) = create or open / "w" = overwrite
        # storage_options (dict): Passed to fsspec (for remote acess)

    # Returns:
        zarr.Group: root Zarr group object

    logger.info(f"[ZARR] Opening store: {path} (mode={mode})")

    # Expand fsspec support later (for remote)
    store = zarr.DirectoryStore(path)

    if mode == "w" and os.path.exists(path):
        logger.warning(f"[ZARR] Overwriting existing store at: {path}")
        zarr.convenience.rmdir(store)

    root = zarr.group(store=store, overwrite=(mode == "w"))
    return root


# ------------------------------------------------------------------------
# Tensor write
# ------------------------------------------------------------------------

def write_tensor(group: zarr.Group, name: str, tensor: torch.Tensor, chunks: tuple = None):
    # Saves a tensor into Zarr group.

    # Args:
        # group (zarr.Group): Parent group to write into
        # name (str): Dataset name
        tensor (torch.Tensor): Tensor to save (moved to CPU automatically)
        chunks (tuple): Optionall manual chuck shape

    array_data = tensor.detach().cpu().numpy()

    if name in group:
        logger.warning(f"[ZARR] Overwriting array: {name}")
        del group[name]

    group.create_dataset(
        name=name,
        data=array_data,
        chunks=chunks,
        overwrite=True,
        compressor=zarr.Blosc(cname="zstd", clevel=5, shuffle=2)
    )

    logger.info(f"[ZARR] Wrote tensor '{name}' with shape {tensor.shape}")


# -----------------------------------------------------------------------------
# Tensor read
# -----------------------------------------------------------------------------

def read_tensor(group: zarr.Group, name: str, device: torch.device = "cpu") -> torch.Tensor:
    # Loads tensor from Zarr group.

    # Args:
        # group (zarr.Group): Parent group to read from
        # name (str): Dataset name
        # Device (torch.device): Target device to load onto

    # Returns:
        # torch.Tensor: Loaded tensor

    if name not in group:
        raise KeyError(f"[ZARR] Tensor '{name}' not found int store")

    np_data = group[name][:]
    tensor = torch.tensor(np_data, device=device)

    logger.info(f"[ZARR] Read tensor '{name}' with shape {tensor.shape}")
    return tensor

    
# -------------------------------------------------------------------------------
# Tensor deletion
# -------------------------------------------------------------------------------


def delete_array(group: zarr.Group, name: str):
    # Deletes array from Zarr group.

    if name in group:
        del group[name]
        logger.info(f"[ZARR] Deleted array '{name}'")
    else:
        logger.warning(f"[ZARR] Array '{name}' not found - cannot delete")

"""

# zarr_wrapper.py       #high-level: save_model(), load_model(), etc.
"""
import os
import logging
from . import zarr_core
from . import schema_utils

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# High-level Save
# --------------------------------------------------------------------------

def save_model(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    step: int,
    path: str,
    schema=None,
    extra: dict = None
):
    # High-level save of full model state into Zarr.

    logger.info(f"[ZARR] Saving full checkpoint to: {path}")

    group = zarr_core.open_store(path, mode="w")

    # Save model parameters
    for name, param in model.state_dict().items():
        zarr_core.write_tensor(group, f"model/{name}", param)

    # Save optimizer state
    if optimizer is not None:
        opt_state = optimizer.state_dict()
        for key, value in opt_state.items():
            sub_group = group.require_group("optimizer")
            zarr_core.write_tensor(sub_group, key, serialize_dict(value))

    # Save scheduler state
    if scheduler is not None:
        sched_state = scheduler.state_dict()
        for key, value in sched_state.items():
            sub_group = group.require_group("scheduler")
            zarr_core.write_tensor(sub_group, key, serialize_dict(value))

    # Save metadata
    group.attrs["epoch"] = epoch
    group.attrs["step"] = step
    group.attrs["extra"] = extra or {}

    # Optional schema validation
    if schema:
        schema_utils.validate_schema(group, schema)


# -----------------------------------------------------------------------------
# High-level Load
# -----------------------------------------------------------------------------

def load_model(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    path: str,
    strict: bool = True
) -> dict:

    # High-level load of full model state from Zarr.

    logger.info(f"[ZARR] Loading checkpoint from: {path}")

    group = zarr_core.open_store(path, mode="r")

    # Optional schema check
    if schema:
        schema_utils.validate_schema(group, schema)

    # Load model weights
    state_dict = {}
    model_group = group.require_group("model")
    for name in model_group.array_keys():
        tensor = zarr_core.read_tensor(model_group, name)
        state_dict[name] = tensor

    model.load_state_dict(state_dict, strict=strict)
    logger.info(f"[ZARR] Model weights loaded (strict={strict})")

    # Load optimizer state
    if optimizer is not None and "optimizer" in group:
        out_group = group.require_group("optimizer")
        opt_state = {}
            for key in opt_group.array_keys():
                opt_state[key] = deserialize_dict(zarr_core.read_tensor(opt_group, key))
        optimizer.load_state_dict(opt_state)
        logger.info("[ZARR] Optimizer state loaded")

    # Load scheduler state
    if scheduler is not None and "scheduler" in group:
        sched_group = group.require_group("scheduler")
        sched_state = {}
        for key in sched_group.array_keys():
            sched_state[key] = deserialize_dict(zarr_core.read_tensor(sched_group, key))
        scheduler.load_state_dict(sched_state)
        logger.info("[ZARR] Scheduler state loaded")

    # Read metadata
    epoch = group.attrs.get("epoch", 0)
    step = group.attrs.get("step", 0)
    extra = group.attrs.get("extra", {})

    logger.info(f"[ZARR] Metadata loaded: epoch={epoch}, step={step}")

    return {
        "epoch": epoch,
        "step": step,
        "extra": extra
    }

         
# ------------------------------------------------------------------------------
# Utility: Serialize optimizer/scheduler states
# ------------------------------------------------------------------------------

def serialize_dict(obj: dict) -> torch.Tensor:
    # Safely serialize a state dict to tensor for Zarr. (make this smarter later)

    import pickle
    buf = pickle.dumps(obj)
    array = torch.tensor(list(buf), dtype=torch.uint8)
    return array

def deserialize_dict(tensor: torch.Tensor) -> dict:
    # Deserialize a state dict from tensor.

    import pickle
    buf = bytes(tensor.tolist())
    obj = pickle.loads(buf)
    return obj

"""

# schema_utils.py       # Optional: schema validation + layout checks
"""
import logging
from typing import Dict, Tuple
import zarr

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------
# Schema Definition Format
# ------------------------------------------------------------------------

# Schema format:

# {
#   "model": {
#       "weight_key_1": (shape_tuple),
#       "weight_key_2": (shape_tuple),
#        ...
#   },
#   "optimizer": {
#       "state_key_1": (shape_tuple),
#       ...
#   },
#   ...
# }


# -------------------------------------------------------------------------
# Schema Validation Core
# -------------------------------------------------------------------------

def validate_schema(group: zarr.Group, schema: Dict[str, Dict[str, Tuple]]):
    # Validates Zarr store group against schema definition.

    logger.info("[SCHEMA] Validating Zarr store against schema")

    for top_level_group, expected_keys in schema.items():
        if top_level_group not in group:
            raise ValueError(f"[SCHEMA] Missing top-level group: {top_level_group}")

            sub_group = group[top_level_group]

            for key, expected_shape in expected_keys.items():
                if key not in sub_group:
                    raise ValueError(f"[SCHEMA] Missing key '{key}' in group '{top_level_group}'")

                    actual_shape = sub_group[key].shape

                    if expected_shape is not None:
                        compare_shapes(expected_shape, actual_shape, group=top_level_group, key=key)

    logger.info("[SCHEMA] Validation passed.")

def compare_shapes(expected: Tuple, actual: Tuple, group: str, key: str):
    # Compare tensor shapes.

    if expected != actual:
        raise ValueError(
            f"[SCHEMA] Shape mismatch in '{group}/{key}': "
            f"expected {expected}, found {actual}"
        )


# ---------------------------------------------------------------------
# Optional schema builder helper
# ---------------------------------------------------------------------

def build_model_schema(model: "torch.nn.Module") -> Dict:
    # Auto-generate model weight schema from actual model.

    schema = {}
    for name, param in model.state_dict().items():
        schema[name] = tuple(param.shape)
    return {"model": schema}

"""

# chunk_tuner.py        # Optional: manual chunk tuning support
"""
import logging
from typing import Tuple, Dict

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Main chunk entry point
# ---------------------------------------------------------------------------

def get_chunk_config(
    weight_shapes: Dict[str, Tuple],
    strategy: str = "auto",
    fixed_size: int = 32768,
    per_key_override: Dict[str, Tuple] = None
) -> Dict[str, Tuple]:
    # Generates a dict mapping weight keys -> chunk shape.

    chunks = {}

    for name, shape in weight_shapes.items():
        if per_key_override and name in per_key_override:
            chunks[name] = per_key_override[name]
        elif strategy == "fixed":
            chunks[name] = fixed_chunk(shape, target_size=fixed_size)
        elif strategy == "auto":
            chunks[name] = auto_chunk(shape, target_size=fixed_size)
        elif stategy == "none":
            chunks[name] = None
        else: 
            raise ValueError(f"[CHUNK] Unknown strategy: {strategy}")

        logger.debug(f"[CHUNK] {name}: shape={shape} -> chunk={chunks[name]}")

    return chunks


# ------------------------------------------------------------------------------
# Chunking strategy: fixed
# ------------------------------------------------------------------------------

def fixed_chunk(shape: Tuple, target_size: int = 32768) -> Tuple:

    # Use a uniform fixed chunk size along the first dim only.

    if len(shape) == 0:
        return ()

    c = max(1, target_size // int(product(shape[1:])))
    return (min(c, shape[0]),) + shape[1:]


# -----------------------------------------------------------------------------
# Chunking strategy: auto (scale over dims)
# -----------------------------------------------------------------------------

def auto_chunk(shape: Tuple, target_size: int = 32768) -> Tuple:
    # Computes chuck shape by scaling evenly over dims to fit ~target elements.

    import math

    if len(shape) == 0:
        return ()

    size = product(shape)
    ratio = target_size / size

    # Scale each dimension down based on ratio
    chunk = tuple(
        max(1, int(math.ceil(dim * (ratio ** (1 / len(shape)))))
        ) for dim in shape
    )

    # Clamp to shape limits
    return tuple(min(chunk[i], shape[i]) for i in range(len(shape)))


# --------------------------------------------------------------------------
# Manual helper: build overrides easily
# --------------------------------------------------------------------------

def layerwise_override(**kwargs) -> Dict[str, Tuple]:
    # Convenience method to define manual per-key chunk overrides.

    return kwargs


# --------------------------------------------------------------------------
# Utility
# --------------------------------------------------------------------------

def product(t: Tuple) -> int:
    result = 1
    for x in t:
        result *= x
    return result

"""

# remote_utils.py       # GCS, S3, or fsspec driver config + retries (only if using fsspec for GCS/S3/Azure)
"""
import logging
import zarr
import fsspec
from .registry import register_driver

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------
# GCS driver
# ------------------------------------------------------------------------

def gcs_driver(path: str, mode: str, storage_options: dict):
    logger.info(f"[REMOTE] Opening GCS store at: {path}")
    fs = fspec.filesystem("gcs", **storage_options)
    store = zarr.FSStore(path, fs=fs)

    if mode == "w":
        logger.warning(f"[REMOTE] Removing existing GCS Zarr store: {path}")
        store.rmdir("")

    group = zarr.group(store=store, overwrite=(mode == "w"))
    return group

# --------------------------------------------------------------------------
# S3 driver
# --------------------------------------------------------------------------

def s3_driver(path: str, mode: str, storage_options: dict):
    logger.info(f"[REMOTE] Opening S3 store at: {path}")
    fs = fsspec.filesystme("s3", **storage_options)
    store = zarr.FSStore(path, fs=fs)

    if mode == "w":
        logger.warning(f"[REMOTE] Removing existing S3 Zarr store: {path}")
        store.rmdir("")

    group = zarr.group(store=store, overwrite=(mode == "w"))
    return group

# -------------------------------------------------------------------------
# Azure Blob driver
# -------------------------------------------------------------------------

def azure_driver(path: str, mode: str, storage_options: dict):
    logger.info(f"[REMOTE] Opening Azure Blob store at: {path}")
    fs = fsspec.filesystem("az", **storage_options)
    stre = zarr.FSStore(path, fs=fs)

    if mode == "w":
        logger.warning(f"[REMOTE] Removing existing Azure Zarr store: {path}")
        store.rmdir("")

    group = zarr.group(store=store, overwrite=(mode == "w"))
    return group

# -----------------------------------------------------------------------
# Register drivers at import time
# -----------------------------------------------------------------------

register_driver("gcs", gsc_driver)
register_driver("s3", s3_driver)
register_driver("azure", azure_driver)

"""

# registry.py           # Optional: zarr driver registry abstraction
"""
import zarr
import logging
from typing import Callable, Dict, Optional

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# Internal driver registry
# -----------------------------------------------------------------------

_DRIVERS: Dict[str, Callable] = {}

# -----------------------------------------------------------------------
# Public driver registration interface
# -----------------------------------------------------------------------

def register_driver(name: str, builder: Callable):
    # Register a new Zarr store driver.

    if name in _DRIVERS:
        logger.warning(f"[ZARR-REGISTRY] Driver 'name' already registered, overwriting.")
        _DRIVERS[name] = builder
        logger.info(f"[ZARR-REGISTRY] Registered store driver '{name}'")

# -----------------------------------------------------------------------
# Store creation interface
# -----------------------------------------------------------------------

def get_store(
    path: str,
    mode: str = "a",
    config: Optional[dict] = None
) -> zarr.Group:
    # Create/open Zarr store using registered backend.

    driver_name = (config or {}).get("driver", "local")
    storage_options = (config or {}).get("storage_options", {})

    if driver_name not in _DRIVERS:
        raise ValueError(f"[ZARR-REGISTRY] Unknown driver '{driver_name}'")

    builder = _DRIVERS[driver_name]
    store = builder(path, mode, storage_options)

    logger.info(f"[ZARR-REGISTRY] Opened store '{driver_name}' at '{path}' (mode={mode})")
    return store


def list_registered_drivers():
    # Print all registered drivers.
    logger.info("[ZARR-REGISTRY] Available store drivers:")
    for name in _DRIVERS.keys():
        logger.info(f" - {name}")


# ---------------------------------------------------------------------------
# Built-in default drivers
# ---------------------------------------------------------------------------

def _local_driver(path: str, mode: str, storage_options: dict) -> zarr.Group:
    store = zarr.DirectoryStore(path)
    if mode == "w":
        zarr.convenience.rmdir(store)
    return zarr.group(store=store, overwrite=(mode == "w"))


# Register default local driver immediately
register_driver("local", local_driver)

"""

# metadata_utils.py     # Versioning, timestamps, index files (if needed)
"""
import datetime
import hashlib
import subprocess
import logging
from typing import Dict, Any, Optional
import zarr

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------
# Generate full metadata entry
# ------------------------------------------------------------------------

def generate_metadata(
    config_dict: Dict,
    run_id: str,
    code_version: Optional[str] = None
) -> Dict[str, Any]:
    # Generates full metadata dict to attach to checkpoint.

    metadata = {}

    metadata["timestamp_utc"] = datetime.datetime.utcnow().isoformat()
    metadata["run_id"] = run_id
    metadata["git_commit"] = get_git_commit_hash()
    metadata["config_hash"] = hash_config(config_dict)
    metadata["code_version"] = code_version or "unknown"

    return metadata


# ------------------------------------------------------------------------
# Write metadata into Zarr store
# ------------------------------------------------------------------------

def attach_metadata(group: zarr.Group, metadata: Dict[str, Any]):
    # Attach metadata to Zarr store attributes.

    logger.info("[METADATA] Attaching metadata to store...")
    for key, value in metadata.items():
        group.attrs[f"meta_{key}"] = value
        logger.info(f" - {key}: {value}")

# ------------------------------------------------------------------------
# Read metadata back from Zarr store.
# ------------------------------------------------------------------------

def read_metadata(group: zarr.Group) -> Dict[str, Any]:
    # Reads metadata atrributes from Zarr store.

    metadata = {}
    for key, value in group.attrs.items():
        if key.startswith("meta_"):
        stripped_key = key[len("meta_"):]
        metadata[stripped_key] = value
    return metadata


# -------------------------------------------------------------------------
# Pretty-print metadata
# -------------------------------------------------------------------------

def summarize_metadata(metadata: Dict[str, Any]):
    # Logs formmated metadata summary.

    logger.info("[METADATA] Checkpoint metadata:")
    for key, value in metadata.items():
        logger.info(f"  {key}: {value}")


# ------------------------------------------------------------------------
# Git hash capture helper
# ------------------------------------------------------------------------

def get_git_commit_hash() -> str:
    # Return current Git commit has (if inside a git repo).

    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
        return commit
    expect Exception:
        return "unknown"


# --------------------------------------------------------------------------
# Config hash helper
# --------------------------------------------------------------------------

def hash_config(config_dict: Dict) -> str:
    # Compute has of full config dictionary (consistent across runs).

    import yaml
    canonical = ymal.dump(config_dict, sort_keys=True)
    hash_digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return hash_digest

"""


#    utils/tensorstore_checkpointing/

# File: __init__.py             #makes this a package
"""
from .tensorstore_wrapper import (
    save_checkpoint,
    load_checkpoint
)

from .schema_utils import (
    generate_schema,
    validate_schema_async
)

from .chunk_tuner import (
    get_chunk_config
)

from.registry import (
    register_driver,
    get_kvstore,
    list_drivers
)

from .remote_utils import (
    inject_auth_storage_options
)


"""

# File: tensorstore_core.py     #low-level async same/load logic
"""
import tensorstore as ts
import numpy as np
import torch
import asyncio
import logging
from typing import Optional, Tuple, Dict

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# Open or create TensorStore root directory
# -----------------------------------------------------------------------

async def open_store(
    path: str,
    mode: str = "r",
    storage_options: Optional[Dict] = None
) -> ts.TensorStore:
    # Opens a TensorStore root directory.

    logger.info(f"[TS] Opening store: {path} (mode={mode})")

    kvstore = {"driver": "file", "path": path}  # Default: local file driver

    # Cloud drivers override here if storage_options provided
    if storage_options:
        kvstore = storage_options

    if mode == "w":
        # Wipe existing store clean
        await ts.delete(kvstore)

    # Root TensorStore handle itself is just metadata layer
    return kvstore


# -------------------------------------------------------------------------
# Write tensor fully (blocking wrapper provided by wrapper layer)
# -------------------------------------------------------------------------

async def write_tensor(
    kvstore: Dict,
    name: str,
    tensor: torch.Tensor,
    chunks: Optional[Tuple] = None,
    compressor: Optional[Dict] = None
):

    # Write tensor to TensorStore dataset asynchronously.

    tensor_np = tensor.detach().cpu().numpy()

    spec = {
        "kvstore": kvstore,
        "path": name,
        "dtype": str(tensor_np.dtype),
        "shape": tensor_np.shape,
        "create": True,
        "delete_existing": True,
        "chunk_layout": {"chunk_shape": chunks} if chunks else {},
        "compression" compressor or {"id": "zstd", "level": 5}
    }

    logger.info(f"[TS] Creating TensorStore dataset: {name} | shape={tensor_np.shape}")

    tstore = await ts.open(spec, open=True, create=True)
    await tstore.write(tensor_np)
    logger.info(f"[TS] Write complete: {name}")


# ----------------------------------------------------------------------------
# Read tensor (returns torch.Tensor)
# ----------------------------------------------------------------------------

async def read_tensor(
    kvstore: Dict,
    name: str,
    device: torch.device = "cpu"
) -> torch.Tensor:
    # Load tensor from TensorStore dataset asynchronously.

    spec = {
        "kvstore": kvstore,
        "path": name,
        "open": True
    }

    logger.info(f"[TS] Reading TensorStore dataset: {name}")

    tstore = await ts.open(spec)
    logger.info(f"[TS] Read complete: {name} | shape={tensor.shape}")
    return tensor


# -------------------------------------------------------------------------------
# Delete tensor (optional cleanup tool)
# -------------------------------------------------------------------------------

async def delete_tensor(kvstore: Dict, name: str):
    # Delete tensor dataset.
    logger.warning(f"[TS] Deleting dataset: {name}")
    await ts.delete({"kvstore": kvstore, "path": name})


# -------------------------------------------------------------------------------
# Generate spec from tensor (helper for wrapper layer)
# -------------------------------------------------------------------------------

def tensor_to_spec(
    kvstore: Dict,
    name: str,
    tensor: torch.Tensor,
    chunks: Optional[Tuple] = None,
    compressor: Optional[Dict] = None
) -> Dict:
    # Build full TensorStore spec for this tensor.

    tensor_np = tensor.detach().cpu().numpy()

    spec = {
        "kvstore": kvstore,
        "path": name,
        "dtype": str(tensor_np.dtype),
        "shape": tensor_np.shape,
        "create": True,
        "delete_existing": True,
        "chunk_layout": {"chunk_shape": chunks} if chunks else {},
        "compression": compressor or {"id": "zstd", "level": 5}
    }
    return spec

"""

# File: tensorstore_wrapper.py  #high-level inference for versioned checkpoints
"""
import torch
import asyncio
import logging
from typing import Optional, Dict, Any

from . import tensorstore_core
from . import schema_utils
from . import chunk_tuner
from . import metadata_utils

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# High-level save 
# ----------------------------------------------------------------------------

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[Any],
    epoch: int,
    step: int,
    path: str,
    schema: Optional[Dict] = None, 
    metadata: Optional[Dict] = None,
    storage_options: Optional[Dict] = None,
    chunk_strategy: str = "auto"
)

    # Full checkpoint save interface

    state_dict = model.state_dict()
    shapes = {k: tuple(v.shape) for k, v in state_dict.items()}
    chunks = chunk_tuner.get_chunk_config(shapes, strategy=chuck_strategy)

    async def ansyc_save():
        kvstore = await tensorstore_core.open_store(path, mode="w", storage_options=storage_options)

        # Model weight
        for name, tensor in state_dict.items():
            await tensorstore_core.write_tensor(
                kvstore, f"model/{name}", tensor, chunks=chucks[name]
            )

        # Optimizer state
        if optimizer is not None:
            opt_state = optimizer.state_dict()
            opt_tensor = serialize_dict(opt_state)
            await tensorstore_core.write_tensor(kvstore, "scheduler/state", sched_tensor)

        # Store metadata directly into store root (attributes)
        meta = {
            "epoch": epoch,
            "step": step,
            **(metadata or {})
        }
        metadata_utils.attach_metadata_to_kvstore(kvstore, meta)

        # Schema validation post-write
        if schema:
            await schema_utils.validate_schema_async(kvstore, schema)

        asyncio.run(async_save())
        logger.info("[TS-WRAPPER] Checkpoint save complete")

        
# --------------------------------------------------------------------------
# High-level load
# --------------------------------------------------------------------------

def load_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[Any],
    path: str,
    schema: Optional[Dict] = None,
    storage_options: Optional[Dict] = None,
    strict: bool = True
) -> Dict:

    # Full checkpoint load interface.

    state_dict = model.state_dict()
    shapes = {k: tuple(v.shape) for k, v in state_dict.items()}

    async def async_load():
        kvstore = await tensorstore_core.open_store(path, mode="r", storage_options=storage_options)

        # Schema validation before reading
        if schema:
            await schema_utils.validate_schema_async(kvstore, schema)

        loaded_state = {}

        for name in state_dict.keys():
            tensor = await tensorstore_core.read_tensor(kvstore, f"model/{name}")
            loaded_state[name] = tensor

        model.load_state_dict(loaded_state, strict=strict)
        logger.info("[TS-WRAPPER] Model weights loaded")

        # Optimizer
        if optimizer is not None:
            opt_tensor = await tensorstore_core.read_tensor(kvstore, "optimizer/state")
            opt_state = deserialize_dict(opt_tensor)
            optimizer.load_state_dict(opt_state)
            logger.info("[TS-WRAPPER] Optimizer state loaded")

        # Scheduler
        if scheduler is not None:
            sched_tensor = await tensorstore_core.read_tensor(kvstore, "scheduler/state")
            sched_state = deserialize_dict(sched_tensor)
            scheduler.load_state_dict(sched_state)
            logger.info(f"[TS-WRAPPER] Scheduler state loaded")

        meta = metadata_utils.read_metadata_from_kvstore(kvstore)
        logger.info(f"[TS-WRAPPER] Metadata loaded: {meta}")
        return meta

    metadata = asyncio.run(async_load())
    return metadata


# -------------------------------------------------------------------------------
# (De)serialization for optimizer/scheduler
# -------------------------------------------------------------------------------

def serialize_dict(obj: dict) -> torch.Tensor:
    import pickle
    buf = pickle.dumps(obj)
    array = torch.tensor(list(buf), dtype=torch.uint8)
    return array


def deserialize_dict(tensor: torch.Tensor) -> dict:
    import pickle
    buf = bytes(tensor.tolist())
    obj = pickle.loads(buf)
    return obj    

"""

# File: schema_utils.py         #schema validation, dtype/shape optimization logic
"""
import logging
import tensorstore as ts
from typing import Dict, Tuple, Optional

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------
# Schema format:
# { group: { tensor_name: (shape_tuple, dtype_str) } }
# -------------------------------------------------------------------------

# -------------------------------------------------------------------------
# Generate schema from model state_dict
# -------------------------------------------------------------------------

def generate_schema(model: "torch.nn.Module") -> Dict:
    # Auto-generate schema from model state_dict.

    schema = {}
    for name, tensor in model.state_dict().items():
        shape = tuple(tensor.shape)
        dtype = str(tensor.detach().cpu().numpy().dtype)
        schema[name] = (shape, dtype)
    return {"model": schema}

# -------------------------------------------------------------------------
# Full async schema validator
# -------------------------------------------------------------------------

async def validate_schema_async(kvstore, schema: Dict):
    # Validates live TensorStore againt schema definition.

    logger.info("[TS-SCHEMA] Starting schema validation")

    for top_group, keys in schema.items():
        for key, (expected_shape, expected_dtype) in keys.items():
            dataset_path = f"{top_group}/{key}"

            spec = {
                "kvstore": kvstore,
                "path": dataset_path,
                "open": True
            }

            try:
                tstore = await ts.open(spec)
            except Exeption as e:
                raise ValueError(f"[TS-SCHEMA] Missing dataset '{dataset_path}': {e}")

            actual_shape = tuple(tstore.shape)
            actual_dtype = str(tstore.dtype)

            compare_shape_dtype(
                dataset_path,
                expected_shape, actual_shape,
                expected_dtype, actal_dtype,
            )

    logger.info("[TS-SCHEMA] Validation passed ✅")


# ----------------------------------------------------------------------
# Shape/dtype checker 
# ----------------------------------------------------------------------

def compare_shape_dtype(
    name: str,
    expected_shape: Tuple,
    actual_shape: Tuple,
    expected_dtype: str,
    actual_dtype: str
):
    if expected_shape != actual_shape:
        raise ValueError(
            f"[TS-SCHEMA] Shape mismatch for '{name}': expected {expected_shape}, found {actual_shape}"
        )
    if expected_dtype != actual_dtype:
        raise ValueError(
            f"[TS-SCHEMA] Dtype mismatch for '{name}' expected {expected_dtype}, found {actual_dtype}"
        )

        
# ----------------------------------------------------------------------------
# Pretty print schema (optional debug)
# ----------------------------------------------------------------------------

def log_schema(schema: Dict):
    logger.info("[TS-SCHEMA] Schema summary:")
    for group, keys in schema.items():
        for name, (shape, dtype) in keys.items():
            logger.info(f"  {group}/{name} | shape={shape} | dtype={dtype}")

"""

# File: chunk_tuner.py         #manual/per-param chuck optimization logic
"""
import math
import logging
from typing import Dict, Tuple, Optional

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Main chunk config generator
# --------------------------------------------------------------------------

def get_chunk_config(
    weight_shapes: Dict[str, Tuple],
    strategy: str = "auto",
    target_elements: int = 128*1024,
    per_key_override: Optional[Dict[str, Tuple]] = None
) -> Dict[str, Tuple]:
    # Generate chunk shapes for each tensor.

    chunks = {}

    for name, shape in weight_shapes.items():
        if per_key_override and name in per_key_override:
            chunks[name] = per_key_override[name]
        elif strategy == "fixed":
            chunks[name] = fixed_chunk(shape, target_elements)
        elif strategy == "auto":
            chunks[name] = auto_chunk(shape, target_elements)
        elif strategy == "none":
            chunks[name] = None
        else:
            raise ValueError(f"[CHUNK] Unknown strategy: {strategy}")

        logger.debug(f"[CHUNK] {name}: shape={shape} -> chunk={chunks[name]}")

    return chunks


# -----------------------------------------------------------------------
# Fixed chunk (uniform first dimension)
# -----------------------------------------------------------------------

def fixed_chunk(shape: Tuple, target_elements: int) -> Tuple:
    # Fixed chunk: try to preserve full dims, but limit first dim.

    if len(shape) == 0:
        return ()

    product_other_dims = product(shape[1:])
    first_dim_chunk = max(1, target_elements // max(1, product_other_dims))
    return (min(fisrt_dim_chunk, shape[0]),) + shape[1:]


# ------------------------------------------------------------------------
# Auto chunk (scale across all dims)
# ------------------------------------------------------------------------

def auto_chunk(shape: Tuple, target_elements: int) -> Tuple:
    # Auto chunk: scales proportionally across all dims.

    if len(shape) == 0:
        return ()

    total_elements = product(shape)
    ratio = target_elements / total_elements

    scaled = [
        max(1, int(math.ciel(dim * (ratio ** (1 / len(shape))))))
        for dim in shape
    ]
    # Clamp to tensor dimensions
    return tuple(min(scaled[i], shape[i]) for i in range(len(shape)))

# ----------------------------------------------------------------------
# Per-layer manual override helper
# ----------------------------------------------------------------------

def layerwize_override(**kwargs) -> Dict[str, Tuple]:
    # Example usage:
    #   layerwize_override(
    #       linear__weight=(128, 5),
    #       conv1__weight=(32, 3, 3, 3)
    # )
    return kwargs


# ----------------------------------------------------------------------
# Utility product
# ----------------------------------------------------------------------

def product(shape: Tuple) -> int:
    result = 1
    for dim in shape:
        result *= dim
    return result

"""

# File: registry.py             #registry for driver configs
"""
import tensorstore as ts
import logging
from typing import Callable, Dict, Optional

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------
# Internal driver registry
# ------------------------------------------------------------------------

_DRIVERS: Dict[str, Callable[[str, Optional[Dict]], Dict]] = {}

# ------------------------------------------------------------------------
# Public registration API
# ------------------------------------------------------------------------

def register_driver(name: str, builder: Callable[[str, Optional[Dict]], Dict]):
    # Register a new storage backend

    if name in _DRIVERS:
        logger.warning(f"[REGISTRY] Overwriting existing driver '{name}'")
    _DRIVERS[name] = builder
    logger.info(f"[REGISTRY] Registered TensorStore driver: '{name}'")


def get_kvstore(
    path: str,
    driver: str = "local",
    storage_options: Options[Dict] = None
) -> Dict:
    # Build kvstore spec TensorStore based on registered backend.

    if driver not in _DRIVERS:
        raise ValueError(f"[REGISTRY] Unknown driver '{driver}'")

    return _DRIVERS[driver](path, storage_options)


def list_drivers():
    logger.info("[REGISTRY] Available drivers:")
    for name in _DRIVERS.keys():
        logger.info(f"  - {name}")


# ------------------------------------------------------------------------
# Built-in driver implementations
# ------------------------------------------------------------------------

# Local filesystem
def _local_driver(path: str, storage_options: Optional[Dict]) -> Dict:
    return {"driver": "file", "path": path}

# GCS driver
def _gcs_driver(path: str, storage_options: Optional[Dict]) -> Dict:
    cfg = storage_options or {}
    return {"driver": "gcs", "path", **cfg}

# S3 driver
def _s3_driver(path: str, storage_options: Optional[Dict]) -> Dict:
    cfg = storage_options or {}
    return {"driver": "s3", "path": path, **cfg}

# Azure Blob driver
def _azure_driver(path: str, storage_options: Optional[Dict]) -> Dict:
    cfg = storage_options or {}
    return {"driver": "azure", "path": path, **cfg}


# -----------------------------------------------------------------------
# Register default drivers on import
# -----------------------------------------------------------------------

register_driver("local", _local_driver)
register_driver("gcs", _gcs_driver)
register_driver("s3", _s3_driver)
register_driver("azure", _azure_driver)

"""

# File: remote_utils.py         #auth, retries, GCS/S3 drivers, region support
"""
import os
import logging
import time
import random

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# GCS Credentials Loader
# ----------------------------------------------------------------------

def load_gcs_credentials(service_account_path: Optional[str] = None) -> dict:
    # Load GCS creadientials for TensorStore GCS driver.

    creds = {}

    if service_account_path and os.path.exists(service_account_path):
        logger.info("f{REMOTE} Using GCS service account from: {service_account_path}")
        creds["credentials"] = {"json_keyfile_dict": load_json(service_account_path)}
    elif os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        logger.info("[REMOTE] Using GOOGLE_APPLICATION_CREDENTIALS from env")
        # TensorStore GCS driver will handle environment automatically
    else:
        logger.warning("[REMOTE] No GCS credentials provided - relying on default ADC environment")

    return creds


# --------------------------------------------------------------------------
# S3 Credentials Loader
# --------------------------------------------------------------------------

def load_s3_credentials(
    access_key: Optional[str] = None,
    secrect_key: Optional[str] = None
) -> dict:
    # Load S3 credentials for TensorStore S3 driver

    creds = {}

    access_key = access_key or os.environ.get("AWS_ACCESS_KEY_ID")
    secret_key = secrect_key or os.environ.get("AWS_SECRECT_ACCESS_KEY")

    if access_key and secret_key:
        creds["access_key_id"] = access_key
        creds["sceret_access_key"] = secret_key
        logger.info("[REMOTE] Loaded S3 credentials")
    else:
        logger.warning("[REMOTE] No S3 credentials provided - relying on default AWS env")

    return creds


# -------------------------------------------------------------------------
# Helper to load JSON file
# -------------------------------------------------------------------------

def load_json(path: str) -> dict:
    import json
    with open(path, "r") as f:
        return json.load(f)


# --------------------------------------------------------------------------
# Inject storage_options for registry
# --------------------------------------------------------------------------

def inject_auth_storage_options(
    driver: str,
    service_account_path: Optional[str] = None,
    s3_access_key: Optional[str] = None,
    s3_secrect_key: Optional[str] = None
) -> dict:
    # Generates full storage_options payload for registry drivers.

    if driver == "gcs":
        return load_gcs_credentials(service_account_path)
    elif driver == "s3":
        return load_s3_credentials(s3_access_key, s3_secrect_key)
    else:
        return {}

        
# --------------------------------------------------------------------------
# Simple retry/backoff decorator
# --------------------------------------------------------------------------

def add_retry_backoff(
    max_retries: int = 5,
    base_delay: float = 0.5,
    jitter: float = 0.2
):

    # Decorator for retry-wrapped cloud I/O operations.

    def decorator(func):
        def wrapper(*args, **kwargs):
            for attempt in range(1, max_retries+1):
                try:
                    return func(*args, **kwargs)
                except Exception as eL
                    delay = base_delay * (2 ** (attempt - 1))
                    delay += random.uniform(-jitter, jitter)
                    logger.warning(f"[REMOTE-RETRY] Attempt {attempt} failed: {e} - retrying in {delay:2f} sec")
                    time.sleep(max(0, delay))
            raise RuntimeError(f"[REMOTE-RETRY] All {max_retries} attempts failed)
        return wrapper
    
    return decorator

"""

#    utils/

# File: visualiser.py --------------------------------------------------------------
"""
import os
import torch
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from einops import rearrange
import numpy as np


VISUALIZER_REGISTERY = {}

def register_visualizer(name):
    def decorator(func):
        VISUALIZER_REGISTRY[name] = func
        return func
    return decorator


class Visualizer:
    def __init__(self, config, save_dir="visuals"):
        self.cfg = config.visualizer
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def _save_image_grid(self, tensors, filename, nrow=4, normalize=True):
        grid = vutils.make_grid(tensors, nrow=nrow, normalize=normalize, scale_each=True)
        path = os.path.join(self.save_dir, filename)
        vutils.save_image(grid, path)

        
    def compare_noising(self, x0, xt, xhat, step):
        # compare original, noised, and denoised samples at a step.
        imgs = torch.cat([x0[:self.cfg.max_images],
                        xt[:self.cfg.max_images],
                        xhat[:self.cfg.max_images]], dim=0)
        self._save_image_grid(imgs, f"denoising_step{step:06}.png")

    def plot_latent_trajectory(self, latent_list, step):
        # Visualize how latent evolves over time (e.g., t=1000->0).
        imgs = torch.stack(latent_list[:self.cfg.max_images]) # [T, B, C, H, W]
        imgs = rearrange(imgs, 't b c h w -> (t b) c h w')
        self._save_image_grid(imgs, f"latent_trajectory_{step:06}.png", nrow=len(latent_list))

    def visualize_attention(self, attn_maps, step):
        # visualize attention maps per head
        for i, attn in enumerate(attn_maps): # [B, Heads, H, W]
            B, H, W = attn.shape[0], attn.shape[-2], attn.shape[-1]
            fig, axs = plt.subplots(1, B, figsize=(B * 2, 2))
            for j in range(B):
                axs[j].imshow(attn[j].mean(0).cpu().detach().numpy(), cmap="viridis")
                axs[j].axis("off")
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_dir, f"attn_layer{i}_step{step:06}.png))
            plt.close()

    def plot_timestep_embedding(self, t_embeds, step):
        # PCA projection of time embeddings.
        from sklearn.decomposition import PCA
        t_embeds = t_embeds.cpu().detach().numpy()
        pca = PCA(n_components=2)
        proj = pca.fit_transform(t_embeds)
        plt.figure(figsize=(6, 6))
        plt.scatter(proj[:, 0], proj[:, 1], c=np.arange(len(proj)), cmap='viridis', s=10)
        plt.colorbar(label="Timestep")
        plt.title("Time Embedding PCA")
        plt.savefig(os.path.join(self.save_dir, f"time_embedding_pca_{step:06}.png"))
        plt.close()

    def plot_beta_schedule(self, betas, name="schedule"):
        # Plot beta (or alpha) schedules over time.
        plt.figure()
        plt.plot(betas.cpu().numpy(), label="Beta")
        plt.title("Beta Schedule")
        plt.xlabel("Timestep")
        plt.ylabel("Beta")
        plt.legend()
        plt.savefig(os.path.join(self.save_dir, f"{name}.png"))
        plt.close()

    def plot_guidance_effects(self, samples, guidance_scales, step):
        # Visualize samples under different guidance strengths. 
        imgs = rearrange(samples, 'g b c h w -> (g b) c h w')
        self._save_image_grid(imgs, f"guidance_grid_{step:06.png}", nrow=len(guidance_scales))

    @register_visualizer("grad_flow")
    def plot_grad_flow(named_parameters, save_path="grid_flow.png"):
        # Visualize gradient magnitudes across the model layers.
        # Great for checking vanishing or exploading gradients.
        ave_grads = []
        max_grads = []
        layers = []

        for n, p in named_parameters:
            if p.requires_grad and p.grad is not None:
                layers.append(n)
                ave_grad.append(p.grad.abs().mean().item())
                max_grads.append(p.grad.abs().max().item())

        plt.figure(figsize=(12, 6))
        plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.5, color="c", label='Max Grad')
        plt.bar(np.arange(len(av_grads)), ave_grads, alpha=0.5, color="b", label='Mean Grad')
        plt.hlines(0, 0, len(ave_grads), linewidth=1.5, color="k")
        plt.xticks(range(len(ave_grads)), layers, rotation="vertical", fontsize=8)
        plt.title("Gradient Flow per Layer")
        plt.xlabel("Layers")
        plt.ylabel("Gradient Mangitude")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

        
    @register_visualizer("param_norms")
    def plot_param_norms(model, save_path="param_norms.png"):
        # Visualize L2 norms of model parameters per layer.
        # Useful for debugging abnormal parameter magnitudes.

        norms = []
        names = []

        for name, param in model.named_parameters():
            if param.requires_grad:
                norms.append(param.data.norm().item())
                names.append(name)

        plt.figure(figsize=(12, 6))
        plt.bar(np.arange(len(norms)), norms, color='orange')
        plt.xticks(range(len(norms)), names, rotation='vertical', fontsize=8)
        plt.title("Weight Norms Per Layer")
        plt.xlabel("Layer")
        plt.ylabel("L2 Norm")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

        
    @register_visualizer("latents")
    def visualize_latents(latents, step=None, save_dir="latents"):
        # Visualizes a grid of latent features (e.g., from encoder bottleneck).
        # Expects latents of shape [B, C, H, W] and saves first few channels.

        os.makedirs(save_dir, exist_ok=True)
        b, c, h, w = latents.shape
        n = min(B, c)

        for i in range(n):
            img = latents[0, i].detach().cpu().numpy()
            plt.imshow(img, cmap='viridis')
            plt.axis("off")
            plt.title(f"Latent ch {i}")
            plt.savefig(os.path.join(save_dir, f"latent_ch_{i}_step{step}.png"))
            plt.close()


def visualize_everything(
    model,
    latents=None,
    step=None,
    named_parameters=None,
    cfg=None
):
    if not cfg or not cfg.enabled:
        return

    step_str = f"{step:05d}" if step is not None else "final"
    base_kwargs = {
        "model": model,
        "latents": latents,
        "params": named_parameters,
        "step": step,
        "save_path": cfg.save_dir,
        "n_channels": cfg.latent_channels
    }

    for name in cfg.use:
        vis_fn = VISUALIZER_REGISTRY.get(name)
        if vis_fn:
            try:
                vis_fn(**base_kwargs)
            except Exception as e:
                print(f"[WARN] Failed to run visualizer '{name}': {e}")

"""

# File: memory_tools.py ------------------------------------------------------------
"""
import torch
import gc
import weakref
import logging
from typing import Optional, Dict, List, Callable
from contextlib import contextmanager

logger = logging.getLogger(__name__)

def get_memory_summary(device: Optional[torch.device] = None, detailed: bool = False) -> Dict:
    # Returns memory usage summary for the specified device.
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summary = {}

    if device.type == "cuda":
        device_inx = device.index if device.index is not None else torch.cuda.current_device()
        summary = {
            "device": f"cuda:{device_idx}",
            "allocated_MB": round(torch.cuda.memory_allocated(device_idx) / 1024**2, 2),
            "reserved_MB": round(torch.cuda.memory_reserved(device_idx) / 1024**2, 2),
            "max_allocated_MB": round(torch.cuda.max_memory_allocated(device_idx) / 1024**2, 2),
            "max_reserved_MB": round(torch.cuda.max_memory_reserved(device_idx) / 1024**2, 2),
        }
        if detailed:
            summary["cuda_summary"] = torch.cuda.memory_summary(device_idx)
    else:
        summary = {"device": str(device), "note": "No GPU memory to track on CPU."}

    return summary

    
def log_memory_usage(tag: Optional[str] = "", device: Optional[torch.device] = None, level=logging.INFO):
    # Logs current memory usage.
    summary = get_memory_summary(device)
    message = f"[MEMORY][{tag}] {summary}"
    logger.log(level, message)


def auto_clear_cache():
    # Clears unused GPU memory and forces garbage collection.
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


# -- Optional tensor allocation tracking (experimental) --

_allocated_tensors = weakref.WeakSet()

def track_tensor_allocation(tensor: torch.Tensor):
    # Adds tensor to tracking set. Usefull for debugging persistent memory leaks.
    _allowcated_tensors.add(tensor)

def get_tracked_tensor_count() -> int:
    return len(_allocated_tensors)

def print_tracked_tensor_shapes():
    print("Tracked tensor shapes:")
    for t in _allocated_tensors:
        print(t.shape, t.device, t.dtype)


# -- Memory profiling context --

@contextmanager
def enable_memory_profiling(description: str = "Memory profile", device: Optional[torch.device] = None):
    # Context manager for profiling memory usage.
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.reset_peak_memory_stats(device)
    try:
        yield
    finally:
        used = torch.cuda.max_memory_allocated(device) / 1024**2
        logger.info(f"[PROFILE] {description} - Peak memory usage: {used:.2f} MB")


# -- Multi-GPU & Future Clustering Placeholders --

def get_all_gpu_memory_summaries() -> List[Dict]:
    # Returns memory summaries across all visible CUDA devices.
    if not torch.cuda.is_available():
        return []

    return [get_memory_summary(torch.device(f"cuda:{i}")) for i in range(torch.cuda.device_count())]


def initialize_distributed_memory_hooks(strategy: str = "ddp"):
    # Placeholder for initializing memory profiling hooks for distributed training.
    logger.info(f"Initializing memory hooks for distributed strategy: {strategy}")
    # Future: Hook into DeepSpeed, FullyShardedDDP, HuggingFace Accelerate, etc.


-- Future TODOs --
- Per-layer memory stats (needs torch.fx or custom hooks)
- TensorStore offload monitoring
- Remote memory telemetry push

"""


#       utils/tests/

# File: test_all.py (extra last)

# File: test_losses.py
"""
import torch
import pytest
from trainer.losses import LOSS_REGISTRY, DiffusionLoss

class DummyConfig:
    class Training:
        num_epochs = 1
        steps_per_epoch = 100
    training = Training()


# -----------------------
# Fixtures
# -----------------------

@pytest.fixture
def dummy_inputs():
    pred = torch.randn(4, 3, 64, 64)
    target = torch.randn(4, 3, 64, 64)
    return pred, target

@pytest.fixture
def step():
    return 50  # Mid-training


# --------------------------
# Loss Registry Tests
# --------------------------

def test_loss_registry_includes_basic_losses():
    assert "mse" in LOSS_REGISTRY
    assert "l1" in LOSS_REGISTRY

def test_loss_registry_all_callable():
    for name, fn in LOSS_REFISTRY.items():
        assert callable(fn), f"{name} is not callable"


# --------------------------
# Basic Loss Tests
# --------------------------

def test_mse_loss(dummy_inputs):
    pred, target = dummy_inputs
    loss = LOSS_REGISTRY["mse"](pred, target)
    assert loss > 0
    assert loss.dim() == 0

def test_l1_loss(dummy_inputs):
    pred, target = dummy inputs
    loss = LOSS_REGISTRY["l1"](pred, target)
    assert loss > 0
    assert loss.dim() == 0


# -------------------------
# Optional Losses
# -------------------------

def test_lipis_loss(dummy_inputs):
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


# --------------------------------
# Weighting / Schedule Logic
# --------------------------------

def test_weighted_loss_combination(dummy_inputs, step):
    pred, target = dummy_inputs

    config = DummyConfig()
    config.losses = [
        {"type": "mse", "weight": 1.0},
        {"type": "l1", "start_weight": 0.0, "end_weight": 2.0, "schedule": "linear"},
    ]
    loss_fn = DiffusionLoss(config)
    out = loss_fn(pred, target, step=step)

    assert "total_loss" in out
    assert out["mse_loss"].dim() == 0
    assert out["l1_loss"].dim() == 0
    assert isinstance(out["l1_weight"], float)


# ----------------------
# Visual Mode
# ----------------------

def test_visualize_output(dummy_inputs, step):
    pred, target = dummy_inputs
    config = DummyConfig()
    config.losses = [{"type": "mse", "weight": 1.0}]

    loss_fn = DiffusionLoss(config)
    out = loss_fn(pred, target, step=step, visualize=True)

    assert "output_sample" in out
    assert "target_sample" in out
    assert out["output_sample"].shape[0] == 4

"""

# File: test_schedule.py
"""
import pytest
import torch
from model.schedule import make_beta_schedule

# 1. Validate that all supported types return tensors of correct shape and values
@pytest.mark.parametrize("schedule_type", ["linear", "cosine", "quadratic", "sigmoid"])
def test_schedule_output_shape_and_range(schedule_type):
    timesteps = 1000
    betas = make_beta_schedule(schedule_type, timesteps)

    assert isinstance(betas, torch.Tensor)
    assert betas.shape == (timesteps,)
    assert torch.all(betas > 0), "All beta values must be positive"
    assert torch.all(betas < 1), "All beta values must be < 1"
    assert betas.isfinite().all(), "Beta schedule contains NaNs or Infs"

# 2. Check monotonic increate for certain schedule types
def test_linear_schedule_is_monotonic():
    betas = make_beta_schedule("linear", 1000)
    assert torch.all(betas[1:] >= betas[:-1]), "Linear schedule should be increasing"

# 3. Edge case: single timestep
def test_single_step_schedule():
    betas = make_beta_scheudle("cosine", 1)
    assert betas.shape == (1,)
    assert 0 < betas[0] < 1

# 4. Check known fixed value at start/end for cosine schedule
def test_cosine_schedule_endpoints()
    betas = make_beta_schedule("cosine", 1000)
    assert betas[0].item() < betas[-1].item(), "Cosine schedule should increase"

# 5. Fuzz test for numeric stability
@pytest.mark.parametrize("schedule_type", ["linear", "cosine", "quadratic", "sigmoid"])
@pytest.mark.parametrize("timesteps", [10, 100, 1000, 5000])
def test_schedule_stability(schedule_type, timesteps):
    betas = make_beta_schedule(schedule_type, timesteps)
    assert betas.isfinite().all(), f"{schedule_type} with {timesteps} steps has unstable values"


"""

# File: test_time_embedding.py
"""
import pytest
import torch
from modules.time_embedding import build_time_embedding

@pytest.mark.parametrize("kind", ["sinusoidal", "learned", "fourier"])
@pytest.mark.parametrize("dim", [128, 256])
def test_embedding_output_shape(kind, dim):
    batch_size = 8
    timesteps = torch.randint(0, 1000, (batch_size,))
    emb_layer = build_time_embedding(kind=kind, dim=dim)

    emb = emb_layer(timesteps)
    assert emb.shape == (batch_size, dim), f"{kind} embedding output wrong shape"
    assert emb.isfinite().all(), f"{kind} embedding has NaNs or Infs"

def test_sinusoidal_embedding_consistency():
    from modules.time_embedding import SinusoidalTimeEmbedding
    emb1 = SinusoidalTimeEmbedding(dim=128)
    emb2 = SinusoidalTimeEmbedding(dim=128)

    t = torch.Tensor([0, 1, 2])
    out1 = emb1(t)
    out2 = emb2(t)

    # Sinusoidal is non-parametric => deterministic
    assert torch.allclose(out1, out2, atol=1e-6)

def test_learned_embedding_trains():
    from modules.time_embedding import LearnedTimeEmbedding

    emb = LearnedTimeEmbedding(max_steps=1000, dim=64)
    t = torch.randint(0, 1000, (4,))
    out = emb(t)
    loss = out.sum()
    loss.backward()

    assert t.grad is not None
    assert t.grad.shape == t.shape

"""

# File: test_resblock.py
"""
import torch
import pytest
from modules.residual_block import build_resblock, FiLMResBlock
from modules.norm_utils import get_norm
from tests.helpers.snapshot_tools import assert_tensor_close_to_snapshot


# --- Base Utilities ---

VARIANTS = ["vanilla", "film"]
NORM_TYPES = ["group", "batch", "layer"]



# ----- Shared Test Group for ResBlocks -----

class TestResBlockShared:
    @pytest.mark.parametrize("varient", VARIANTS)
    @pytest.mark.parametrize("norm_type", NORM_TYPES)
    def test_output_shape(variant, norm_type):
        x = torch.randn(4, 128, 32, 32)
        t = torch.randn(4, 512) # Time embedding
        block = build_resblock(varient=varient, in_channels=128, out_channels=128, time_dim=512, norm=norm_type)
        y = block(x, t)
        assert y.shape == x.shape, f"{varient} {norm_type} shape mismatch"

    @pytest.mark.parametrize("variant", VARIANTS)
    def test_backprop(varient):
        x = torch.randn(2, 64, 16, 16, requires_grad=True)
        t = torch.randn(2, 128, requires_grad=True)

        block = build_resblock(variant=variant, in_channels=64, out_channels=64, time_dim=128)
        y = block(x, t)
        loss = y.mean()
        loss.backward()

        assert x.grad is not None, f"{variant} input not differentiable"
        assert torch.isfinite(x.grad).all(), f"{variant} NaNs in gradient"

    @pytest.mark.parameterize("variant", VARIANTS)
    def test__cross_channel(variant):
        # Tests ResBlock that changes number of channels
        x = torch.randn(2, 64, 16, 16)
        t = torch.randn(2, 128)
        block = build_resblock(variant=variant, in_channels=64, out_channels=128, time_dim=128)
        y = block(x, t)
        assert y.shape == (2, 128, 16, 16), "Channel mismatch on upscale"

    def test_invalid_variant_raises(self):
        with pytest.raises(ValueError):
            build_resblock(variant="nonexistent", in_channels=64, out_channels=64, time_dim=128)

    def test_checkpoint_compatibility(self):
        x = torch.randn(2, 64, 16, 16, requires_grad=True)
        t = torch.randn(2, 128, requires_grad=True)

        block = build_resblock(
            variant="vanilla",
            in_channels=64,
            out_channels=64,
            time_dim=128,
            use_checkpoint=True # <-- Activate inside block
        )
        y = block(x, t)
        y.mean().backward()

        assert x.grad is not None, "No gradient with checkpointing"
        assert torch.isfinite(x.grad).all(), "NaNs with checkpointing"          

    def test_forward_hook(self):
    
        activations = []

        def hook_fn(module, input, output):
            activations.append(output.detach())

        x = torch.randn(1, 64, 16, 16)
        t = torch.randn(1, 128)

        block = build_resblock("vanilla", 64, 64, time_dim=128)
        handle = block.conv2.register_forward_hook(hook_fn)

        _ = block(x, t)
        handle.remove()
        
        assert len(activations) == 1
        assert activations[0].shape == (1, 64, 16, 16)

        
# ----- Specialized Test Group: FiLM-specific Logic -----

class TestFiLMResBlock:
    def test_film_modulation_effects(self):
        x = torch.randn(2, 64, 16, 16)
        t1 = torch.randn(2, 128)    # One time embedding
        t2 = torch.randn(2, 128) * 10.0 # Strongly scaled to cause different FiLM output.

        block = FiLMResBlock(64, 64, time_dim=128)

        out1 = block(x, t1)
        out2 = block(x, t2)

        diff = (out1 - out2).abs().mean().item()
        assert diff > 1e-2, f"FiLM modulation too weak, mean diff={diff:.4f}"

        
# ----- Normalization Test Group -----

@pytest.mark.parametrize("norm_type", NORM_TYPES)
def test_norm_layer_output_stability(self, norm_type):
    norm = get_norm(norm_type, num_channels=64)
    x = torch.randn(8, 64, 32, 32)
    y = norm(x)
    assert torch.isfinite(y).all(), f"{norm_type} norm produced NaNs"


# --- Snapshot Test ---
@pytest.mark.parametrize("variant", VARIANTS)
def test_snapshot_output_consistency(variant):
    x = torch.randn(1, 32, 16, 16)
    t = torch.randn(1, 64)

    block - build_resblock(variant, in_channels=32, out_channels=32, time_dim=64)
    out = block(x, t)

    # Compare output to a stored snapshot for regression testing
    assert_tensor_close_to_snapshot(out, f"resblock_{variant}_snapshot.pt")

# --- Visualization Test ---
@pytest.mark.parametrize("variant", VARIANTS)
def test_activation_visualization(variant):
    x = torch.randn(1, 32, 16, 16)
    t = torch.randn(1, 128)
    block = build_resblock(variant, 32, 32, time_dim=128)

    # Register intermediate hook
    activations = []
    def hook_fn(module, inp, out): activations.append(out.detach().cpu())

    hook = block.conv2.resister_forward_hook(hook_fn)
    _ = block(x, t)
    hook.remove()

    # Plot the first channel
    act = activations[0][0, 0]
    plt.imshow(act.numpy(), cmap="viridis")
    plt.title(f"{variant} ResBlock Activation")
    plt.savefig(f"tests/snapshots/{variant}_resblock_activation.png")
    plt.close()

"""

# File: test_attention_blocks.py
"""
import torch
import pytest
from modules.attention.vanilla_attention import VanillaAttention
from modules.attention.window_attention import WindowAttention
from modules.attention.flash_attention import FlashAttention
from modules.attention.registry import get_attention
from modules.attention.base_attention import BaseAttention

# ------------------------------- #
#   Shape + Backprop tests
# ------------------------------- #

class TestVanillaAttention:
    def setup_method(self):
        self.variant = "vanilla"
        self.dim = 64
        self.heads = 4
        self.dim_head = 16
        self.x = torch.randn(2, self.dim, 32, 32, requires_grad=True)

    def test_output_shape(self):
        block = build_attention(self.variant, dim=self.dim, heads=self.heads, dim_head=self.dim_head)
        out = block(self.x)
        assert out.shape == self.x.shape

    def test_backprop(self):
        block = build_attention(self.variant, dim=self.dim, heads=self.heads, dim_head=self.dim_head)
        out = block(self.x)
        out.mean().backward()
        assert self.x.grad is not None
        assert torch.isfinite(self.x.grad).all()


class TestWindowAttention:
    def setup_method(self):
        self.variant = "window"
        self.dim = 64
        self.heads = 4
        self.dim_head = 16
        self.window_size = 8
        self.x = torch.randn(2, self.dim, 32, 32, requires_grad=True)

    def test_output_shape(self):
        block = build_attention(self.variant, dim=self.dim, heads=self.heads, dim_head=self.dim_head, window_size=self.window_size)
        out = block(self.x)
        assert out.shape == self.x.shape

    def test_backprop(self):
        block = build_attention(self.variant, dim=self.dim, heads=self.heads, dim_head=self.dim_head, window_size=self.window_size)
        out = block(self.x)
        out.mean().backward()
        assert self.x.grad is not None
        assert torch.isfinite(self.x.grad).all()

    @pytest.mark.parametrize("window_size", [4, 8, 16])
    def test_divisible_window(self, window_size):
        x = torch.randn(2, 64, window_size, window_size)
        block = WindowAttention(dim=64, heads=4, dim_head=16, window_size=window_size)
        out = block(x)
        assert out.shape == x.shape


class TestFlashAttention:
    def setup_method(self):
        self.variant = "flash"
        self.dim = 64
        self.heads = 4
        self.dim_head = 16
        self.x = torch.randn(2, self.dim, 32, 32, requires_grad=True)

    @pytest.mark.skipif(not torch.cuda.is_avaliable(), reason="Flash attention requires CUDA")
    def test_output_shape(self):
        block = build_attention(self.variant, dim=self.dim, heads=self.heads, dim_head=self.dim_head).cuda()
        out = block(self.x.cuda())
        assert out.shape == self.x.shape

    @pytest.mark.skipif(not torch.cuda.is_avilable(), reason="Flash attention requires CUDA")
    def test_backprop(self):
        block = build_attention(self.variant, dim=self.dim, heads=self.heads, dim_head=self.dim_head).cuda()
        x = self.x.cuda()
        out = block(x)
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_requires_cuda(self):
        if not torch.cuda.is_available():
            with pytest.raises(RuntimeError):
                FlashAttention(dim=64, heads=4, dim_heads=16)(self.x)

# --------------------------- #
#   API, BUILD, EGDE CASES 
# --------------------------- #

class TestAttentionSystemAPI:
    def test_interface_compliance(self):
        dummy_input = torch.randn(2, 64, 32, 32)
        for cls in [VanillaAttention, WindowAttention, FlashAttention]:
            if cls is FlashAttention and not torch.cuda.is_available():
                continue
            model = cls(dim=64, heads=4, dim_head=16)
            assert isinstance(model, BaseAttention)
            assert model(dummy_input).shape == dummy_input.shape

    def test_invalid_variant(self):
        with pytest.raises(ValueError):
            build_attention("unknown", dim=64, heads=4, dim_head=16)

"""

# File: test_down_block.py
"""
import torch
import pytest
from modules.down_block import DownBlock


class TestDownBlock

    def test_shape_compatability(self):
        x = torch.randn(2, 64, 32, 32)
        t = torch.randn(2, 128)
        block = DownBlock(64, 128, time_dim=128)
        y = block(x, t)
        assert y.shape[1] == 128
        assert y.shape[2] < 32 and y.shape[3] < 32

    def test_hook_traceability(self):
        activations = []
        def hook_fn(module, input, output):
            activations.append(output)

        x = torch.randn(1, 64, 32, 32)
        t = torch.randn(1, 128)
        block = DownBlock(64, 128, time_dim=128)
        handle = block.resblock2.register_forward_hook(hook_fn)
        _ = block(x, t)
        handle.remove()
        assert len(activations) == 1

    def test_checkpointing_compatibility(self):
        x = torch.randn(2, 64, 32, 32, requires_grad=True)
        t = torch.randn(2, 128)
        block = DownBlock(64, 128, time_dim=128, use_checkpoint=True)
        y = block(x, t)
        y.mean().backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()

    def test_differentiability(self):
        x = torch.randn(2, 64, 32, 32, requires_grad=True)
        t = torch.randn(2, 128)
        block = DownBlock(64, 128, time_dim=128)
        y = block(x, t)
        loss = y.mean()
        loss.backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()

"""

# File: test_up_block.py
"""
import torch
import pytest 
from modules.up_block import UpBlock


class TestUpBlock:

    def test_shape_compatibitily(self):
        x = torch.randn(2, 128, 16, 16)
        skip = torch.randn(2, 64, 32, 32)
        t = torch.randn(2, 128)
        block = UpBlock(128, 64, time_dim=128)
        y = block(x, t, skip)
        assert y.shape == skip.shape

    def test_skip_connection_merge(self):
        x = torch.randn(2, 128, 16, 16)
        skip = torch.randn(2, 64, 32, 32)
        t = torch.randn(2, 128)
        block = UpBlock(128, 64, time_dim=128)
        out = block(x, t, skip)
        assert out.shape[2:] == skip.shape[2:]

    def test_hook_traceability(self):
        activations = []
        def hook_fn(module, input, output):
            activations.append(output)

        x = torch.randn(1, 128, 16, 16)
        skip = torch.randn(1, 64, 32, 32)
        t = torch.randn(1, 128)
        block = UpBlock(128, 64, time_dim=128)
        handle = block.resblock2.resgister_forward_hook(hook_fn)
        _ = block(x, t, skip)
        handle.remove()
        assert len(activations) == 1

    def test_checkpointing_compatibility(self):
        x = torch.randn(2, 128, 16, 16, requires_grad=True)
        skip = torch.randn(2, 128)
        t = torch.randn(2, 128)
        block = UpBlock(128, 64, time_dim=128, use_checkpoint=True)
        y = block(x, t, skip)
        y.mean().backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()

    def test_differentiability(self):
        x = torch.randn(2, 128, 16, 16, requires_grad=True)
        skip = torch.randn(2, 64, 32, 32)
        t = torch.randn(2, 128)
        block = UpBlock(128, 64, time_dim=128)
        y = block(x, t, skip)
        loss = y.mean()
        loss.backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()    

"""

# File: test_mid_block.py
"""
import torch
import pytest
from modules.mid_block import MidBlock

class TestMidBlock:

    def test_midblock_shape_preservation(self):
        x = torch.randn(2, 128, 16, 16)
        t = torch.randn(2, 128)
        block = MidBlock(128, time_dim=128)
        y = block(x, t)
        assert y.shape == x.shape, "Shape mismatch after MidBlock"

    def test_midblock_gradient_flow(self):
        x = torch.randn(2, 128, 16, 16, requires_grad=True)
        t = torch.randn(2, 128)
        block = MidBlock(128, time_dim=128)
        y = block(x, t)
        loss = y.mean()
        loss.backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()

    def test_midblock_with_checkpointing(self):
        x = torch.randn(2, 128, 16, 16, requires_grad=True)
        t = torch.randn(2, 128)
        block = MidBlock(128, time_dim=128, use_checkpoint=True)
        y = block(x, t)
        loss = y.mean()
        loss.backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()

    def test_attention_effectiveness(self):
        # Strongly modulate input to confirm attention pathway has influence
        x = torch.randn(2, 128, 16, 16)
        t = torch.randn(2, 128)
        x2 = x * 10.0
        block = MidBlock(128, time_dim=128)
        out1 = block(x, t)
        out2 = block(x2, t)
        diff = (out1 - out2).abs().mean().item()
        assert diff > 1e-2, f"Attention in MidBlock too weak or not active (diff={diff:.4f})"

    def test_hook_activation_trace(self):
        x = torch.randn(1, 128, 16, 16)
        t = torch.randn(1, 128)
        block = MidBlock(128, time_dim=128)
        activations = []

        def hook_fn(mod, inp, outp):
            activations.append(outp)

        handle = block.attn.register_forward_hook(hook_fn)
        _ = block(x, t)
        handle.remove()

        assert len(activations) == 1
        

"""

# File: test_final_head.py
"""
import torch
import pytest
from modules.final_head import FinalHead


class TestFinalHead:

    def test_output_shape(self):
        model = FinalHead(in_channels=128, out_channels=3)
        x = torch.randn(4, 128, 32, 32)
        y = model(x)
        assert y.shape == (4, 3, 32, 32), "Output shape mismatch."

    def test_backprop(self):
        model = FinalHead(in_channels=64, out_channels=3)
        x = torch.randn(2, 64, 16, 16, requires_grab=True)
        y = model(x)
        y.mean().backward()
        assert x.grad is not None, "No gradient"
        assert torch.isfinite(x.grad).all(), "Gradient contains NaNs"

    def test_zero_output_channels(self):
        with pytest.raises(ValueError):
        _ = FinalHead(in_channels=64, out_channels=0)

    def test_spatial_consistency(self):
        model = FinalHead(in_channels=32, out_channels=3)
        x = torch.randn(1, 32, 64, 64)
        y = model(x)
        assert y.shape == (1, 3, 64, 64), "Spatial size mismatch"

    def test_parameter_count(self):
        model = FinalHead(in_channels=16, out_channels=4)
        param_count = sum(p.numel() for p in model.parameters())
        expected = 16 * 4 + 4 # Weights + biases of Conv2d
        assert param_count == expected, f"Expected {expected}, got {param_count}"

    def test_output_distribution(self):
        model = FinalHead(64, 3)
        x = torch.randn(16, 64, 32, 32)
        y = model(x)
        assert torch.abs(y.mean()) < 1.0, "Mean too large, maybe uninitalized?"
        assert torch.abs(y.std()) > 0.1, "Standard deviation too small"

"""

# File: test_unet.py
"""
import torch
import pytest
from model.unet import UNet


# --------------------------
# Dummy building blocks
# --------------------------
class DummyTimeEmbed(torch.nn.Module):
    def forward(self, t): return torch.randn(t.size(0), 64, device=t.device)

class DummyBlock(torch.nn.Module):
    def forward(self, x, temb): return x, x     # DownBlock behavior

class DummyMid(torch.nn.Module):
    def forward(self, x, temb): return x

class DummyUp(torch.nn.Module):
    def forward(self, x, skip, temb): return x + skip

class DummyHead(torch.nn.Module):
    def forward(self, x): return x


# --------------------------
# Basic Functional Tests
# --------------------------
class TestUNetBasic:
    def setup_method(self):
        self.B, self.C, self.H, self.W = 2, 3, 32, 32
        self.base = 16
        self.x = torch.randn(self.B, self.C, self.H, self.W)
        self.t = torch.randint(0, 100, (self.B,))

        self.unet = UNet(
            in_channels=self.C,
            base_channels=self.base,
            time_embedding=DummyTimeEmbed(),
            downs=torch.nn.ModuleList([DummyBlock(), DummyBlock()]), mid=DummyMid(),
            ups=torch.nn.ModuleList([DummyUp(), DummyUp()]),
            final_head=DummyHead()
        )

    def test_forward_pass_runs(self):
        out = self.unet(self.x, self.t)
        assert out.shape == (self.B, self.base, self.H, self.W)

    def test_device_transfer(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.unet.to(device)
        self.x = self.x.to(device)
        self.t = self.t.to(device)
        out = self.unet(self.x, self.t)
        assert out.device == device


# ----------------------------
# Backward & Gradient Tests
# ----------------------------
class TestUNetGradients:
    def setup_method(self):
        self.B, self.C, self.H, self.W = 2, 3, 32, 32
        self.base = 16
        self.x = torch.randn(self.B, self.C, self.H, requires_grad=True)
        self.t = torch.randint(0, 1000, (self.B,))

        class GradTimeEmbed(torch.nn.Module):
            def forward(self, t): return torcj.randn(self.B, 64, requires_grad=True)

        class GradBlock(torch.nn.Module):
            def __init__(self): super().__init__()
            self.conv = torch.nn.Conv2d(self.base, self.base, 3, padding=1)
            def forward(self, x, temb): return self.conv(x), x

        class GradMid(torch.nn.Module):
            def __init__(self): super().__init__()
            self.conv = torch.nn.Conv2d(self.base, self.base, 3, padding=1)
            def forward(self, x, skip, temb): return self.conv(x + skip)

        class GradHead(torch.nn.Module):
            def __init__(self): super().__init__()
            self.out = torch.nn.Conv2d(self.base, 3, 1)
            def forward(self, x): return self.out(x)

        self.unet = UNet(
            in_channels=self.C,
            base_channels=self.base,
            time_embedding=GradTimeEmbed(),
            downs=torch.nn.ModuleList([GradBlock(), GradBlock()]),
            mid=GradMid(),
            ups=torch.nn.ModuleList([GradUp(), GradBlock()]),
            mid=GradMid(),
            ups=torch.nn.ModuleList([GradUp(), GradUp()]),
            final_head=GradHead()
        )

        def test_backward_pass_no_nans(self):
            out = self.unet(self.x, self.t)
            loss = out.mean()
            loss.backward()

            for name, param in self.unet.named_parameters():
                if param.grad in not None:
                    assert not torch.isnan(param.grad).any(), f"NaN in grad: {name}"

        def test_nonzero_grads(self):
            out = self.unet(self.x, self.t)
            loss = out.mean()
            loss.backward()

            grads = [p.grad for p in self.unet.parameters() if p.requires_grad]
            assert any(g is not None and g.abs().sum() > 0 for g in grads), "No gradients flowed"


# ------------------------------
# Optional: Hook Tracking
# ------------------------------
   
class TestUNetHooks:
    def setup_method(self):
        self.B, self.C, self.H, self.W = 2, 3, 32, 32
        self.x = torch.randn(self.B, self.C, self.H, self.W, requires_grad=True)
        self.t = torch.randint(0, 1000, (self.B,))

        
        self.unet = UNet(
            in_channels=self.C,
            base_channels=16,
            time_embedding=DummyTimeEmbed(),
            downs=torch.nn.ModuleList([DummyBlock(), DummyBlock()]),
            mid=DummyMid(),
            ups=torch.nn.ModuleList([DummyUp(), DummyUp()]),
            final_head=DummyHead()
        )

    def test_hooks_are_triggered(self):
        hits = []

        def hook_fn(mod, grad_in, grad_out):
            hits.append(mod)

        for mod in self.unet.modules():
            if any(p.requires_grad for p in mod.parameters()):
                mod.register_backward_hook(hook_fn)

        out = self.unet(self.x, self.t)
        out.mean().backward()

        assert len(hits) > 0, "No hooks triggered - check grad flow"

"""

# File: test_ddpm_ddim_edm.py
"""
import torch
import pytest

from diffusion.ddpm import DDPM
from diffusion.ddim import DDIM
from diffusion.edm import EDM

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
def model_and_config():
    cfg = load_config("configs/unet_config.yaml")
    model = build_unet_from_config(cfg).eval()
    return model, cfg


@pytest.mark.parametrize("sampler_type", ["ddpm", "ddim", "edm"])
def test_sampler_forward_pass(dummy_inputs, model_and_config, sampler_type):
    x, t, device = dummy_inputs
    model, cfg = model_and_config

    if sampler_type == "ddpm":
        sampler = DDPM(cfg.diffusion, model)
    elif sampler_type == "ddim":
        sampler == DDIM(cfg.diffusion, model)
    elif sampler_type == "edm":
        sampler = EDM(cfg.diffusion, model)
    else:
        raise ValueError("Unknown sampler type")

    with torch.no_grad()
        result = sampler.sample(batch_size=2, image_size=32, channels=3, device=device)

    assert result.shape == (2, 3, 32, 32), f"{sampler_type} output shape mismatch"
    assert result.isfinite().all(), f"{sampler_type} output contains NaNs"


def test_sampler_differentiability(dummy_inputs, model_and_config):
    # Ensure DDPM can backpropagate noise prediction if needed (used in loss).

    x, t, device = dummy_inputs
    model, cfg = model_and_config

    
    x.requires_grad = True
    ddpm = DDPM(cfg.diffusion, model)
    noise, loss = ddpm.q_sample_and_loss(x, t)      # Assuming this exists

    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_ddim_consistency(dummy_inputs, model_and_config):
    x, t, device = dummy_inputs
    model, cfg = model_and_config

    ddim = DDIM(cfg.diffusion, model)
    out1 = ddim.sample(batch_size=1, image_size=32, channels=3, device=device)
    out2 = ddim.sample(batch_size=1, image_size=32, channels=3, device=device)

    assert (out1 != out2).any(), "DDIM sampling is not stochastic by defalt. Unexpected repeat?"

    
@pytest.mark.parametrize("sampler_class", [DDPM, DDIM, EDM])
def test_sampler_instantiation(sampler_class, model_and_config):
    model, cfg = model_and_config
    sampler = sampler_class(cfg.diffusion, model)
    assert sampler is not None, f"{sampler_class.__name__} failed to instantiate"

"""

# File: test_forward_process.py
"""
import torch
import pytest
from diffusion.forward_process import ForwardProcess

BATCH = 4
CHANNELS = 3
HIGHT = WIDTH = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@pytest.mark.parametrize("schedule", ["linear", "cosine", "sigmoid"])
def test_beta_schedule_shapes(schedule):
    fp = ForwardProcess(schedule=schedule, timesteps=1000)
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
    assert torch.allclose(x1, x2), "Noised samples aren't deterministic with fixed seed"

def test_alpha_product_monotonicity():
    fp = ForwardProcess(schedule="linear", timesteps=1000)
    assert torch.all(fp.alphas_cumprod[1:] < fp.alphas_cumprod[:-1]), "Alphas should decrease over time"

def test_returned_noise_stats():
    fp = ForwardProcess(schedule="linear", timesteps=1000)
    x_start = torch.randn(BATCH, 3, 32, 32)
    t = torch.randint(0, 1000, (BATCH,))
    x_t, noise = fp.q_sample(x_start, t, return_noise=True)

    assert noise.mean().abs().item() < 0.1, "Noise mean too high"
    assert 0.8 < noise.std().item() < 1.2, "Noise std suspicious"

def test_visual_debug_compatibility():
    # Ensure this is hook compatible for visualizations.
    from utils.visualizer import visualize_noising_process
    fp = ForwardProcess(schedule="cosine", timesteps=1000)
    x_start = torch.randn(1, 3, 32, 32)
    visualize_noising_process(fp, x_start, save_path="/tmp/test_noising.gif")

"""

# File: test_sampler_utils.py
"""
import torch
import pytest
from diffusion.sampler_utils import (
    dynamic_thesholding,
    cfg_sample, 
    scheduled_guidance_scale,
    apply_self_conditioning,
    compute_snr,
    snr_weight,
    importance_sample_timesteps,
    debug_denoising,
)


# === Test: Dynamic Thresholding ===
class TestDynamicThresholding:
    def test_output_range(self):
        x = torch.randn(4, 3, 64, 64) * 0.5 # Exaggerated values
        out = dynamic_thresholding(x, percientile=0.95)
        assert out.min() >= -1.0 and out.max() <= 1.0, "Output not clamped correctly"

    def test_preserves_shape(self):
        x = torch.randn(2, 3, 32, 32)
        out = dynamic_thresholding(x)
        assert out.shape == x.shape, "Shape mismatch after dynamic thresholding"

    
# === Test: Classifier-Free Guidance ===
class TestClassifierFreeGuidance:
    class DummyModel(torch.nn.Module):
        def forward(self, x, t, cond):
            return cond + 0.1 * x   # Toy function

    def test_cfg_output_diff(self):
        model = self.DummyModel()
        x = torch.randn(2, 4)
        t = torch.tensor([10, 20])
        cond = torch.ones_like(x)
        uncond = torch.zeros_like(x)

        out = cfg_sample(model, x, t, cond, uncond, guidance_scale=2.0)
        expected = 0.0 + 2.0 * (1.0 - 0.0)
        assert torch.allclose(out.mean(), torch.tensor(expected), atol=0.2)

    def test_scheduled_guidance_values(self):
        t = torch.tensor([0, 250, 500, 750, 1000])
        scale = scheduled_guidance_scale(t, max_scale=7.0)
        assert torch.all(scaled <= 7.0) and torch.all(scaled >= 0.0)
        assert scale[0] > scale[-1], "Scheduled scale should decay over time."


# === Test: Self-Conditioning ===
class TestSelfConditioning:
    class DummyModel(torch.nn.Module):
        def forward(self, x, t, cond=None):
            return x[:, :3] # Just return the x-part (simulate trunction)

    def test_concat_behavior(self):
        model = self.DummyModel()
        x = torch.randn(2, 3, 64, 64)
        t = torch.tensor([10, 10])
        cond = None
        x_prev = torch.randn_like(x)
        out = apply_self_conditioning(model, x, t, cond=cond, x_prev=x_prev)
        assert out.shape == x.shape, "Self-conditioned output shape mismatch"

        
# === Test: SNR and Loss Weights ===
class TestSNRAndWeights:
    def test_snr_monotonicity(self):
        alphas_cumprod = torch.linspace(0.01, 0.99, 1000)
        t = torch.tensor([10, 100, 500])
        snr = compute_snr(t, alphas_cumprod)
        assert snr[0] < snr[-1], "SNR should grow with t"

    def test_snr_weight_output(self):
        alphas_cumprod = torch.linspace(0.01, 0.99, 1000)
        t = torch.randint(0, 999, (32,))
        weights = snr_weight(t, alphas_cumprod, gamma=2.0)
        assert torch.isfinite(weights).all()
        assert (weights >= 0).all()


# === Test: Curriculum Timestep Sampling === 
class TestTimestepCurriculum:
    def test_progressive_bounds(self):
        for epoch in [0, 25, 50, 75, 100]:
            t = importance_sample_timesteps(batch_size=128, epoch=epoch, max_epoch=100)
            assert t.min() >= 20 and t.max() <= 999

    def test_shape_correctness(self):
        t = importance_sample_timesteps(batch_size=64, epoch=50, max_epoch=100)
        assert t.shape == (64,)


# === Test: Debug Hook (smoke only) === 
class TestDebugHooks:
    class DummyModel(torch.nn.Module):
        def forward(self, x, t, cond=None):
            return x * 0.5

        def test_debug_prints_and_runs(self, capsys):
            x = torch.randn(1, 3, 32, 32)
            t = torch.tensor([100])
            model = self.DummyModel()
            _ = debug_denoising(model, x, t)
            captured = capsys.readouterr()
            assert "Step 100" in captured.out

"""

# File: test_train_loop.py
"""
import torch
import pytest
from trainer.train_loop import train_one_step
from model.build_unet import build_unet
from diffusion.ddpm import DDPM
from trainer.losses import build_loss_fn
from trainer.optim_utils import build_optimizer
from model.config import UNetConfig
from model.schedule import DiffusionSchedule
from types import SimpleNamespace


class DummyDataset(torch.utils.data.Dataset):
    def __getitem__(self, idx):
        x = torch.randn(3, 32, 32)
        return {"x": x}

    def __len__(self):
        return 100

def make_dummy_cfg():
    return SimpleNamespace(
        unet=UNetConfig(
            in_channels=3,
            base_channels=32,
            channel_mults=[1, 2],
            num_blocks=[1, 1],
            attention_resolutions=[],
            time_embedding="sinusoidal",
            resblock_type="vanilla",
            norm_type="group",
        ),
        schedule=SimpleNamespace(
            schedule_type="linear",
            timesteps=1000,
            beta_start=1e-4,
            beta_end=0.02,
        ),
        trainer=SimpleNamespace(
            loss_type="simple",
            optimizer="adam",
            lr=1e-4.
            ema_decay=0.995,
            amp=True,
            grad_clip=1.0,
            log_every=1,
        )
    )

@pytest.mark.parameterize("device", ["cpu", "cuda"] if torch.cuda.is_avalible() else ["cpu"])
def test_train_one_step_runs(device):
    cfg = make_dummy_cfg()
    model = build_unet(cfg.unet).to(device)
    schedule = DiffusionSchedule(cfg).to(device)
    diffusion = DDPM(schedule)
    loss_fn = build_loss_fn(cfg)
    optimizer = build_optimizer(model.parameters(), cfg.trainer)

    batch = {"x": torch.randn(4, 3, 32, 32).to(device)}
    logs = train_one_step(
        model=model,
        batch=batch,
        optimizer=optimizer,
        diffusion=diffusion,
        loss_fn=loss_fn,
        scaler=None,    # If AMP not used
        cfg=cfg,
        ema_model=None, 
        step=0
    )

    assert "loss" in logs 
    assert isinstance(logs["loss"], float), "Loss should, be a float"

def test_dummy_dataset_loading():
    loader = torch.utils.data.DataLoader(DuppyDataset(), batch_size=4)
    for batch in loader:
        assert "x" in batch and batch["x"].shape == (4, 3, 32, 32)
        break

"""

# File: test_data_pipeline.py 
"""
import os
import tempfile
import shutil
import json
import torch
from PIL import Image
from utils.data import build_loader
from utils.data.tokenizer import build_tokenizer
from omegaconf import OmegaConf
from distributed_mocks import DistributedMockContext

# --------------------------------------------------------------------------
# Helpers for generatung small synthetic datasets
# --------------------------------------------------------------------------

def create_dummy_image(path):
    img = Image.new("RGB", (64, 64), color=(255, 0, 0))
    img.save(path)

def create_dummy_basic_dataset(tmp_path)
    root = os.path.join(tmp_pth, "basic")
    os.makedirs(os.path.join(root, "cat"))
    os.makedirs(os.path.join(root, "dog"))
    create_dummy_image(os.path.join(root, "cat", "cat1.jpg"))
    create_dummy_image(os.path.join(root, "dog", "dog1.jpg"))
    return root

def create_dummy_caption_dataset(tmp_path):
    img_root = os.path.join(tmp_path, "captions/images")
    meta_file = os.path.join(tmp_path, "captions/metadata.jsonl")
    os.makedirs(img_root)
    create_dummy_image(os.path.join(img_root, "img1.jpg"))

    with open(meta_file, "w") as f:
        json.dump({"image": "img1.jpg", "caption": "A red square."}, f)
        f.write("\n")

    return img_root, meta_file


# -------------------------------------------------------------------------------
# Test: BasicImageFolder pipeline():
    tmp_path = tempfile.mkdtemp()
    try:
        root = create_dummy_basic_dataset(tmp_path)

        cfg = OmegaConf.create({
            "data": {
                "name": "BasicImageFolder",
                "path": root,
                "image_size": 64,
                "distributed": False,
                "train": {"batch_size": 2, "num_workers": 0},
            }
        })
    
        loader = build_loader(cfg, mode="train")

        for batch in loader:
            assert "image" in batch
            assert "label" in batch
            print("Basic dataset batch OK:", batch["image"].shape)
            break
            
        finally:
            shutil.rmtree(tmp_path)

            

# -------------------------------------------------------------------------------
# Test: CaptionDataset pipeline
# -------------------------------------------------------------------------------

def test_caption_dataset_pipeline():
    tmp_path = tempfile.mkdtemp()
    try:
        img_root, meta_file = create_dummy_caption_dataset(tmp_path)

        cfg = OmegaConf.create({
            "data": {
                "name": "CaptionDataset",
                "image_root": img_root,
                "metadata_file": meta_file,
                "metadata_format": "jsonl",
                "image_size": 64,
                "distributed": False,
                "train": {"batch_size": 1, "num_workers": 0},
                "tokenizer": {
                    "name": "clip",
                    "max_length": 77
                }
            }
        })

        loader = build_loader(cfg, mode="train")

        for batch in loader:
            assert "image" in batch
            assert "input_ids" in batch
            print("Caption dataset batch OK:", batch["image"].shape)
            break
            

        finally:
            shutil.rmtree(tmp_path)


    def test_basic_dataset_pipeline_distributed():
        temp_path = tempfile.mkdtemp()
        try:
            root = create_dummy_basic_dataset(tmp_path)
            cfg = OmegaConf.create({
                "data": {
                    "name": "BasicImageFolder",
                    "path": root,
                    "image_size": 64,
                    "distributed": True,
                    "train": {"batch_size": 2, "num_workers": 0},
                }
            })

            with DistributedMockContext(world_size=4) as mock:
                for simulated_rank in range(4):
                    mock.set_rank(simulated_rank)
                    loader = build_loader(cfg, mode="train")
                    batch = next(iter(loader))
                    print(f"Simulated rank {simulated_rank}: batch['image'].shape")
            
                    
        finally:
            shutil.rmtree(tmp_path)



# --------------------------------------------------------------------------
# Run tests (or integrate into pytest later)
# --------------------------------------------------------------------------

if __name__ == "__main__":
    print("Running data pipeline tests...")
    test_basic_dataset_pipeline()
    test_caption_dataset_pipeline()
    print("All data pipeline tests passed!")


"""

# File: test_run_script.py (last)

# File: test_vanilla_checkpointing.py
"""
import os
import torch
import tempfile
from utils.checkpointing import save_checkpoint, load_checkpoint
from model.unet import UNetModel
from trainer.optim_utils import build_optimizer
from trainer.ema import EMA


def get_dummy_state():
    model = UNetModel(in_channels=3, out_channels=3)
    optimizer = build_optimizer(model.parameters(), lr=1e-3)
    ema = EMA(model, decay=0.999)
    dummy_cfg = type("Dummy", (), {"checkpoint_path": None})()
    return model, optimizer, ema, dummy_cfg

    
def test_checkpoint_save_load_consistency():
    model, optimizer, ema, cfg = get_dummy_state()
    dummy_input = torch.randn(2, 3, 64, 64)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "ckpt.pt")
        save_checkpoint(model, optimizer, ema, epoch=5, step=100, loss=0.123, path=path)

        # Modify weights to verify restore
        for p in model.parameters():
            p.data += 1.0

        load_checkpoint(path, model, optimizer, ema)
        out = model(dummy_input)

        # Make sure model doesn't produce NaNs
        assert torch.isfinite(out).all(), "Model contains NaNs after load"

        
def test_checkpoint_restores_epoch_step_loss():
    model, optimizer, ema, cfg = get_dummy_state()

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "ckpt.pt")
        save_checkpoint(model, optimizer, ema, epoch=3, step=77, loss=0.42, path=path)

        meta = load_checkpoint(path, model, optimizer, ema)
        assert meta["epoch"] == 3
        assert meta["step"] == 77
        assert abs(meta["loss"] - 0.42) < 1e-6


def test_checkpoint_handles_missing_file():
    try: 
        load_checkpoint("nonexistent.pt", *get_dummy_state()[:3])
    except FileNotFoundError
        assert True
    else:
        assert False, "Expected FileNotFoundError"


def test_checkpoint_cpu_gpu_transfer():
    model, optimizer, ema, cfg = get_dummy_state()

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "ckpt.pt")
        save_checkpoint(model, optimizer, ema, epoch=1, step=1, loss=0.0, path=path)

        # Simulate loading on CPU even if saved on GPU
        cpu_model = UNetModel(in_channels=3, out_channels=3).cpu()
        cpu_optimizer = build_optimizer(cpu_model.parameters(), lr=1e-3)
        cpu_ema = EMA(cpu_model, decay=0.999)

        meta = load_checkpoint(path, cpu_model, cpu_optimizer, cpu_ema, device="cpu")
        assert isinstance(meta, dict)

"""

# File: test_zarr_checkpointing.py
"""
import os
import tempfile
import torch
import logging
import numpy as np

from utils.zarr_checkpointing import (
    zarr_wrapper,
    metadata_utils,
    schema_utils,
    chunk_tuner,
    registry
)

# Dummy toy model for full system test
class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)
        self.bn = torch.nn.BatchNorm1d(5)

    def test_full_zarr_checkpoint_roundtrip():
        logging.basicConfig(level=logging.INFO)

        # Setup dummy model, optimizer, scheduler
        model = ToyModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

        tmp_dir = tempfile.mkdtemp()
        checkpoint_path = os.path.join(tmp_dir, "zarr_test_store")

        # Build schema from model
        model_shapes = {k: tuple(v.shape) for k, v in model.state_dict().items()}
        schema = {"model": model_shapes}

        # Build chunk config (test chunk_tuner integration)
        chunk_config = chunk_tuner.get_chunk_config(
            model_shapes, strategy="auto", fixed_size=32768
    ) 

    # Generate metadata
    dummy_config = {"lr": 0.001, "optimizer:" "Adam"}
    metadata = metadata_utils.generate_metadata(
        config_dict=dummy_config,
        run_id="unit_test_run"
    )

    # Preform full save
    group = registry.get_store(checkpoint_path, mode="w")
    zarr_wrapper.save_model(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=3,
        step=500,
        path=checkpoint_path,
        schema=schema,
        extra=metadata
    )
    metadata_utils.attach_metadata(group, metadata)

    # Validate schema after write
    schema_utils.validate_schema(group, schema)

    # Read metadata back for safety
    loaded_meta = metadata_utils.read_metadata(group)
    metadata_utils.summarize_metadata(loaded_meta)

    # Create new freash model to load into
    model_reloaded = ToyModel()
    optimizer_reloaded = torch.optim.Adam(model_reloaded.parameters(), lr=0.001)
    scheduler_reloaded = torch.optim.lr_scheduler.StepLR(optimizer_reloaded, step_size=10, gamma=0.5)

    # Full load
    state = zarr_wrapper.load_model(
        model=model_reloaded,
        optimizer=optimizer_reloaded,
        scheduler=scheduler_reloaded,
        path=checkpoint_path,
        strict=True,
        schema=schema
    )

    # Validate tensor equality after roundtrip
    for name, param in model.state_dict().items():
        reloaded_param = model_reloaded.state_dict()[name]
        assert torch.allclose(param,reloaded_param, atol=1e-6), f"Mismatch in param: {name}"

    print("[TEST] Full checkpoint roundtrip suceeded ✅")

    # Cleanup
    import shutil
    shutil.rmtree(tmp_dir)

if __name__ == "__main__":
    test_full_zarr_checkpoint_roundtrip()

"""

# File: test_zarr_schema_utils.py
"""
import os
import tempfile
import zarr
import numpy as np
import pytest

from utils.zarr_checkpointing import schema_utils

# -------------------------------------------------------------------------
# Create dummy Zarr store for schema testing
# -------------------------------------------------------------------------

def create_vaild_store(tmp_path):
    store = zarr.DirectoryStore(tmp_path)
    root = zarr.group(store=store, overwrite=True)

    # Create vaild groups & arrays
    model_group = root.require_group("model")
    model_group.create_dataset("linear.weight", shape=(10, 5), dtype=np.float32)
    model_group.create_dataset("linear.bias", shape=(5,), dtype=np.float32)
    return root


# ------------------------------------------------------------------------
# Valid schema definition
# ------------------------------------------------------------------------

vaild_schema = {
    "model": {
        "linear.weight": (10, 5),
        "linear.bias": (5,)
    }
}

# -------------------------------------------------------------------------
# Test: Valid schema should pass
# -------------------------------------------------------------------------

def test_vaild_schema_passes():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = create_vaild_store(tmpdir)
        schema_utils.validate_schema(root, valid_schema)

# -------------------------------------------------------------------------
# Test: Missing group
# -------------------------------------------------------------------------

def test_missing_group_detected():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = create_vaild_store(tmpdir)
        # Delete full model group
        del root["model"]

        with pytest.raises(ValueError) as exc:
            schema_utils.validate_schema(root, valid_schema)

        assert "Missing top-level group" in str(exc.value)

        
# -------------------------------------------------------------------------
# Test: Missing key inside group
# -------------------------------------------------------------------------

def test_missing_key_dected():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = create_valid_store(tmpdir)
        del root["model"]["linear.bias"]

        with pytest.raises(ValueError) as exc:
            schema_utils.validate_schema(root, vaild_schema)

        assert "Missing key 'linear.bias' in str(exc.value)"

# -------------------------------------------------------------------------
# Test: Shape mismatch
# -------------------------------------------------------------------------

def test_shape_mismatch_detected():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = create_valid_store(tmpdir)
        # Resize one dataset to wrong shape
        root["model"]["linear.weight"].resize((8, 5))

        with pytest.raises(ValueError) as exc:
            schema_utils.validate_schema(root, valid_schema)

        assert "Shape mistmatch" un str(exc.value)

"""

# File: test_tensorstore_checkpointing.py
"""
import os
import tempfile
import torch
import asyncio

from utils.tensorstore_checkpointing import (
    tensorstore_wrapper,
    schema_utils,
    chunk_tuner,
    registry,
    metadata_utils
)


# ------------------------------------------------------------------------
# Dummy test model
# ------------------------------------------------------------------------

class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)
        self.bn = torch.nn.BatchNorm1d(5)


# ------------------------------------------------------------------------
# Full end-to-end checkpoint test
# ------------------------------------------------------------------------

def test_tensorstore_checkpoint_roundtrip():
    # Setup dummy model / optimizer / scheduler
    model = DummyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # Generate schema
    schema = schema_utils.generate_schema(model)

    # Generate metadata
    config_dict = {"lr": 0.001, "schedule": "step"}
    metadata = metadata_utils.generate_metadata(
        config_dict=config_dict,
        run_id="unit_test_run"
    )

    # Setup temporary test directory
    tmp_dir = tempfile.mkdtemp()
    checkpoint_path = os.path.join(tmp_dir, "ts_checkpoint")

    # Prepare storage backend
    kvstore = registry.get_kvstore(checkpoint_path, driver="local")

    # Save checkpoint 
    tensorstore_wrapper.save_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=3,
        step=42,
        path=checkpoint_path,
        schema=schema,
        metadata=metadata,
        storage_options=None    # Local test
    )

    # Create new fresh model to verify full reload
    model_reloaded = DummyModel()
    optimizer_reloaded = torch.optim.Adam(model_reloaded.parameters(), lr=0.001)
    scheduler_reloaded = torch.optim.lr_scheduler.StepLR(optimizer_reloaded, step_size=5, gamma=0.5)

    # Load checkpoint
    loaded_meta = tensorstore_wrapper.load_checkpoint(
        model=model_reloaded,
        optimizer=optimizer_reloaded,
        scheduler=scheduler_reloaded,
        path=checkpoint_path,
        schema=schema,
        strict=True
    )

    # Validate model weights identical after reload
    for name, orig_tensor in model.state_dict().items():
        reloaded_tensor = model_reloaded.state_dict()[name]
        assert torch.allclose(orig_tensor, reloaded_tensor, atol=1e-6), f"Tensor mismatch: {name}"

    print("[TEST] Model tensors indentcal ✅")

    # Validate metadata survived roundtrip
    assert loaded_meta["epoch"] == 3
    assert loaded_meta["step"] == 42
    print([TEST] Metadata roundtrip succeeded ✅)

    # Cleanup
    import shutil
    shutil.rmtree(tmp_dir)

if __name__ == "__main__":
    test_tensorstore_checkpoint_roundtrip()

"""

# File: test_tensorstore_schema_utils.py
"""
import os
import tempfile
import pytest
import torch
import ansyncio
import numpy as np
import tensorstore as ts

from utils.tensorstore_checkpointing import (
    tensorstore_core,
    schema_utils,
    registry
)

# -------------------------------------------------------------------------
# Create vaild TensorStore datasets for controlled schema tests
# -------------------------------------------------------------------------

async def create_valid_store(path):
    kvstore = registry.get_kvstore(path, driver="local")

    # Create model group datasets
    await tensorstore_core.write_tensor(
        kvstore, "model/linear.weight",
        tensor=torch.random(10, 5, dtype=torch.float32)
    )
    await tensorstore_core.write_tensor(
        kvstore, "model/linear.bias",
        tensor=torch.randn(5, dtype=torch.float32)
    )

    return kvstore

# ---------------------------------------------------------------------------
# Reference vaild schema definition
# ---------------------------------------------------------------------------

vaild_schema = {
    "model": {
        "linear.weight": ((10, 5), "float32"),
        "linear.bias": ((5,), "float32")
    }
}

# --------------------------------------------------------------------------
# Test: schema passes on vaild store
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_vaild_schema_passes():
    with tempfile.TemoraryDirectory() as tmpdir:
        kvstore = await create_vaild_store(tmpdir)
        await schema_utils.validate_schema_async(kvstore, vaild_schema)


# ----------------------------------------------------------------------------
# Test: missing dataset triggers failure
# ----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_missing_dataset_detected():
    with tempfile.TemporaryDirectiory() as tmpdir:
        kvstore = await create_valid_store(tmpdir)

        # Delete one dataset manually
        spec = {"kvstore": kvstore, "path": "model/linear.bias"}
        await ts.delete(spec)

        with pytest.raises(ValueError, match="Missing dataset"):
            await schema_utils.validate_schema_async(kvstore, valid_schema)

            
# -----------------------------------------------------------------------------
# Test: shape mismatch triggers failure
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_shape_mismatch_detected():
    with tempfile.TemporaryDirectory() as tmpdir:
        kvstore = await create_valid_store(tmpdir)

        # Recreate dataset with wrong shape
        spec = {
            "kvstore": kvstore,
            "path": "model/linear.weight",
            "dtype": "float32"
            "shape": [8, 5],
            "delete_existing": True
        }
        tstore = await ts.open(spec)
        await tstore.write(np.random.randn(8, 5).astype(np.float32))

        with pytest.raises(ValueError, match="Shape mismatch"):
            await schema_utils.validate_schema_async(kvstore, vaild_schema)


# ----------------------------------------------------------------------------
# Test: dtype mismatch triggers failure
# ----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_dtype_mismatch_detected():
    with tempfile.TemporaryDirectory() as tmpdir:
        kvstore = await create_vaild_store(tmpdir)

        # Recreate dataset with wrong dtype
        spec = {
            "kvstore": kvstore,
            "path": "model/linear.bias",
            "dtype": "float64",
            "shape": [5],
            "create": True,
            "delete_existing": True
        }
        tstore = await ts.open(spec)
        await tstore.write(np.random.randn(5).astype(np.float64))

        with pytest.raises(ValueError, match="Dtype mismatch"):
            await schema_utils.validate_schema_async(kvstore, vaild_schema)


"""

# File: test_device_utils.py
"""
import unittest
import torch
import os
from unittest import mock
from device_utils import (
    parse_device,
    get_defualt_device,
    get_device_count,
    summarize_device,
    autocast_context,
    distributed_initalize,
    get_local_rank,
    get_global_rank,
    get_world_size
)


class TestDeviceUtils(unittest.TestCase):
    
    def test_parse_device(self):
        d1 = parse_device("cuda:0")
        self.assertEqual(d1.type, "cuda")
        self.assertEqual(d1.index, 0)

        d2 = parse_device("cpu")
        self.assertEqual(d2.type, "cpu")

        d3 = parse_device(None)
        self.assertIsInstance(d3, torch.device)

    def test_default_device(self):
        device = get_default_device()
        self.assertIsInstance(device, torch.device)
        # We don't assert which device because that depends on machine availability.

    def test_device_count(self):
        count = get_device_count()
        self.assertGreaterEqual(count, 1)

    def test_autocast_context(self):
        device = parse_device("cpu")
        with autocast_context(device=device):
            x = torch.tensor([1.0])
            self.assertTrue(torch.is_tensor(x))

    @mock.patch.dict(os.environ, {"LOCAL_RANK": "3"})
    def test_get_local_rank_env(self):
        self.assertEqual(get_local_rank(), 3)

    @mock.patch("torch.distributed.is_available", return_value=False)
    def test_is_distributed_when_unavailable(self, mock_is_available):
        self.assertFalse(is_distributed())

    @mock.patch("torch.distributed.is_available", return_value=True)
    @mock.patch("torch.distributed.is_initialized", return_value=False)
    def test_is_distributed_when_not_initialized(self, mock_init, mock_avail):
        def.assertFalse(is_distributed())

    @mock.patch("torch.distributed.is_available", return_value=True)
    @mock.patch("torch.distributed.is_initialized", return_value=True)
    def test_is_distributed_when_initialized(self, mock_init, mock_avail):
        self.assertTrue(is_distributed())

    @mock.patch("torch.distributed.is_initialized", return_value=True)
    def test_distributed_initialize_safe_when_already_initialized(self, mock_init):
        # Should safely return without error
        distributed_initialize()

    @mock.patch.dict(os.environ, {"RANK": "1", "WORLD_SIZE": "8"})
    @mock.patch("torch.distributed.is_available", return_value=True)
    @mock.patch("torch.distributed.is_initialized", return_value=False)
    @mock.patch("torch.distributed.init_process_group")
    @mock.patch("torch.cuda.set_device")
    def test_distributed_initialize_runs(
        self, mock_set_device, mock_init_pg, mock_is_initialized, mock_is_available
    ):
        distributed_initialize()
        mock_init_pg.assert_called_once()
        mock_set_device.assert_called_once()
    
if __name__ == "__main__":
    unittest.main()

"""

# File: test_config_loading.py
"""
import pytest
import yaml
from omegaconf import OmegaConf
from model.config import UNetConfig, load_config


# ---- Test 1: YAML file parses cleanly ----
def test_yaml_file_parse(full_cfg):
    assert "model" in full_cfg
    assert full_cfg.model.base_channels > 0


# ---- Test 2: Dataclass parsing and validation ----
def test_dataclass_instantiation(parsed_cfg):
    assert isinstance(parsed_cfg.model.in_channels, int)
    assert parsed_cfg.schedule.schedule_type in ["linear", "cosine", "quadratic"]


# ---- Test 3: Invalid key raises clear error ----
def test_invalid_key_raises():
    invalid_yaml = {
    model:
        in_channels: 3
        out_channels: 3
        invalid_extra_field: 999
    }

    cfg_dict = ymal.safe_load(invalid_yaml)
    with pytest.raises(TypeError):
        UNetConfig(**cfg_dict)


# ---- Test 4: Missing required key triggers error ----
def test_missing_required_fields():
    incomplete_yaml = {
    model:
        in_channels: 3
    }
    cfg_dict = yaml.safe_load(incomplete_yaml)
    with pytest.raises(TypeError)
        UNetConfig(**cfg_dict)


# ---- Test 5: Defaults properly filled ----
def test_defaults_fill_minimal(minimal_cfg):
    assert minimal_cfg.model.in_channels == 3
    assert minimal_cfg.schedule.beta_start == 1e-4


# ---- Test 6: OmegaConf merge preserves structure ----
def test_omegaconf_merge_behavior(full_cfg):
    structurred = OmegaConf.structured(UNetConfig)
    merged = OmegaConf.merge(structured, full_cfg)
    assert hasattr(merged.model, "in_channels")
    assert hasattr(merged.schedule, "schedule_type")


# ---- Test 7: Unknown schedule type is caught later ----
def test_schedule_enum_validation():
    bad_cfg = {
        "model": {"in_channels": 3, "out_channels": 3, "base_channels": 32},
        "schedule": {"schedule_type": "INVALID", "timesteps": 100}
    }
    with pytest.raises(ValueError):
        load_config(bad_cfg)

"""

# File: test_config_itegration.py
"""
import pytest
import torch

from model.build_unet import build_unet_from_config
from diffusion.ddpm import DDPM
from model.config import UNetConfig, load_config


# ---- Test 1: Full model builds from config ----
def test_unet_builds_from_full_config(parsed_cfg):
    model = build_unet_from_config(parsed_cfg)

    # Optional: lightweight forward sanity check
    dummy_input = torch.randn(2, parsed_cfg.model.in_channels, 64, 64)
    dummy_t = torch.randint(0, parsed_cfg.schedule.timesteps, (2,))
    out = model(dummy_input, dummy_t)
    aasert out.shape == dummy_input.shape


# ---- Test 2: Forward process builds from config ----
def test_ddpm_schedule_builds_from_config(parsed_cfg):
    ddpm = DDPM(parsed_cfg.schedule)
    assert ddpm.betas.shape[0] == parsed_cfg.schedule.timesteps


# ---- Test 3: Invalid schedule types still caught here ----
    bad_cfg = {
        "model": {"in_channels": 3, "out_channels": 3, "base_channels": 32},
        "schedule": {"schedule_type": "invalid_type", "timesteps": 100}
    }
    with pytest.raises(ValueError):
        load_config(bad_cfg)


# ---- Test 4: Full end-to-end config load pipeline ----
def test_end_to_end_build(minimal_cfg)
assert model is not None
ddpm = DDPM(minimal_cfg.schedule)
assert ddpm.betas.shape[0] == minimal_cfg.schedule.timesteps

"""

# File: test_config_roundtrip.py
"""
import pytest
import os
import tempfile
from omegaconf import OmegaConf
from model.config import UNetConfig, load_config


# ---- Test 1: Config can save and reload without loss ----
def test_yaml_roundtrip(full_cfg):
    # Save to temporary file
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "saved_config.yaml")
        OmegaConf.save(full_cfg, path)

        # Reload
        reloaded_cfg = OmegaConf.load(path)

        # Assert identical structure
        assert OmegaConf.to_container(full_cfg, resolve=True) == OmegaConf.to_container(reloaded_cfg, resolve=True)


# ---- Test 2: Dataclass roundtrip from object ----
def test_dataclass_roundtrip(parsed_cfg):
    # Serialize dataclass back to YAML
    cfg_dict = OmegaConf.to_container(parsed_cfg, resolve=True)

    # Reload from dict
    reconstructed = load_config(cfg_dict)

    assert reconstructed == parsed_cfg


# ---- Test 3: Roundtrip stays valid with structured OmegaConf ----
def test_structured_omegaconf_roundtrip(full_cfg):
    structured = OmegaConf.structured(UNetConfig)
    merged = OmegaConf.merge(structured, full_cfg)

    # Serialize and reload structured config
    dumped = OmegaConf.to_yaml(merged)
    reloaded = OmegaConf.create(dumped)

    # This will test both serialization and type-structure recovery
    assert hasattr(reloaded.model, "in_channels")
    assert resloaded.model.in_channels == merged.model.in_channels

"""

# File: test_build_unet.py
"""
import torch
import pytest 
from model.build_unet import build_unet_from_config


# --------------------------------------------------------------
# Shared Fixture: Real UNet Model from Config
# --------------------------------------------------------------
@pytest.fixture(scope="module")
def model_and_config(unet_config):
    model = build_unet_from_config(unet_config)
    model.eval()
    return model, unet_config


# --------------------------------------------------------------
# Basic Model-Level Integration
# --------------------------------------------------------------
class TestUNetIntegration:
    def test_forward_pass_runs(self, model_and_config):
        model, cfg = model_and_config
        x = torch.randn(2, cfg.model.in_channels, 32, 32)
        t = torch.randint(0, 1000, (2,))
        out = model(x, t)

        assert out.shape == x.shape
        assert torch.isfinite(out).all()

        
    def test_backward_pass_runs(self, model_and_config):
        model, cfg = model_and_config
        x = torch.randn(2, cfg.model.in_channels, 32, 32, requires_grad=True)
        t = torch.randint(0, 100, (2,))
        out = model(x, t)
        loss = out.mean()
        loss.backward()

        grads = [p.grad for p in model.parameters() if p.requires_grad]
        assert any(g is not None and g.abs().sum() > 0 for g in grads), "No gradients flowed"



# -----------------------------------------------------------
# Skip Connection Logic
# -----------------------------------------------------------
class TestSkipShapeFlow:
    def test_skip_shapes_match(self, model_and_config):
        model, cfg = model_and_config
        x = torch.randn(1, cfg.model.in_channels, 64, 64)
        t = torch.randint(0, 1000, (1,))

        skip_shapes = []

        def record_skip_shapes(mod, inp, out):
            if isinstance(out, tuple):
                skip_shapes.append(out[1].shape)

        for mod in model.downs:
            mod.register_forward_hook(record_skip_shapes)

        model(x, t)

        for i, (up, skip_shape) in enumerate(zip(model.ups, reversed(skip_shapes[:-1]))):
            assert skip_shape is not None
            # Optional: check spatial size or channel count
            assert all(s > 1 for s in skip_shape[2:]), f"Invalid skip shape at layer {i}"


# --------------------------------------------------
# Attention/ResBlock Variant Coverage
# --------------------------------------------------
class TestAttentionVariantIntegration:
    @pytest.mark.parametrize("attn_type", ["none", "vanilla", "window", "flash"])
    def test_resblock_with_attention_variants_runs(self, attn_type, unet_config):
        cfg = unet_config.copy()
        cfg.attention.type = attn_type

        model = build_unet_from_config(cfg)
        model.eval()

        x = torch.randn(2, cfg.model.in_channels, 32, 32)
        t = torch.randint(0, 1000, (2,))
        out = model(x, t)

        assert out.shape == x.shape
        assert torch.isfinite(out).all()

"""

# File: conftest.py
"""
import pytest
from omegaconf import OmegaConf
from model.config import UNetConfig

import torch


# ---- Load full YAML config from disk ----
@pytest.fixture(scope="session")
def full_cfg():
    # Loads the full config from YAML as OmegaConf object.
    cfg = OmegaConf.load("configs/unet_config.yaml")
    return cfg


# ---- Provide parsed dataclass version ----
@pytest.fixture(scope="session")
def parsed_cfg(full_cfg):
    # Fully validated datacalss config object.

    # If you load OmegaConf structureed config:
    from model.config import load_config
    parsed = load_config(full_cfg)
    return parsed


# ---- Minimal config for unit-level tests ----
@pytest.fixture
def minimal_cfg
    # A minimal config that can build a toy model.
        return UNetConfig(
            model {
            "in_channels": 3,
            "out_channels": 3,
            "base_channels": 32,
            "ch_mults": [1, 2]
        },
        schedule = {
            "schedule_type": "linear",
            "timesteps": 10,
            "beta_start": 1e-4,
            "beta_end": 0.02,
        }
    )

    
# ---- Dummy input batch ----
@pytest.fixture
def dummy_batch():
    # Dummy input tensor for forward tests.
    return torch.randn(2, 3, 64, 64)


# ---- Dummy timestep tensor ----
@pytest.fixture
def dummy_timesteps():
    # Dummy timesteps for forward process testing.
    return torch.randint(0, 1000, (2,))

"""

# File: test_mock_train_batch.py
"""
import unittest 
import torch
from mock_train_batch import (
    generate_mock_image_batch,
    generate_mock_latent_batch,
    generate_mock_text_embedding,
    generate_mock_unet_inputs,
    set_mock_seed
)


class TestMockTrainBatch(unittest.TestCase):

    def test_image_batch_shapes(self):
        batch = generate_mock_image_batch(batch_size=4, channels=3, height=128, width=128)
        self.assertEqual(batch.shape, (4, 3, 128, 128))
        self.assertTrue(torch.is_tensor(batch))

    def test_latent_batch_shapes(self):
        embeddings = generate_mock_latent_batch(batch_size=2, latent_dim=8, height=16, width=16)
        self.assertEqual(embedding.shape, (5, 512))

    def test_text_embedding_shapes(self):
        embeddings = generate_mock_latent_batch(batch_size=2, latent_dim=8, height=16, width=16)
        self.assertEqual(embeddings.shape, (5, 512))

    def test_unet_inputs_shapes(self):
        inputs = generate_mock_unet_inputs(batch_size=3, image_size=256, latent_dim=4, embedding_dim=1024)
        self.assertEqual(inputs["latents"].shape, (3, 4, 32, 32))
        self.assertEqual(inputs["timesteps"].shape, (3,))
        self.assertEqual(inputs["text_embeddings"].shape, (3, 1024))

    def test_seed_reproducibility(self):
        set_mock_seed(42)
        batch1 = generate_mock_image_batch(batch_size=2)
        set_mock_seed(42)
        batch2 = generate_mock_image_batch(batch_size=2)
        self.assertTrue(torch.allclose(batch1, batch2))

    def test_dtype_and_device(self):
        device = torch.device("cpu")
        dtype = torch.float16
        batch = generate_mock_image_batch(batch_size=1, dtype=dtype, device=device)
        self.assertEqual(batch.dtype, dtype)
        self.assertEqual(batch.device, device)

    
if __name__ == "__main__":
    unittest.main()

"""

# File: test_cuda_health.py
"""
import pytest
import torch
from helpers.test_utils import controlled_test
from model.build_unet import build_unet_from_config

CATEGORY = "health"
MODULE = "cuda"

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@controlled_test(CATEGORY, MODULE)
def test_unet_cuda_forward_no_nan(test_config, unet_config):
    # Basic CUDA smoke test: forward pass runs, no NaNs/Infs.
    model = build_unet_from_config(unet_config).cuda()
    model.eval()

    dummy_input = torch.randn(2, unet_config.model.in_channels, 64, 64, device="cuda")
    with torch.no_grad():
        out = model(dummy_input)

    assert torch.isfinite(out).all(), "Non-finite values in output (NaN/Inf)"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@controlled_test(CATEGORY, MODULE)
def test_unet_amp_autocast_forward(test_config, unet_config):
    # test_unet_amp_autocast_forward(test_config, unet_config):
    model = build_unet_from_config(unet_config).cuda()
    model.eval()

    dummy_input = torch.randn(2, unet_config.model.in_channels, 64, 64, device="cuda")
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        with torch.no_grad():
            out = model(dummy_input)

    assert torch.isfinite(out).all(), "AMP output contains NaN/Inf" 


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@controlled_test(CATEGORY, MODULE)
def test_unet_output_consistency_vs_cpu(test_config, unet_config):
    # Compare CPU and CUDA outputs (sanity, not exact match).
    model_cpu = build_unet_from_config(unet_config).cpu().eval()
    model_gpu = build_unet_from_config(unet_config).cuda().eval()

    input_cpu = torch.randn(1, unet_config.model.in_channels, 64, 64)
    input_gpu = input_cpu_clone().cuda()

    with torch.no_grad():
        out_cpu = model_cpu(input_cpu)
        out_gpu = model_gpu(input_gpu).cpu()

    torch.testing.assert_close(out_cpu.float(), out_gpu.float(), rtol=1e-3, atol=1e-3)  
      


"""

#           / helpers/

# File: __init__.py     # Empty
"""
# Unified export point for all helper modules

# -------------- Dummy data generation ----------------
from .dummy_data import DummyDataFactory

# --------------- Testing utilities -------------------
from .test_utils import (
    assert_shapes_math,
    assert_close,
    assert_no_nans,
    assert_no_infs,
    count_parameters,
    count_layers,
    attach_activation_hook,
    fix_torch_seed,
    SimpleTimer
)

# -------------- Device utilities ---------------------
from .device_utils import (
    parse_device,
    get_default_device,
    get_device_count,
    distributed_initialize,
    is_distributed,
    get_global_rank,
    get_local_rank,
    get_world_size,
    summarize_device,
    autocast_context
)

# -------------- Reproducibility control ------------------
from .reproducibility import (
    set_seed,
    get_current_seed,
    log.reproducibility_settings,
    warn_about_nondeterminism
)


# ----------- Serialization utilities -----------------
from .serialization_utils import (
    atomic_save,
    save_checkpoint,
    load_checkpoint,
    load_model_weights_only,
    summarize_checkpoints
)

# -------------- Mock config generation -------------------
from .mock_configs import (
    create_mock_config,
    create_large_mock_config
)

# ------------------ Mock models ------------------------
from .fake_model import (
    build_fake_model_for_loss_tests,
    build_fake_model_for_sampler_tests
)

# -------------------- Failure injection ----------------------
from .failure_injection import FailureInjector

# ----------------- Corruption injection -----------------------
from .data_corruption_injector import (
    inject_broken_images,
    inject_missing_images,
    inject_broken_captions
)

# ---------------- Monketpatch system -----------------------
from .mockeypatches import (
    safe_patcher,
    build_patch_manager
)

# ---------------- Snapshot testing -------------------
from .snapshot_tools import (
    assert_tensor_close_to_snapshot
)

# --------------- Snythetic metrics ------------------
from .metrics_synthetic import (
    generate_synthetic_loss_curve,
    generate_synthetic_accuracy_curve,
    generate_synthetic_gradient_norms,
    generate_synthetic_learning_rate_schedule,
    generate_synthetic_timing_profile
)

# -------------- Benchmark tools -------------------
from .benchmark_utils import (
    benchmark_forward_pass,
    benchmark_backward_pass,
    estimate_flops,
    log_model_size,
    log_memory_snapshot,
    benchmark_summary
)


"""

# File: dummy_data.py
"""
import torch
import numpy as np
import random

class DummyDataFactory:
# Fully configurable sythetic data generator for unit tests and debugging. 

    def __init__(self,
                batch_size=2,
                channels=3,
                height=64,
                width=64,
                device="cpu",
                seed=None,
                dtype=torch.float32):
        self.batch_size = batch_size
        self.channels = channels
        self.height = height
        self.width = width
        self.device = device
        self.dtype = dtype

        if seed is not None:
            self._set_seed(seed)


    def _set_seed(self, seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    def random_images(self, low=-1.0, high=1.0):
        return (low + (high - low) * torch.rand(
            self.batch_size, self.channels, self.height, self.width,
            device=self.device, dtype=self.dtype
        ))

    def random_timesteps(self, max_timesteps=1000):
        return torch.randint(0, max_timesteps, (self.batch_size,), device=self.device)

    def random_noise(self):
        return torch.randn(self.batch_size, self.channels, self.height, self.width, device=self.device, dtype=self.dtype)

    def random_noised_input(self):
        x_start = self.random_images()
        noise = self.random_noise()
        return x_start, noise

    def conditioning(self, embed_dim=512):
        return torch.randn(self.batch, embed_dim, device=self.device, dtype=self.dtype)

    def classifier_labels(self, num_classes=1000):
        return torch.randint(0, vocab_size, (self.batch_size, seq_len), device=self.device)

    def tokenized_prompts(self, seq_len=77, vocab_size=32000):
        return torch.randint(0, vocab_size, (self.batch_size, seq_len), device=self.device)

    def latents(self, latent_channels=4, downsample=8):
        return torch.randn(
            self.batch_size, latent_channels,
            self.height // downsample, self.width // downsample,
            device=self.device, dtype=self.dtype
        )

    def deterministic_noise(self, noise_seed=123):
        # Allows determinsitc noise for perfect repeatable tests.

        g = torch.Generator(device=self.device).manual_seed(noise_seed)
        return torch.randn(self.batch_size, self.channels, self.height, self.width, device=self.device, generator=g, dtype=self.dtype)

    def mixed_precision_images(self):
        # Simulate mixed-precision forward tests.

        return self.random_images().half()

    def curriculum_timesteps(self, current_epoch, max_epochs, min_t=10, max_t=1000):
        # Simulate curriculum sampling schedule for timestep distributions.
        limit_t = int(min_t + (max_t - min_t) * min(current_epoch / max_epochs, 1.0))
        return torch.randint(min_t, limit_t + 1, (self.batch_size,), device=self.device)

"""

# File: data_corruption_injector.py
"""
import os
import random

def inject_broken_images(image_dir, num_corrupt=1):
    # Create intentionally broken image files.

    images = [
        f for f in os.listdir(image_dir)
        if f.endswith(".jpg") or f.endswith(".png")
    ]
    to_corrupt = random.sample(images, min(num_corrupt, len(images)))

    for img in to_corrupt:
        path = os.path.join(image_dir, img)
        with open(path, "wb") as f:
            f.write(b"this is not a real image file")
        print(f"Injected broken image: {path}")

def inject_missing_images(image_dir, num_remove=1):
    # Delete files to simulate missing data.

    images = [
        f for f in os.listdir(image_dir)
        if f.endswith(".jpg") or f.endswith(".png")
    ]
    to_remove = random.sample(images, min(num_remove, len(images)))

    for img in to_remove:
        path = os.path.join(image_dir, img)
        os.remove(path)
        print(f"Injected missing image: {path}")

def inject_broken_captions(metadata_file, num_corrupt=1):
    # Corrupt captions inside JSONL metadata.
    lines = []
    with open(metadata_file, "r") as f:
        lines = f.readlines()

    corrupted_indices = random.sample(range(len(lines)), min(num_corrupt, len(lines)))

    for idx in corrupted_indices:
        lines[idx] = '{"image": "img1.jpg", "caption": ""}\n'   # empty caption

    with open(metadata_file, "w") as f:
        f.writelines(lines)
    print(f"Injected {len(corrupted_indices)} broken captions")

"""

# File: mock_configs.py (used)
"""
from model.config import UNetConfig

# ---- Minimal dummy config for unit tests ----
def create_mock_config():
    # Returns a minimal vaild UNetConfig object.

    return UNetConfig(
        model = {
            "in_channels": 3,
            "out_channels": 3,
            "base_channels": 32,
            "ch_mults": [1, 2],
        },
        time_embedding = {
            "kind": "sinusoidal",
            "dim": 128
        },
        resblock = {
            "kind": "vanilla",
            "norm": "group"
        },
        attention = {
            "kind": "vanilla",
            "start_layer": 1
        },
        schedule = {
            "schedule_type": "linear",
            "timesteps": 10,
            "beta_start": 1e-4,
            "beta_end": 0.02
        },
        optim = {
            "lr": 1e-4,
            "weight_decay": 0.0
        },
        training = {
            "batch_size": 2,
            "grad_clip": 1.0,
            "num_epochs": 1,
            "gradient_checkpointing": False,
            "amp": False
        }
    )


# ---- Variant for bigger model tests ----
def create_large_mock_config():
    # Returns a larger but still fast-to-test config.
    return UNetConfig(
        model = {
            "in_channels": 3,
            "out_channels": 3,
            "base_channels": 64,
            "ch_mults": [1, 2, 4],
        },
        time_embedding = {
            "kind": "sinusoidal",
            "dim": 256
        },
        resblock = {
            "kind": "film",
            "norm": "batch"
        },
        attention = {
            "kind": "window",
            "start_layer": 2
        }, 
        schedule = {
            "schedule_type": "cosine",
            "timesteps": 20,
            "beta_start": 1e-4,
            "beta_end": 0.02,
        },
        optim = {
            "lr": 5e-5,
            "weight_decay": 0.01
        },
        training = {
            "batch_size": 2,
            "grad_clip": 1.0,
            "num_epochs": 2,
            "gradient_checkpointing": True,
            "amp": True
        }
    )

"""

# File: fake_model.py
"""
import torch
import torch.nn as nn

class FakeNoisePredictor(nn.Module):
    # A fully mockable noise predictor model.

    def __init__(self, channels=3, constant_value=0.0, scale_t_embedding=False):
        super().__init__()
        self.constant_value = constant_value
        self.channels = channels
        self.scale_t_embedding = scale_t_embedding

    def forward(self, x, t, *args, **kwargs):
        # Simulates model forward: predicts noise given x_t and timestep t. 
        # 'x' is (B, C, H, W)
        # 't' is (B,) timesteps
        # If 'scale_t_embedding' is True, injects deterministic timestep scaling.

        B, C, H, W = x.shape
        out = torch.full((B, self.channels, H, W), self.constant_value, device=x.device, dtype=x.dtype)

        if self.scale_t_embedding:
            # Inject simple dependency on timestep for testing samplers
            scale = (t.float().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) / 1000.0)
            out += scale

        return out


class NoisyIdentityModel(nn.Module):
    # Another toy model: outputs slightly noised inputs instead of true prediction. 

    def __init__(self, noise_scale=0.01):
        super().__init__()
        self.noise_scale = noise_scale

    def forward(self, x, t, *args, **kwargs):
        noise = torch.randn_like(x) * self.noise_scale
        return x + noise


# ---- Factory methods for easy instantiation ----

def build_fake_model_for_loss_tests():
    # Cheap pure-zero noise prediction for testing loss functions. 
    return FakeNoisePredictor(constant_value=0.0)

def build_fake_model_for_sampler_tests():
    # Predicts small positive outputs depending on timestep, simulating basic learning.
    return FakeNoisePredictor(constant_value=0.1, scale_t_embedding=True)


"""

# File: test_utils.py
"""
import torch


# ---- General tensor assertions ----

def assert_shapes_math(a: torch.Tensor, b: torch.Tensor):
    assert a.shape == b.shape, f"Shape mismatch: {a.shape} vs {b.shape}"

def assert_close(a: torch.Tensor, b: torch.Tensor, atol=1e-5, rtol=1e-5):
    if not torch.allclose(a, b, atol=atol, rtol=rtol):
        max_diff = (a - b).abs().max().item()
        raise AssertionError(f"Tensors differ by max diff {max_diff:.6f}")

def assert_no_nans(tensor: torch.Tensor):
    if torch.isnan(tensor).any():
        raise AssertionError("Tensor contains NaNs")

def assert_no_infs(tensor: torch.Tensor):
    if torch.isinf(tensor).any():
        raise AssertionError("Tensor contains infs")


# ---- Activation inspection ----

def count_parameters(module):
    return sum)p.numel() for p in module.parameters() if p.requires_grad)

def count_layers(module):
    return sum(1 for _ in module.modules())

def attach_activation_hook(module, activations_list):
    # Appends output of a module into the provided list during forward pass. 

    def hook_fn(_, __, output):
        activations_list.append(output.detach())
    handle = module.register_forward_hook(hook_fn)
    return handle   # caller must handle cleanup


# ---- Controlled random see for tests ----

def fix_torch_seed(seed: int = 42):
    torch.manual_seed(seed)


# ---- Cheap runtime benchmark ----

import time

class SimpleTimer:
    # Lightweight timer context manager. 

    def __init__(self, name=""):
        self.name = name

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        elpased = time.perf_counter() - self.start
        print(f"[TIMER] {self.name}: {elapsed:.4f} sec")

"""

# File: monkeypatches.py
"""
import sys
import torch
import logging
import functools
import importlib
import yaml
from types import ModuleType
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------
# -------------------------- SafeMonkeyPatcher Core System -------------------
# ----------------------------------------------------------------------------

class SafeMonkeyPatcher:
    # Safely applies and tracks monkeypatches with full reversibility.

    def __init__(self):
        self._originals: Dict[str, Callable] = {}

    def patch(self, module: str, attribute: str):
        # Decorator for patching functions or methods safely. 

        def decorator(new_func):
            full_key = f"{module}.{attribute}"
            target_module = importlib.import_module(module)
            original_func = getattr(target_module, attribute)

            if full_key in self._originals:
                logger.warning(f"[SAFEPATCH] {full_key} already patch; skipping double patch")
                return new_func     # skip double patch

            self._originals[full_key] = original_func

            @functools.wraps(new_func)
            def wrapped(*args, **kwargs):
                return new_func(original_func, *args, **kwargs)

                setattr(target_module, attribute, wrapped)
                logger.info(f"[SAFEPATCH] Patched {full_key}")
                return new_func

            return decorator
        
        def undo_patch(self, module: str, attribute: str):
            # Revert a single patch. 
            full_key = f"{module}.{attribute}"
            if full_key not in self._originals:
                logger.warning(f"[SAFEPATCH] No patch found for {full_key}")
                return
            target_module = importlib.import_module(module)
            setattr(target_module, attribute, self.originals[full_key])
            logger.info(f"[SAFEPATCH] Reverted patch for {full_key}")
            del self.originals[full_key]

    def undo_all(self):
        # Revert all applied patches.

        for full_key in list(self._originals.keys()):
            module, attribute = full_key.split(".")
            self.undo_patch(module, attribute)


    def lost_patches(self):    
        logger.info("[SAFEPATCH] Current active patches:")
        for full_key in self._originals:
            logger.info(f" - {full_key}")


safe_patcher = SafeMonkeyPatcher()


# -------------------------------------------------------------------------------------
# ----------------------- PatchManager System -----------------------------------------
# -------------------------------------------------------------------------------------

class Patch:
    # Represents a single monkeypatch entry. 

    def __init__(
        self,
        module: str,
        name: str,
        apply_func: Callable,
        verson_getter: Optional[Callable[[], str]] = None,
        min_version: Optional[str] = None,
        notes: str = ""
    ):
        self.module = module
        self.name = name
        self.apply_func = apply_func
        self.version_getter = version_getter
        self.min_verison = min_version
        self.notes = notes
        self.applied = False

    def should_apply(self) -> bool:
        if not self.version_getter or not self.min_version:
            return True
        current_version = self.version_getter()
        return current_version < self.min_version

    def apply(self):
        if self.should_apply():
            self.apply_func()
            self.applied = True
            logger.info(f"[PATCH] Applied {self.module}.{self.name} ({self.notes})")
        else:
            logger.info(f"[PATCH] Skipped {self.module}.{self.name} (version OK)")


            
class PatchGroup:
    # Groups multiple patches under a logical subsystem. 

    def __init__(self, name: str):
        self.name = name
        self.patches: Dict[Patch] = []

    def register(self, patch: Patch):
        self.patches[patch.name] = patch

    def apply_all(self, patch_overrides: Optional[Dict[str, bool]] = None):
        logger.info(f"[PATCH] Applying patch group '{self.name}'")
        for patch_name, patch in self.patches.items():
            should_run = True
            if patch_overrides and patch_name in patch_overrides:
                should_run = patch_overrides[patch_name]
            if should_run:
                patch.apply()
            else:
                logger.info(f"[PATCH] Skipped {self.name}.{patch.name} (disabled via config)")

        
class PatchManager:
    # Central controller for all patch groups. 

    def __init__(self):
        self.groups: Dict[str, PatchGroup] = {}

    def register_group(self, group: PatchGroup):
        self.groupds[group.name] = group

    def apply(self, enabled_groups: Optional[List[str]] = None):
        enabled_groups = config.get("enabled_patch_groups", self.groups.keys())
        patch_overrides = config.get("patch_overrides", {})

        logger.info("[PATCH] Starting patch application")

        for group_name in enabled_groups:
            group = self.groups.get(group_name)
            if group:
                overrides = patch_overrides.get(group_name, {})
                group.apply_all(overrides)
            else:
                logger.warning(f"[PATCH] Patch group '{group_name}' not found")

        def summary(self):
            logger.info("[PATCH] Applied patches summary:")
            for group in self.groups.values():
                for patch in group.patches.values():
                    if patch.applied:
                        logger.info(f" - {group.name}.{patch.module}.{patch.name} ({patch.notes})")



# -------------------------------------------------------------------------------
# -------------------- Version Helpers -------------------------------------
# ---------------------------------------------------------------------------------


def get_version(module_name: str) -> str:
    try: 
        mod = importlib.import_module(module_name)
        return getattr(mod, "__version__", "0.0.0")
    except ImportError:
        return "0.0.0"


# ----------------------------------------------------------------------------
# ----------------------- Example Patch Definitions --------------------------
# ----------------------------------------------------------------------------

def build_patch_manager() -> PatchManager:
    pm = PatchManager()

    # -------- Torch group ---------
    torch_group = PatchGroup("torch")

    # Example safe monkeypatch using SafeMonkeyPatcher (just as exapmle)
    @safe_patcher.patch("torch", "inference_mode")
    def patch_inference_mode(original, *args, **kwargs):
        logger.debug("[SAFEPATCH] torch.inference_mode used")
        return original(*args, **kwargs)

    torch_group.register(Patch(
        module="torch",
        name="inference_mode_logger",
        apply_func=lambda: None,    # Patch already applied above
        notes="Log calls to inference_mode via SafeMonkeyPatcher"
    ))

    # Example version-guarded patch
    def dummy_bugfix():
        logger.info("[PATCH] Dummy bugfix applied")

    torch_group.register(Patch(
        module="torch",
        name="dummy_bugfix",
        apply_func=dummy_bugfix,
        version_getter=lambda: torch.__version__,
        min_version="2.1.0",
        notes="Dummy version-guarded patch"
    ))

    pm.register_group(torch_group)

    # -------- Deepspeed group --------
    deepspeed_group = PatchGroup("deepspeed")

    try:
        import deepspeed

        def dummy_ds_patch():
            logger.info("[PATCH] Deepspeed dummy patch applied")

        deepspeed_group.register(Patch(
            module="deepspeed",
            name="dummy_patch",
            apply_func=dummy_ds_patch,
            notes="Example deepspeed patch"
        ))

    except ImportError:
        logger.warning("[PATCH] Deepspeed not installed - skipping deepspeed patches")

    pm.register_group(deepspeed_group)

    return pm


# ---------------------------------------------------------------------------------------
# ----------------------------------- Config Loader -------------------------------------
# ---------------------------------------------------------------------------------------

def load_patch_config(path: str) -> Dict:
    try:
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
            logger.info(f"[PATCH] Loaded patch config from {path}")
            return config
        except Exception as e:
            logger.warning(f"[PATCH] Failed to load patch config: {e}")
            return {}   # fallback to defaults.

            
# --------------------------------------------------------------------------------------
# --------------------------- Repo Startup ---------------------------------------------
# --------------------------------------------------------------------------------------

# Example runtime startup:

# from monkeypatches import build_patch_manager, load_patch_config, safe_patcher

# Load config:
# patch_config = load_patch_config("patch_config.yaml")

# Build patch manager:
# patch_manager = build_patch_manager()

# Apply patches based on config
# patch_manager.apply(patch_config)

# Summary:
# patch manager.summary()
# safe_patcher.list_patches()

"""

# File: mock_train_batch.py
"""
import torch
import numpy as np
import yaml
from typing import Optional, Tuple


# ----------------------------------------------------------------------------
# Core random seed control (optional for reproducibility)
# ----------------------------------------------------------------------------

def set_mock_seed(seed: int):
    # Set random seed for reproducibility of mock data.
    torch.manual_seed(seed)
    np.random.seed(seed)


# ----------------------------------------------------------------------------
# Image batch generator (primary)
# ----------------------------------------------------------------------------

def generate_mock_image_batch(
    batch_size: int = 8,
    channels: int = 3,
    height: int = 256,
    width: int = 256,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    value_range: Tuple[float, float] = (0.0, 1.0)
) -> torch.Tensor:

    # Generate a mock image batch of random values.
    device = device or torch.device("cpu")
    low, high = value_range
    data = (high - low) * torch.rand(batch_size, channels, width, device=device, dtype=dtype) + low
    return data


# ----------------------------------------------------------------------------
# Latent batch generator (for diffusion latents, VAE, etc)
# ----------------------------------------------------------------------------

def generate_mock_latent_batch(
    batch_size: int = 8,
    latent_dim: int = 4,
    height: int = 32,
    width: int = 32,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:

    # Generate a mock latent batch (e.g. VAE latent space).
    device = device pr torch.device("cpu")
    data = torch.randn(batch_size, latent_dim, height, width, device=device, dtype=dtype)
    return data


# ----------------------------------------------------------------------------
# Mock text embedding generator (for conditioning inputs)
# ----------------------------------------------------------------------------

def generate_mock_text_embedding(
    batch_size: int = 8,
    embedding_dim: int = 768,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:

    # Generate a mock test embedding batch (e.g. CLIP text encoder output).

    device = device or torch.device("cpu")
    data = torch.randn(batch_size, embedding_dim, device=device, dtype=dtype)
    return data



# ----------------------------------------------------------------------------
# UNet-style combined mock input generator (for diffusion testing)
# ----------------------------------------------------------------------------

def generate_mock_unet_inputs(
    batch_size: int = 8,
    image_size: int = 256,
    latent_dim: int = 4,
    embedding_dim: int = 768,
    device: Optional[torch.deivce] = None
) -> dict:
    # Generate full mock input set for a diffusion U-Net forward pass.
    device = device or torch.device("cpu")

    latents = generate_mock_latent_batch(
        batch_size=batch_size,
        latent_dim=latent_dim,
        height=image_size//8,
        width=image_size//8,
        device=device,
        dtype=dtype
    )
    timesteps = torch.randint(0, 1000, (batch_size,), device=device, dtype=torch.int64)
    text_embeddings = generate_mock_text_embedding(
        batch_size=batch_size,
        embedding_dim=embedding_dim,
        device=device,
        dtype=dtype
    )

    return {
        "latents": latents,
        "timesteps": timesteps,
        "text_embeddings": text_embeddings
    }

    
# ------------------------------------------------------------------------------------
# Config loader 
# ------------------------------------------------------------------------------------

def load_mock_data_config(path: str) -> Dict:
    try:
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
            print(f"[MOCK] Loaded mock data config from {path}")
            return config
    except Exception as e:
        print(f"[MOCK] Failed to load mock data config: {e}")
        return {}


# ------------------------------------------------------------------------------------
# Dispatcher that uses loaded config
# ------------------------------------------------------------------------------------

def generate_mock_data_from_config(config: Dict) -> Dict:
    defaults = config.get("defaults", {})
    device = torch.device(defaults.get("device", "cpu"))
    dtype_str = defaults.get("dtype", "float32")
    dtype = getattr(torch, dtype_str)

    seed = defaults.get("seed", None)
    if seed is not None:
        set_mock_seed(seed)

    outputs = {}

    if "image_batch" in config:
        image_cfg = config["image_batch"]
        outputs["image_batch"] = generate_mock_image_batch(
            batch_size=image_cfg.get("batch_size", 8),
            channels=image_cfg.get("channels", 3),
            height=image_cfg.get("height", 256),
            device=device,
            dtype=dtype,
            value_range=tuple(image_cfg.get("value_range", [0.0, 1.0]))
        )

    if "latent_batch" in config:
        latent_cfg = config["latent_batch"]
        outputs["Latent_batch"] = generate_mock_latent_batch(
            batch_size_latent_cfg.get("batch_size", 8),
            latent_dim=latent_cfg.get("latent_dim", 4),
            height=latent_cfg.get("height", 32),
            width=latent_cfg.get("width", 32),
            device=device,
            dtype=dtype
        )

    if "text_embedding" in config:
        text_cfg = config["text_embedding"]
        outputs["text_embedding"] = generate_mock_text_embedding(
            batch_size=text_cfg.get("batch_size", 8),
            embedding_dim=text_cfg.get("embedding_dim", 768),
            device=device,
            dtype=dtype
        )

    if "unet_inputs" in config:
        unet_cfg = config["unet_inputs"]
        outputs["unet_inputs"] = generate_mock_unet_inputs(
            batch_size=unet_cfg.get("batch_size", 8),
            image_size=unet_cfg.get("image_size", 256),
            latent_dim=unet_cfg.get("latent_dim", 4),
            embedding_dim=unet_cfg.get("embedding_dim", 768),
            device=device,
            dtype=dtype
        )

    return outputs

    
# ---------------------------------------------------------------------------
# Convenience CLI test
-----------------------------------------------------------------------------

if __name__ =="__main__":
    config = load_mock_data_config("mock_data_config.yaml")
    data = generate_mock_data_from_config(config)
    for k, v in data.items():
        if isinstance(v, dict):
            print(f"{k}: {subk: subv.shape for subl, subv in v.items()} }")
        else:
            print(f"{k}: {v.shape}")

"""

# File: snapshot_tools.py 
"""
import torch
import os

SNAPSHOT_DIR = "tests/snapshots/"

def assert_tensor_close_to_snapshot(tensor, snapshot_name, atol=1e-4):
    path = os.path.join(SNAPSHOT_DIR, snapshot_name)
    if not os.path.exists(path):
        torch.save(tensor, path)
        raise AssertionError(f"Snapshot {snapshot_name} created. Rerun test.")
    expected = torch.load(path)
    if not torch.allclose(tensor, expected, atol=atol):
        raise AssertionError(f"snapshot_name} output differs from snapshot.)

"""

# File: device_utils.py   
"""
import os
import torch
import logging
from typing import Optional, Union, Tuple
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# Device parsing and discovery
# ------------------------------------------------------------------------------

def parse_device(device: Optional[Union[str, torch.device]] = None) -> torch.device:
    if device is None:
        return get_defualt_device()
    if isinstance(device, torch.device):
        return device
    return torch.device(device)

    
def get_defualt_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_device_count() -> int:
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    return 1

    
# --------------------------------------------------------------------------------
# Distributed / multi-GPU helpers
# --------------------------------------------------------------------------------

def distributed_initialize(
    backend: str = "nccl",
    init_method: Optional[str] = None,
    timeout_seconds: int = 1800
):
    # Safe distributed initialization helper.
    # Can be called safely on both single-node or multi-node cluster.
    # No-op if distributed already initialized.

    if not torch.distributed.is_available():
        logger.warning("[DIST] torch.distributed not available")
        return

    if torch.distributed.is_initialized():
        logger.info("[DIST] torch.distributed already initialized")
        return

    rank = int(os.eviron.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK," 0))

    init_method = init_method or "env://"

    logger.info(f"[DIST] Initializing distributed backend: {backend}")
    logger.info(f"[DIST] RANK: {rank} WORLD_SIZE: {world_size} LOCAL_RANK: {local_rank} INIT_METHOD: {init_method}")

    torch.distributed.init_process_group(
        backend=backend,
        init_method=init_method,
        timeout=torch.distributed.timedelta(seconds=timeout_seconds)
    )

    torch.cuda.set_device(local_rank)
    logger.info(f"[DIST] Successfully initialized distributed on device cuda:{local_rank}")


def is_distributed() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()

    
def get_global_rank() -> int:
    if is_distributed():
        return torch.distributed.get_rank()
    return 0


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))


def get_world_size() _> int:
    if is_distributed():
        return torch.distributed.get_world_size()
    return 1

    
# -------------------------------------------------------------------------------
# Device memory summary
# -------------------------------------------------------------------------------

def summarize_device(device: Optional[torch.device] = None):
    device = parse_device(device)
    logger.info(f"[DEVICE] Using device: {device}")

    if device.type == "cuda":
        idx = device.index if device.index is not None else torch.cuda.current_device()
        name = torch.cuda.get_device_name(idx)
        cap = torch.cuda.get_device_capability(idx)
        mem_total = torch.cuda.get_device_properties(idx).total_memory / 1024**3
        logger.info(f"[DEVICE] CUDA device: {name} (Capability {cap[0]}.{cap[1]})")
        logger.info(f"[DEVICE] Total memory: {mem_total:.2f} GB")

    elif device.type == "mps":
        logger.info("[DEVICE] Running on Apple MPS backend.")


    else: 
        logger.into("[DEVICE] Using CPU.")


# -------------------------------------------------------------------------------
# Automatic mixed precision context
# -------------------------------------------------------------------------------

@contextmanager
def autocast_context(device: Optional[torch.device] = None, enabled: bool = True):
    device = parse_device(device)

    if device.type == "cuda" and enabled:
        with torch.cuda.amp.autocast():
            yield
    elif device.type == "cpu" and hasattr(torch.cpu.amp, "autocast") and enabled:
        with torch.cpu.amp.autocast():
            yield
    else:
        yield


# ---------------------------------------------------------------------------------
# Device ID parsing from string input
# ---------------------------------------------------------------------------------

def parse_device_id(device_str: str) -> Tuple[str, int]:
    device = torch.device(device_str)
    return device.type, device.index or 0


# ---------------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------------

if __name__ == "__main__":
    summarize_device()
    logger.info(f"Device count: {get_device_count()}")
    logger.info(f"Distributed: {is_distributed()} | Global Rank: {get_global_rank()} | World Size: {get_world_size()}")


"""  

# File: reproducibility.py
"""
import os
import random
import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------------
# Global state tracker (for safety & debugging)
# --------------------------------------------------------------------------------

_current_seed: int = None
_strict_mode: bool = True


# --------------------------------------------------------------------------------
# Main reproducibility control
# --------------------------------------------------------------------------------

def set_seed(seed: int, strict: bool = True):
    # Fully seeds all random sources for reproducibility.

    # Args:
    # seed (int): The master seed to use.
    # strict (bool): If True, enfore full determinism (slower). If False, allows non-deterministic optimizations.

    global _current_seed, _strict_mode

    _current_seed = seed
    _strict_mode = strict

    logger.info(f"[REPRO] Setting global seed to {seed} | Strict mode: {strict}")

    # Python random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch CPU
    torch.manual.seed(seed)

    # PyTorch CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Torch CUDNN deterministic behavoir
    torch.backends.cudnn.benchmark = not strict

    if strict:
        logger.info("[REPRO] CUDNN deterministic mode enabled (slower, safer).")
    else: 
        logger.info("[REPRO] CUDNN benchmark mode enabled (faster, less strict).")

    # Enable reproducibility for some PyTorch dataloaders (if used with multiple workers)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_current_seed() -> int:
    # Return the last seed value set. 
    return _current_seed


def log_reproducibility_settings():
    # Print current reproducibility mode.
    logger.info(f"[REPRO] Current seed: {_current_seed} | Strict mode: {_strict_mode}")


def warn_about_nondeterminim():
    # Optional helper to warn if strict reproducibility might not be fully guaranteed.
    if torch.cuda.is_available():
        unsafe_ops = torch.cuda.get_device_properties(0).major < 7
        if unsafe_ops:
            logger.warning("[REPRO] WARNING: some older GPUs may not fully support determinsitc ops.")
        # Could be expanded for known nondeterministic ops.


# --------------------------------------------------------------------------------------
# Example entry point for repo startup
# --------------------------------------------------------------------------------------

if __name__ == "__main__":
    set_seed(42, strict=True)
    log_reproducibility_settings()


"""

# File: serialization_utils.py
"""
import os
import torch
import shutil
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------------
# Safe atomic file writing
# ---------------------------------------------------------------------------------

def atomic_save(obj: Any, path: str, tmp_suffix=".tmp"):
    # Atomically write a file to disk (avoid partial corrupt files on crash).

    tmp_path = path + tmp_suffix
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)
    logger.info(f"[SERIALIZE] Saved checkpoint to: {path}")


# ---------------------------------------------------------------------------------
# Move nested state dicts to CPU before saving
# ---------------------------------------------------------------------------------

def dict_to_cpu(state_dict: Dict) -> Dict:
    # Recursively move tensors in a nested state_dict to CPU.

    cpu_state = {}
    for k, v in state_dict.items():
        if isinstance(v, dict):
            cpu_state[k] = dict_to_cpu(v)
        elif torch.is_tensor(v):
            cpu_state[k] = v.cpu()
        else:
            cpu_state[k] = v
    return cpu_state


# ---------------------------------------------------------------------------------
# Save full checkpoint
# ---------------------------------------------------------------------------------

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[Any],
    step: int,
    epoch: int,
    path, str,
    extra: Optional[Dict] = None
):

    # Save full training checkpoint.

    state = {
        "model": dict_to_cpu(model.state_dict()),
        "optimizer": dict_to_cpu(optimizer.state_dict()) if optimizer else None,
        "scheduler": dict_to_cpu(scheduler.state_dict()) if scheduler else None,
        "step": step,
        "epoch": epoch,
        "extra": extra or {}
    }
    atomic_save(state, path)

    
# ----------------------------------------------------------------------------------
# Load full checkpoint 
# ----------------------------------------------------------------------------------


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    map_location: Optional[str] = "auto",
    strict: bool = True
) -> Dict:
    # Load full training checkpoint.
    # Returns full checkpoint dict (step, epoch, extra...).

    if map_location == "auto":
        map_location = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"[SERIALIZE] Loading checkpoint from: {path}")
    state = torch.load(path, map_location=map_location)

    model.load_state_dict(state["model"], strict=strict)

    if optimizer and state.get("optimizer"):
        optimizer.load_state_dict(state["optimizer"])

    if scheduler and state.get("scheduler"):
        scheduler.load_state_dict(state["optimizer"])

    logger.info(f"[SERIALIZE] Loaded model (strict={strict}), optimizer, scheduler from checkpoint.")

    return {
        "step": state.get("step", 0),
        "epoch": state.get("epoch", 0),
        "extra": state.get("extra", {})
    }


# ----------------------------------------------------------------------------------------
# Load model weights only (partial warm-start loading)
# ----------------------------------------------------------------------------------------

def load_model_weights_only(
    path: str,
    model: torch.nn.Module,
    map_location: Optional[str] = "auto",
    strict: bool = True
):
    # Load model weights only (ingore optimizer/scheduler states).
    if map_location == "auto":
        map_location = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"[SERIALIZE] Loading model weights only from: {path}")
    state = torch.load(path, map_location=map_location)

    model.load_state_dict(state["model"], strict=strict)
    logger.info(f"[SERIALIZE] Loaded model weights only (strict={strict}).")


# ---------------------------------------------------------------------------------------
# Summarize checkpoint contents
# ---------------------------------------------------------------------------------------

def summarize_checkpoints(path: str):
    # Print simple summary of checkpoint contents.

    state = torch.load(path, map_location="cpu")
    logger.info(f"[SERIALIZE] Checkpoint summary for {path}:")
    logger.info(f" - Step: {state.get('step')}")
    logger.info(f" - Epoch: {state.get('epoch')}")
    logger.info(f" - Extra: {state.get('extra')}")
    logger.info(f" - Model keys: {list(state['model'].keys())[:5]} ...")













"""

# File: config_mutator.py
"""
import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------------------
# Core nested config acessors
# ---------------------------------------------------------------------------------------

def get_nested(config: Dict, key_path: str) -> Any:
    # Get a nested value from a config dict given a dotted key path.
    # Example: "train.optimizer.lr"
    keys = key_path.split(".")
    current = config
    for key in keys:
        if key not in current:
            raise KeyError(f"Key '{key}' not founf in config path '{key_path}'")
        current = current[key]
    return current


def set_nested(config: Dict, key_path: str, value: Any):
    # Set a nested value in a config dict given a dotted key path. 
    # Example: "train.optimizer.lr"

    keys = key_path.split(".")
    current = config
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}   # create nested dicts if missing
        current = current[key]
    current[keys[-1]] = value



# --------------------------------------------------------------------------------
# Batch apply multiple mutations
# --------------------------------------------------------------------------------


def apply_mutations(config: Dict, mutations: Dict[str, Any]):
    # Apply multiple nested mutations to a config dict. 

    logger.info(f"[MUTATOR] Applying {len(mutations)} mutations...")
    for key_path, value in mutations.items():
        logger.info(f"[MUTATOR] Setting {key_path} -> {value}")
        set_nested(config, key_path, value)


# --------------------------------------------------------------------------------
# CLI-style string parsing helper (optional for command-line overrides)
# --------------------------------------------------------------------------------


def mutation_from_cli_args(args: List[str]) -> Dict[str, Any]:
    # Convert list of CLI args like ["train.optimizer.lr=0.001", model.depth=12]
    # into mutations dict: { "train.optimizer.lr": 0.001, "model.depth": 12 }

    mutations = {}
    for arg in args:
        if "=" not in arg:
            raise ValueError(f"Invalid mutation argument: {arg}")
        key_path, value_str = arg.split("=", 1)
        value = infer_value_type(value_str)
        mutations[key_path] = value
    return mutations

def infer_value_type(value_str: str) -> Any:
# Try to infer correct Python type from string. 

    lowered = value_str.lower()

    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "none":
        return None

    try:
        if "." in value_str:
            return float(value_str)
        return int(value_str)
    except ValueError:
        return value_str







"""

# File: benchmark_utils.py
"""
import time
import torch
import logging
from typing import Callable, Optional, Any, Tuple
from context import contextmanager

logger = logging.getLogger(__name__)

# Try to import ptflops (optional dependency)
try:
    from ptflops import get_model_complexity_info
    PT_FLOPS_AVAILABLE = True
except ImportError:
    logger.warning("[BENCH] ptflops not installed - FLOP estimation disabled")
    PT_FLOPS_AVAILABLE = False

# -------------------------------------------------------------------------------------------
# Simple timing context 
# -------------------------------------------------------------------------------------------

@contextmanager
def time_function(description: str = "Block"):
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    elapsed = end - start
    logger.info(f"[BENCH] {description} took {elapsed:.4f} seconds.")


# -------------------------------------------------------------------------------------------
# Forward pass benchmark
# -------------------------------------------------------------------------------------------

def benchmark_forward_pass(
    model: torch.nn.Module,
    inputs: Any,
    device: Optional[torch.device] = None,
    warmup: int = 3,
    runs: int = 10,
    amp: bool = False
):

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    # Move inputs to device
    if isinstance(inputs, dict):
        inputs = {k: v.to(device) for k, v in inputs.items()}
    elif isinstance(inputs, (list, tuple)):
        inputs = [v.to(device) for v in inputs]
    else:
        inputs = inputs.to(device)

    logger.info("[BENCH] Warming up...")
    with torch.no_grad():
        for _ in range(warmpup):
            with torch.cuda.amp.autocast(enabled=amp)
                modek(**inputs) if isinstance(inputs, dict) else model(inputs)

    torch.cuda.synchronize()

    logger.info("[BENCH] Mesuring forward pass...")
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(runs):
            with torch.cuda.amp.autocast(enabled=amp):
                model(**inputs) if isinstance(inputs, dict) else model(inputs)
    torch.cuda.synchronize()
    end = time.perf_counter()

    avg_time = (end - start) / runs
    logger.info(f"[BENCH] Forward pass average: {avg_time:.6f} seconds per run.")

    
# ----------------------------------------------------------------------------------
# Forward + backward pass benchmark
# ----------------------------------------------------------------------------------

def benchmark_backward_pass(
    model: torch.nn.Module,
    inputs: Any, 
    loss_fn: Callable,
    device: Optional[torch.device] = None,
    warmup: int = 3,
    runs: int = 10,
    amp: bool = False
):
    device = device or torch.device("cuda" if torch.cuda.is_avalable() else "cpu")
    model = model.to(device).train()

    # Move inputs to device
    if isinstance(inputs, dict):
        inputs = {k, v.to(device) for k, v in inputs.items()}
    elif isinstance(inputs, (list, tuple)):
        inputs = [v.to(device) for v in inputs]
    else:
        inputs = inputs.to(device)

    logger.info("[BENCH] Warming up (forward + backward)...")
    for _in range(warmup):
        model.zero_grad()
        with torch.cuda.amp.autocast(enabled=amp):
            output = model(**inputs) if isinstance(inputs, dict) else model(inputs)
            loss = loss_fn(output)
        loss.backward()

    torch.cuda.synchronize()

    logger.info("[BENCH] Mesuring forward + backward...")
    start = time.perf_counter()
    for _ in range(runs):
        model.zero_grad()
        with torch.cuda.amp.autocast(enabled=amp):
            output = model(**inputs) if isinstance(inputs, dict) else model(inputs)
            loss = loss_fn(output)
        loss.backward()
    torch.cuda.synchronize()
    end = time.perf_counter()

    avg_time = (end - start) / runs
    logger.info(f"[BENCH] Forward+Backward average: {avg_time:.6f} seconds per run.")


# -----------------------------------------------------------------------------------
# Model parameter count
# -----------------------------------------------------------------------------------

def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel(), for p in model.parameters())


def log_model_size(model: torch.nn.Module):
    total = count_parameters(model)
    logger.info(f"[BENCH] Model has {torch:,} parmaters ({total / 1e6:.2f}M)")


# -----------------------------------------------------------------------------------
# FLOP estimation using ptflops
# -----------------------------------------------------------------------------------

def estimate_flops(
    model: torch.nn.Module,
    input_shape: Tuple[int, ...],
    device: Optional[torch.device] = None,
    custom_forward: Optional[Callable] = None
):
    if not PT_FLOPS_AVAILABLE:
        logger.warning("[BENCH] pt.flops not installed - skipping FLOP estimation")
        return None

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    with torch.cuda.amp.autocast(enabled=False):
        macs, params = get_model_complexity_info(
            model, input_shape,
            as_strings=True,
            print_per_layer_stat=False,
            verbose=False,
            custom_modules_hooks=custom_forward
        )
    logger.info(f"[BENCH] Estimated MACs: {macs} | Parameters: {params}")


# -------------------------------------------------------------------------------
# Memory reporting
# -------------------------------------------------------------------------------

def log_memory_snapshot(device: Optional[torch.device] = None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        allocated = torch.cuda.memory_allocated(device) / 1024**2
        reserved = torch.cuda.memory_reserved(device) / 1024**2
        logger.info(f"[BENCH] CUDA memory: allocated={allocated:.2f} MB, reserved={reserved:.2f} MB")


# -------------------------------------------------------------------------------
# Full benchmark summary
# -------------------------------------------------------------------------------

def benchmark_summary(
    model: torch.nn.Module,
    inputs: Any,
    input_shape_for_flops: Optional[Tuple[int, ...]] = None
):
    log_model_size(model)
    log_memory_snapshot()
    benchmark_forward_pass(model, inputs)
    if input_shape_for_flops:
        estimate_flops(model, input_shape_for_flops)


"""

# File: logging_helpers.py
"""
import logging
import sys
import os
from typing import Optional

# -------------------------------------------------------------------------------
# Unified log formatter
# -------------------------------------------------------------------------------

class ColorFormatter(logging.Formatter):
    # Simple colored log formatter for console output.

    COLOR_CODES = {
        logging.DEBUG: "\033[36m",      # Cyan
        logging.INFO:  "\033[32m",      # Green
        logging.WARNING: "\033[33m",    # Yellow
        logging.ERROR: "\033[31m",      # Red
        logging
    
    }
    RESET_CODE = "\003[0m"

    def format(self, record):
        color = self.COLOR_CODES.get(record.levelno, self.RESET_CODE)
        message = super().format(record)
        return f"{color}{message}{self.RESET_CODE}"


# -------------------------------------------------------------------------------
# Global logger setup
# -------------------------------------------------------------------------------

def initialize_logging(
    console_level: int = logging.INFO,
    file_path: Optional[str] = None,
    file_level: int = logging.DEBUG,
    use_color: bool = True
):

    # Initialize global logging configuration.

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(logging.DEBUG) # Always capture all logs internally.

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = (
        ColorFormatter("[%(asctime)s] [%(levelname)s] %(message)s")
        if use_color else
        logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
    )
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(console_level)
    root_logger.addHandler(console_handler)

    # Optional file handler
    if file_path
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        file_handler = logging.FileHandler(file_path, mode='a')
        file_formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(file_level)
        root_logger.addHandler(file_handler)

    # Silence noisy libraries if desired (optional)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    root_logger.info("[LOGGING] Logging initialized.")


# -------------------------------------------------------------------------------
# Utility fuction to safely get loggers
# -------------------------------------------------------------------------------

def get_logger(name: str) -> logging.Logger:
    # Safe way to get loggers throughout repo.

    return logging.getLogger(name)


# -------------------------------------------------------------------------------
# Allow changing log level dynamically
# -------------------------------------------------------------------------------

def set_global_log_level(level: int):
    # Dynamically update log level for all handlers.
    
    root_logger = logging.getLogger()
    for handler in rool_logger.handlers:
        handler.setLevel(level)
    root_logger.info(f"[LOGGING] Global log level set to {logging.getLevelName(level)}")

# Todo list:

# wandb integration
# tensorboard scalars
# remote logging or cluster-wide experiment monitoring    

"""

# File: tensor_inspect.py
"""
import torch
import torch.distributed as dist
import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# Distributed-safe tensor aggregation helpers
# ------------------------------------------------------------------------------

def distributed_recude_scalar(value: float, op: str) -> float:
    # All-reduce scalar float across ranks.
    tensor = torch.tensor([value], dtype=torch.float32, deivce="cuda" if torch.cuda.is_available() else "cpu")
    if op == "sum":
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    elif op == "min":
        dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
    elif op == "max":
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    else:
        raise ValueError(f"Unsupported reduce op: {op}")
    return tensor.item()


# -------------------------------------------------------------------------------
# Distributed Collector
# -------------------------------------------------------------------------------

class DistributedTensorCollector:
    # Tensor collector that safely aggregates across all distributed processes.

    def __init_(self):
        self.records: Dict[sir, List[Dict[str, Any]]] = {}

    def collect(self, tensor: torch.Tensor, name: str, mode: str = "forward"):
        stats = {
            "mode": mode,
            "shape": tuple(tensor.shape),
            "local_min": float(tensor.min().item()),
            "local_max": float(tensor.max().item()),
            "local_mean": float(tensor.mean().item()),
            "local_std": float(tensor.std().item()),
            "local_nan": torch.isnan(tensor).any().item(),
            "local_inf": torch.isinf(tensor).any().item(),
        }
        full_key = f"{name}:{mode}"

        if full_key not in self.records:
            self.records[full_key] = []
        self.records[full_key].append(stats)

    def distributed_sync(self):
        # Preform all_reduce aggregetion across all ranks.

        logger.info("[INSPECT-DICT] Performing distribued tensor aggregation...")

        for key, entries in self.records.items():
            latest = entris[-1]
            global_min = distributed_reduce_scalar(latest["local_min"], "min")
            global_max = distributed_reduce_scalar(latest["local_max"], "max")
            global_sum = distributed_reduce_scalar(latest["local_mean"], "sum")
            global_std_sum = distributed_reduce_scalar(latest["local_std"], "sum")

            world_size = dist.get_world_size()

            global_mean = global_sum / world_size
            global_std = global_std_sum / world_size

            latest["global_min"] = global_min
            latest["global_max"] = global_max
            latest["global_mean"] = global_mean
            latest["global_std"] = global_std

    def log_step(self, step: int, only_rank0: bool = True):
        if only_rank0 and dist.get_rank() != 0:
            return

        logger.info(f"[INSPECT-DIST] Global tensor stats at step {step}:")
        for key, entries in self.records.items():
            latest = entries[-1]
            logger.info(
                f"  {key} | min={latest['global_min']:.4g} | max={latest['global_max']:.4g} | "
                f"mean={latest['global_mean']:.4g} | std={latest['global_std']:.4g}"
            )


# -------------------------------------------------------------------------------------
# Forward hook generator 
# -------------------------------------------------------------------------------------

def create_forward_hook(name: str, collector: DistributedTensorCollector) -> Callable:
    def hook(module, inpt, output):
        if isinstance(output, torch.Tensor):
            collector.collect(output, name, mode="forward")
        elif is instance(output, (tuple, list)):
            for idx, tensor in enumerate(output):
                if isinstance(tensor, torch.Tensor):
                    collector.collect(tensor, f"{name}[{idx}]", mode="forward")
        elif isinstance(output, dict):
            for key, tensor in output.items():
                if isinstance(tensor, torch.Tensor):
                    collector.collect(tensor, f"{name}.{key}", mode="forward")
    return hook


# ------------------------------------------------------------------------------------
# Backward hook generator
# ------------------------------------------------------------------------------------

def create_backward_hook(name: str, collector: DistributedTensorCollector) -> Callable:
    def hook(module, input, output):
        if isinstance(output, torch.Tensor):
            output.register_hook(lambda grad: collector.collect(grab, name, mode="backward"))   
        elif isinstance(output, (tuple, list)):
            for idx, tensor in enumerate(output):
                if isinstance(tensor, torch.Tensor):
                    tensor.register_hook(lambda grad, idx=idx: collector.collect(grad, f"{name}[{idx}]", mode="backward"))
    return hook

    
# -----------------------------------------------------------------------------------
# Unified Inspector 
# -----------------------------------------------------------------------------------


class TensorInspector:
    # Unified tensor inspector with full Collector support.

    def __init__(self, model: torch.nn.Module, collector: DistributedTensorCollector, monitor_gradients: bool = False):
        self.model = model
        self.collector = collector
        self.monitor_gradients = monitor_gradients
        self.hooks: List[torch.utils.hooks.RemoveableHandle] = []

    def register_all(self, filter_fn: Optional[Callable[[str, torch.nn.Module], bool]] = None):
        logger.info("[INSPECT] Registering hooks for tensor inspection")

        for name, module in self.model.named_modules():
            if filter_fn and not filter_fn(name, module):
                continue
                
            fwd_hook = module.register_forward_hook(create_forward_hook(name, self.collector))
            self.hooks.append(fwd_hook)

            if self.monitor_gradients:
                post_hook = module.register_forward_hook(create_backward_hook(name, self.collector))
                self.hooks.append(post_hook)

        logger.info(f"[INSPECT] Registered {len(self.hooks)} hooks.")

    def clear_hooks(self):
        for hook in self.hooks:
            hook.remove()
        logger.info("[INSPECT] Cleared all hooks.")
        self.hooks.clear()

    def __enter__(self):
        self.register_all()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.clear_hooks()    

"""

# File: failure_injection.py       
"""
import torch
import numpy as np
import os
import random
import signal
import time
import io
import sys

class FailureInjector:
    # Injects simulated failures into tests to validate error handling robustness. 

    @staticmethod
    def nan_injection(tensor: torch.Tensor, ratio=0.01):
        # Randomly inject NaNs into tensor.

        mask = torch.rand_like(tensor) < ratio
        tensor[mask] = float('nan')
        return tensor

    @staticmethod
    def inf_injection(tensor: torch.Tensor, ratio=0.01):
        # Randomly inject infinities into tensor.
        mask = torch.rand_like(tensor) < ratio
        tensor[mask] = float('inf')
        return tensor

    @staticmethod
    def disk_write_failure(path="./", chance=0.1):
        # Randomly simulate disk faliure by removing permissions.
        if random.random() < chance:
        os.chmod(path, 0o000)
        time.sleep(0.1)
        os.chmod(path, 0o755)

    @staticmethod
    def kill_process(probability=0.05):
        # Randomly kill the current process.
        if random.random() < probability:
            print("Simulated random kill.")
            os.kill(os.getpid(), signal.SIGTERM)

    @staticmethod
    def gpu_memory_spike(size_gb=10):
        # Intenstionally allocate huge GPU tensor to simulate OOM.
        if torch.cuda.is_available():
            torch.randn((size_gb * 1024**3) // 4, device='cuda')

    @staticmethod
    def network_failure(probability=0.05):
        # Simulate network drop (mock for distributed workers).
        if random.random() < probability:
            raise ConnectionError("Simulated netwokr failure.")

    @staticmethod
    def slow_io(delay_seconds=3.0):
        # Simulate extremely slow dick IO (for checkpoint systems).
        print(f"Simulated IO delay of {delay_seconds} seconds")
        time.sleep(delay_seconds)

    @staticmethod
    def simulate_checkpoint_corruption(file_path):
        # Overwrite file with junk to simulate corrupted checkpoint.
        with open(file_path, "wb") as f:
            f.write(os.urandom(1024))

    @staticmethod
    def memory_leak_simulation(iterations=100):
        # Simulate a memory leak over interations.
        leak = []
        for _ in range(iterations):
            leak.appen(bytearray(10**6)) # 1 MB per iteration

    @staticmethod
    def randomized_cpu_spike(duration=2.0):
        # Max out CPU cores for duration. 
        start = time.time()
        while time.time() - start < duration:
            [x**2 for x in range(10**5)]

    @staticmethod
    def corrupt_training_state(model, ratio=0.1):
        # Corrupt some parameters in model (for checkpoint restore tests).
        with torch.no_grad():
            for param in model.parameters():
            mask = torch.rand_like(param) < ratio
            param[mask] = torch.randn_like(param)[mask]
        return model

    @staticmethod
    def throw_random_assert(probability=0.01):
        # Throw random assertaion failure.
        if random.random() < probability:
            assert False, "Random simulated assertaion failure."

    @staticmethod
    def stdout_flood(lines=1000)
        # Overload consule/logging stream.
        for i in range(lines):
            print(f"Debug flood line {i+1}")

    @staticmethod
    def corrupt_optimizer_state(optimizer):
        # Zero out optimizer state to simulate optimizer corruption. 
        optimizer.state = {}

    @staticmethod
    def random_train_loop_skip(probability=0.05):
        # Randomly skip full train step.
        if random.random() < probability:
            print("Skipping simulated train step failure.")
            return True
        return False
"""

# File: distributed_mocks.py
"""
import torch.distributed as dist
import types
import logging

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------------
# Global patch size
# -------------------------------------------------------------------------------


class DistributedMockContext:
    # Context manager to mock torch.distributed for local testing.

    def __init__(self, world_size: int = 4):
        self.world_size = world_size
        self.rank = 0   # Default rank for single-process testing

    def __enter__(self):
        logger.info(f"[MOCK-DIST] Entering distributed mock with world_size={self.world_size}")

        global _original_distributed
        _original_distributed = dist

        # Create fake distributed module
        mock_dist = types.SimpleNamespace()

        mock_dist.is_available = lambda: True
        mock_dist.is_initialized = lambda: True

        def mock_all_reduce(tensor, op=None, group=None):
            # For testing, we assume single-process all_reduce does nothing
            logger.info(f"[MOCK-DIST] all_reduce called (mocked)")
            return tensor

        mock_dist.all_reduce = mock_all_reduce
        mock_dist.ReduceOp = types.SimpleNamespace(SUM="sum", MIN="min", MAX="max")

        # Patch torch.distributed
        torch.distributed = mock_dist

    def set_rank(self, rank: int):
        # For multi-rank simulation within unit tests.

        self.rank = rank

    def __exit__(self, exc_type, exc_val, exc_tb):
        global _original_distributed
        torch.distributed = _original_distributed
        logger.info("[MOCK-DIST] Exiting distributed mock context")

"""

# File: metrics_synthetic.py
"""
import numpy as np
import logging
from typing import List, Tuple

logger = loggin.getLogger(__name__)


# --------------------------------------------------------------------------------
# Loss curve generator
# --------------------------------------------------------------------------------

def generate_sythetic_loss_curve(
    steps: int = 1000,
    start: float = 5.0,
    end: float = 0.1,
    noise_level: float = 0.05,
    seed: int = 42
) -> List[float]:
    # Simulates a decayinging loss curve with added noise.

    np.random.seed(seed)
    decay = np.linspace(start, end, steps)
    noise = np.random.normal(0, noise_level, steps)
    loss_curve = decay + noise
    loss_curve = np.maximum(loss_curve, 0.0)
    logger.info(f"[SYNTHETIC] Generated synthetic loss curve ({steps} steps).")
    return loss_curve.tolist()


# -------------------------------------------------------------------------------
# Accuracy curve generator
# -------------------------------------------------------------------------------

def generate_synthetic_accuracy_curve(
    steps: int = 1000,
    start: float = 0.2,
    end: float = 0.95,
    noise_level: float = 0.02,
    seed: int = 42
) -> List[float]:
    # Simulatesand increasing accuracy curve with noise.

    np.random.seed(seed)
    growth = np.linspace(start, end, steps)
    noise = np.random.normal(0, noise_level, steps)
    acc_curve = growth + noise
    acc_curve = np.clip(acc_curve, 0.0, 1.0)
    logger.info(f"[SYNTHETIC] Generated synthetic accuracy curve ({steps} steps).")
    return acc_curve.tolist()


# --------------------------------------------------------------------------------
# Gradient norm generator
# --------------------------------------------------------------------------------

def generate_synthetic_gradient_norms(
    steps: int = 1000,
    base: float = 1.0,
    spike_chance: float = 0.01,
    spike_magnitude: float = 10.0,
    seed: int = 42
) -> List[float]:
    # Simulates noisy gradient norms with occasional spikes.

    np.random.seed(seed)
    norms = base + np.random.normal(0, 0.2, steps)
    spikes = np.random.rand(steps) < spike_chance
    norms[spikes] += spike_magnitude
    norms = np.maximum(norms, 0.0)
    logger.info(f"[SYNTHETIC] Generated synthetic gradient norm curve ({steps} steps).")
    return norms.tolist()


# --------------------------------------------------------------------------------
# Learning rate schedule generator
# --------------------------------------------------------------------------------

def generate_synthetic_learning_rate_schedule(
    steps: int = 1000,
    base_lr: float = 0.001,
    warmup_steps: int = 100,
    decay_factor: float = 0.5,
    decay_steps: int = 300
) -> List[float]:
    # Simulates a learning rate schedule with warmup and decay.
    lr_schedule = []
    for step in range(steps):
        if step < warmup_steps:
            lr = base_lr * (step + 1) / warmup_steps
        else:
            decay_cycles = (step - warmup_steps) // decay_steps
            lr = base_lr * (decay_factor ** decay_cycles)
        lr_schedule.append(lr)
    logger.info(f"[SYNTHETIC] Generated synthetic learning rate schedule ({steps} steps).")
    return lr_schedule


# ---------------------------------------------------------------------------------
# Timing profile generator
# ---------------------------------------------------------------------------------

def generate_synthetic_timing_profile(
    steps: int = 1000,
    base_time: float = 0.1,
    jitter: float = 0.01,
    seed: int = 42
) -> List[float]:
    # Simulates forward-backward step time fluctuations.

    np.random.seed(seed)
    times = base_time + np.random.normal(0, jitter, steps)
    times = np.maximum(times, 0.0)
    logger.info(f"[SYNTHETIC] Generated synthetic timing profile ({steps} steps.)")
    return times.tolist()


"""


#           / snapshots/

# File: __init__.py

# File: sample_output_ref.pt            #(saved outputs to compare against)

#                       loss_curves/

# File: mse_vs_lpips_curve.json         #(defalt)

# File: sample_weighting_test.json      #(defalt)

#                       outputs/

# File: step_100_sample_grid.png        #(saved grid at x step)

# File: latent_noise_overlay_0.png      #(saved latent noise tragectory)

#                       attention/  

# File: qk_heatmap_step_10_layer2.png   #(saved attention heatmaps)

# File: attention_weights_step20.npy    #(saved attenion step weights)

#                       resblock/

# File: resblock_vanilla_snapshot.pt    #(saved expected outputs from resblock testing)

# File: film_resblock_activation.png    #(Saved visual debug images)














