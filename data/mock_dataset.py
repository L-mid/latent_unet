
import torch 
import numpy as np
import yaml
from typing import Optional, Tuple, Dict

"""
No such file or directory for loading yaml.
"""

# -----------------------------------------------------------------------------
# Core random seed control (optional from reproducibility)
# -----------------------------------------------------------------------------

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
    data = {high - low} * torch.rand(batch_size, channels, width, device=device, dtype=dtype) + low
    return data


# -----------------------------------------------------------------------------------
# Latent batch generator (for diffusion latents, VAE, etc)
# -----------------------------------------------------------------------------------

def generate_mock_latent_batch(
        batch_size: int = 8,
        latent_dim: int = 3,
        height: int = 32,
        width: int = 32, 
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    
    # Generate a mock latent batch (e.g. VAE latent space).
    device = device or torch.device("cpu")
    data = torch.randn(batch_size, latent_dim, height, width, device=device, dtype=dtype)
    return data




# -----------------------------------------------------------------------------
# Mock text embedding genorator (for conditioning inputs)
# ----------------------------------------------------------------------------------

def generate_mock_text_embedding(
        batch_size: int = 8,
        embedding_dim: int = 768,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    
    # Generate a mock test embedding batch (e.g. CLIP text encoder output)

    device = device or torch.device("cpu")
    data = torch.randn(batch_size, embedding_dim, device=device, dtype=dtype)
    return data


# --------------------------------------------------------------------------
# UNet-style combined mock input generator (for diffusion testing)
# -----------------------------------------------------------------------------

def generate_mock_unet_inputs(
    batch_size: int = 8,
    image_size: int = 256,
    latent_dim: int = 4,
    embedding_dim: int = 768,
    device: Optional[torch.device] = None,
    dtype: float = torch.float32,               # i think dtype is a float?
) -> dict:
    # Generate full mock input set for a diffusion U-Net forward pass.
    device = device or torch.device("cpu")

    latents = generate_mock_latent_batch(
        batch_size=batch_size,
        image_size=image_size,
        latent_dim=latent_dim,
        height=image_size//8,
        width=image_size//8,
        device=device,
        dtype=dtype,
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


# ------------------------------------------------------------------------------
# Config loader
# ------------------------------------------------------------------------------

def load_mock_data_config(path: str) -> Dict:
    try:
        with open(path, "r") as f:
            config = yaml.safe_load(f)
            print(f"[MOCK DATA] Loaded mock data config from {path}")
            return config
    except Exception as e: 
        print(f"[MOCK DATA] Failed to load mock data config: {e}")
        return {}
    

# -------------------------------------------------------------------------------
# Dispatcher that uses loaded config
# -------------------------------------------------------------------------------

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
            batch_size=latent_cfg.get("batch_size", 8),
            latent_dim=latent_cfg.get("latent_dim", 4),
            height=latent_cfg.get("height", 32),
            width=latent_cfg.get("width", 32),
            device=device,
            dtype=dtype,
        )


    if "text_embedding" in config:
        text_cfg = config["text_embedding"]
        outputs["text_embedding"] = generate_mock_text_embedding(
            batch_size=text_cfg.get("batch_size", 8),
            embedding_dim=text_cfg.get("embedding_dim", 768),
            device=device,
            dtype=dtype,
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


# ---------------------------------------------------------------------------------
# Convenience CLI test
# ---------------------------------------------------------------------------------

if __name__ == "__main__":
    config = load_mock_data_config("configs/mock_data_config.yaml")         # not sure why this doesn't load, name is correct
    data = generate_mock_data_from_config(config)
    for k, v in data.items():
        if isinstance(v, dict):
            print(f"{k}: {{subk: subv.shape for subl, subv in v.items()}}")
        else:
            print(f"{k}: {v.shape}")











