
from omegaconf import OmegaConf, DictConfig
import os

# Simple YAML reader via OmegaConf (no typed dataclassing)


def load_config(path: str, overrides: dict = None) -> DictConfig:
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    
    # Load YAML into OmegaConf object

    cfg = OmegaConf.load(path)

    # Merge overrides if provided
    if overrides:
        override_cfg = OmegaConf.create(overrides)
        cfg = OmegaConf.merge(cfg, override_cfg)

    return cfg


# Convert OmegaConf object to plain nested dict (for logging or export)

def config_to_dict(cfg: DictConfig) -> dict:
    return OmegaConf.to_container(cfg, resolve=True)







