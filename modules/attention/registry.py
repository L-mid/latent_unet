
from typing import Callable
import torch.nn as nn


ATTENTION_REGISTRY = {}

def register_attention(name: str) -> Callable:
    """Decorator to register an attention class by name."""
    def decorator(cls):
        ATTENTION_REGISTRY[name] = cls
        return cls
    return decorator


def get_attention(dim: int, cfg) -> 'nn.Module | None':
    """Construct an attention module from config and injected dim."""
    name = cfg.kind.lower()

    if name == "none":
        return None

    if name not in ATTENTION_REGISTRY:
            raise ValueError(f"Unknown attention type: {name}")
    
    params = dict(cfg.get("params", {}))
    return ATTENTION_REGISTRY[name](dim, **params)
