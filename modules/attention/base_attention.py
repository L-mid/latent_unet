
import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class BaseAttention(nn.Module, ABC):
    # Abstract base class for attention blocks.
    # All attention subclasses (Vanilla, flash, windowed, etc.) should inherit from this.
    # Handles common constructor args, even if unused by a subclass.

    def __init__(
        self, 
        dim: int, 
        num_heads: int = 4,
        norm_groups: int = None,
        dim_head: int = None,
        start_layer: int = 0,
        window_size: int = None,
        backend: str = None,
        **kwargs    # absorb any extra config params
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.norm_groups = norm_groups
        self.dim_head = dim_head
        self.start_layer = start_layer
        self.window_size = window_size
        self.backend = backend


    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward pass for attention layer.

        """
        Args:
            x (Tensor): shape (B, C, H, W)
        Returns: 
            Tensor: shape (B, C, H, W)
        """
        raise NotImplementedError("All attention modules must impliment forward()")

    def extra_repr(self):
        return (
            f"dim={self.dim}, heads={self.num_heads}, norm_groups={self.norm_groups}, "
            f"dim_head={self.dim_head}, start_layer=(self.start_layer), "
            f"window_size={self.window_size}, backend={self.backend}"
        )












