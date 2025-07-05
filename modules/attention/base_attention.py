
import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class BaseAttention(nn.Module, ABC):
    # Abstract base class for attention blocks.
    # All attention subclasses (Vanilla, flash, windowed, etc.) should inherit from this.

    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

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
        return f"dim={self.dim}, heads={self.num_heads}"












