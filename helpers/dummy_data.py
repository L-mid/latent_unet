
import torch
import numpy as np
import random

# === NOTES
"""
Unused, untested.
"""

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
        return torch.randint(0, max_timesteps, self.channels, self.height, self.width, device=self.device, dtype=self.dtype)
    
    def random_noise(self):
        return torch.randn(self.batch_size, self.channels, self.height, self.width, device=self.device, dtype=self.dtype)
    
    def random_noised_input(self):
        x_start = self.random_images()
        noise = self.random_noise()
        return x_start, noise
    
    def conditioning(self, embed_dim=512):
        return torch.randn(self.batch_size, embed_dim, self.height, self.width, device=self.device, dtype=self.dtype)
    
    def classifier_labels(self, num_classes=1000):
        return NotImplementedError
    
    def tokenized_prompts(self, seq_len=77, vocab_size=32000):
        return torch.randint(0, vocab_size, (self.batch_size, seq_len), device=self.device)
    
    def latents(self, latent_channels=4, downsample=8):
        return torch.randn(
            self.batch_size, latent_channels,
            self.height // downsample, self.width // downsample,
            device=self.device, dtype=self.dtype
        )
    
    def deterministic_noise(self, noise_seed=123):
        # Allows deterministic noise for perfect repeatable tests.

        g = torch.Generator(device=self.device).manual_seed(noise_seed)
        return torch.randn(self.batch_size, self.channels, self.height, self.width, device=self.device, generator=g, dtype=self.dtype)

    def mixprecision_images(self):
        # Simulate mixed-precision forward tests.

        return self.random_images().half()
    
    def curriculum_timesteps(self, current_epoch, max_epochs, min_t=10, max_t=1000):
        # Simulate curriculum sampling schedule for timestep distributions.
        limit_t = int(min_t + (max_t - min_t) * min(current_epoch / max_epochs, 1.0))
        return torch.randint(min_t, limit_t + 1, (self.batch_size,), device=self.device)