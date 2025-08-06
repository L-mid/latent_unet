
import torch
import pytest
from modules.up_block import UpBlock


class TestUpBlock:

    def test_shape_compatibility(self):
        x = torch.randn(2, 128, 16, 16)
        skip = torch.randn(2, 64, 32, 32)
        skip_channels = 64
        t = torch.randn(2, 128)
        block = UpBlock(in_ch=128, out_ch=64, time_emb_dim=128, skip_channels=skip_channels)
        y = block(x, skip, t)
        assert y.shape == skip.shape

    def test_skip_connection_merge(self):
        x = torch.randn(2, 128, 16, 16)
        skip = torch.randn(2, 64, 32, 32)
        t = torch.randn(2, 128)
        block = UpBlock(128, 64, time_emb_dim=128, skip_channels=64)
        out = block(x, skip, t)
        assert out.shape[2:] == skip.shape[2:]

    def test_hook_traceability(self):
        activations = []
        def hook_fn(module, input, output):
            activations.append(output)

        x = torch.randn(1, 128, 16, 16)
        skip = torch.randn(1, 64, 32, 32)
        t = torch.randn(1, 128)
        block = UpBlock(128, 64, time_emb_dim=128, skip_channels=64)
        handle = block.resblocks[0].register_forward_hook(hook_fn)
        _ = block(x, skip, t)
        handle.remove()
        assert len(activations) == 1

    def test_differentiability(self):
        x = torch.randn(2, 128, 16, 16, requires_grad=True)
        skip = torch.randn(2, 64, 32, 32)
        t = torch.randn(2, 128)
        block = UpBlock(128, 64, time_emb_dim=128, skip_channels=64)
        y = block(x, skip, t)
        loss = y.mean()
        loss.backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()
    



