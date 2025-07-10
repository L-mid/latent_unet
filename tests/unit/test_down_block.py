
import torch
import pytest
from modules.down_block import DownBlock


class TestDownBlock:

    def test_shape_compatability(self):
        x = torch.randn(2, 64, 32, 32)
        t = torch.randn(2, 128)
        block = DownBlock(64, 128, time_emb_dim=128)
        y, _ = block(x, t)
        assert y.shape[1] == 128
        assert y.shape[2] < 32 and y.shape[3] < 32

    def test_hook_traceability(self):
        activations = []
        def hook_fn(module, input, output):
            activations.append(output)

        x = torch.randn(1, 64, 32, 32)
        t = torch.randn(1, 128)
        block = DownBlock(64, 128, time_emb_dim=128)
        handle = block.resblocks[0].register_forward_hook(hook_fn)
        _ = block(x, t)
        handle.remove()
        assert len(activations) == 1

    def test_differentiability(self):
        x = torch.randn(2, 64, 32, 32, requires_grad=True)
        t = torch.randn(2, 128)
        block = DownBlock(64, 128, time_emb_dim=128)
        y, _ = block(x, t)
        loss = y.mean()
        loss.backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()















