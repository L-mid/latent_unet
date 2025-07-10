
import torch
import pytest
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from modules.residual_block import get_resblock, FiLMResBlock
from modules.norm_utils import get_norm_layer
from utils.snapshot_tools import assert_tensor_close_to_snapshot


# --- Base Utilities ---

KINDS = ["vanilla", "film"]
NORM_TYPES = ["group", "batch", "layer"]


def cfg_setup(kind: str, norm_type="group"):
                
    base_cfg = {
        "kind": kind,
        "params": {
            "norm_type": norm_type,
            "attention_layer": None
        }
    }
    return OmegaConf.create(base_cfg)


# ------ Shared Test Group for ResBlocks ------

class TestResBlockShared:
    
    @pytest.mark.parametrize("norm_type", NORM_TYPES)
    @pytest.mark.parametrize("kind", KINDS)
    def test_output_shape(self, kind, norm_type):

        cfg = cfg_setup(kind, norm_type)

        x = torch.randn(4, 128, 32, 32)
        t = torch.randn(4, 512) # Time embedding


        block = get_resblock(cfg, in_ch=128, out_ch=128, time_dim=512)

        y = block(x, t)
        assert y.shape == x.shape, f"{kind} {norm_type} shape mismatch"
    

    @pytest.mark.parametrize("kind", KINDS)
    def test_backprop(self, kind):
        cfg = cfg_setup(kind)
        x = torch.randn(2, 64, 16, 16, requires_grad=True)
        t = torch.randn(2, 128, requires_grad=True)

        block = get_resblock(cfg, in_ch=64, out_ch=64, time_dim=128)
        y = block(x, t)
        loss = y.mean()
        loss.backward()

        assert x.grad is not None, f"{kind} input not differentiable"
        assert torch.isfinite(x.grad).all(), f"{kind} NaNs in gradient"

    @pytest.mark.parametrize("kind", KINDS) 
    def test_cross_channel(self, kind):
        cfg = cfg_setup(kind)

        # Tests ResBlock that changes number of channels
        x = torch.randn(2, 64, 16, 16)
        t = torch.randn(2, 128)
        block = get_resblock(cfg, in_ch=64, out_ch=128, time_dim=128)
        y = block(x, t)
        assert y.shape == (2, 128, 16, 16), "Channel mismatch on upscale"


    def test_invalid_variant_raises(self):
        cfg = cfg_setup("invalid")
        with pytest.raises(ValueError):
            get_resblock(cfg, in_ch=64, out_ch=64, time_dim=128)


    def test_forward_hook(self):

        cfg = cfg_setup("vanilla")

        activations = []

        def hook_fn(module, input, output):
            activations.append(output.detach())

        x = torch.randn(1, 64, 16, 16)
        t = torch.randn(1, 128)

        block = get_resblock(cfg, 64, 64, time_dim=128)
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
        t2 = torch.randn(2, 128) * 10.0     # Strongly scaled to cause difference in FiLM output

        block = FiLMResBlock(64, 64, time_dim=128, norm_type="group", attention_layer=False)

        out1 = block(x, t1)
        out2 = block(x, t2)

        diff = (out1 - out2).abs().mean().item()
        assert diff > 1e-2, f"FiLM modulation too weak, mean diff={diff:.4f}"


# ----- Normalization Test Group -----

@pytest.mark.parametrize("norm_type", NORM_TYPES)
def test_norm_layer_output_stability(norm_type):
    norm = get_norm_layer(norm_type, num_channels=64)
    x = torch.randn(8, 64, 32, 32)
    y = norm(x)
    assert torch.isfinite(y).all(), f"{norm_type} norm produced NaNs"


# --- Snapshot Test ---
@pytest.mark.parametrize("kind", KINDS)
def test_snapshot_output_consistency(kind):
    cfg = cfg_setup(kind)
    x = torch.randn(1, 32, 16, 16)
    t = torch.randn(1, 64)

    block = get_resblock(cfg, in_ch=32, out_ch=32, time_dim=64)
    out = block(x, t)

    # Compare output to a stored snapshot for regression testing
    assert_tensor_close_to_snapshot(out, f"resblock_{kind}_resblock_snapshot.pt", kind)


# --- Visualization Test ---
@pytest.mark.parametrize("kind", KINDS)
def test_activation_visualization(kind):
    cfg = cfg_setup(kind)
    x = torch.randn(1, 32, 16, 16)
    t = torch.randn(1, 128)
    block = get_resblock(cfg, 32, 32, time_dim=128)

    # Register intermediate hook
    activations = []
    def hook_fn(module, inp, out): activations.append(out.detach().cpu())

    hook = block.conv2.register_forward_hook(hook_fn)
    _ = block(x, t)
    hook.remove()

    # Plot the first channel
    act = activations[0][0, 0]
    plt.imshow(act.numpy(), cmap="viridis")
    plt.title(f"{kind} ResBlock Activation")
    plt.savefig(f"tests/snapshots/{kind}_resblock_activation.png")
    plt.close()
    