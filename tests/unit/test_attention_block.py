
# CUDA tests not tested for flash attention.

import torch
import pytest
from omegaconf import OmegaConf
from modules.attention.vanilla_attention import VanillaAttention
from modules.attention.flash_attention import FlashAttention
from modules.attention.base_attention import BaseAttention
from helpers.test_utils import controlled_test
from modules.attention.registry import get_attention

CATEGORY = "unit"
MODULE = "attention_block"


def make_attention_cfg(kind: str = "vanilla", overrides=None):
    base_cfg = {
        "kind": kind,
        "params": {
            "dim": 64,
            "num_heads": 4,
            "norm_groups": 8,
            "dim_head": None,
            "start_layer": 0,
            "window_size": None,
            "backend": "auto"
        }
    }

    if overrides:
        return OmegaConf.merge(OmegaConf.create(base_cfg), OmegaConf.create(overrides))
    return OmegaConf.create(base_cfg)

# ------------------------ #
#   Shape + Backprop tests
# ------------------------ #

class TestVanillaAttention:
    def setup_method(self):
        self.cfg = make_attention_cfg("vanilla")
        self.x = torch.randn(2, 64, 32, 32, requires_grad=True)

    def test_output_shape(self):
        block = get_attention(self.cfg)
        out = block(self.x)
        assert out.shape == self.x.shape

    def test_backprop(self):
        block = get_attention(self.cfg) 
        out = block(self.x)
        out.mean().backward()
        assert self.x.grad is not None
        assert torch.isfinite(self.x.grad).all()


class TestFlashAttention:
    def setup_method(self):
        self.base_flash_cfg = make_attention_cfg("flash", {
            "params": {
                "backend": "auto",
            }
        })
        self.x = torch.randn(2, 64, 32, 32, requires_grad=True)

    @pytest.mark.parametrize("backend", ["auto", "fallback_only"])
    @controlled_test(CATEGORY, MODULE)
    def test_output_shape_cpu(self, backend, test_config):
        cfg = OmegaConf.merge(self.base_flash_cfg, {
            "params": {
                "backend": backend
            }
        })
        block = get_attention(cfg)
        out = block(self.x)
        assert out.shape == self.x.shape

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for flash backend")
    @pytest.mark.parametrize("backend", ["auto", "flash_only"])
    @controlled_test(CATEGORY, MODULE)
    def test_output_shape_cuda(self, backend, test_config):
        cfg = OmegaConf.merge(self.base_flash_cfg, {
            "params": {
                "backend": backend
            }
        })
        block = get_attention(cfg).cuda()
        out = block(self.x.cuda())
        assert out.shape == self.x.shape


    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for flash backend")
    @controlled_test(CATEGORY, MODULE)
    def test_backprop(self, test_config):
        cfg = OmegaConf.merge(self.base_flash_cfg, {
            "params": {
                "backend": "auto"
            }
        })
        block = get_attention(cfg).cuda()

        x = self.x.cuda().detach().requires_grad_()

        out = block(x)
        out.sum().backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    @controlled_test(CATEGORY, MODULE)
    def test_requires_cuda(self, test_config):
        if not torch.cuda.is_available():
            cfg = OmegaConf.merge(self.base_flash_cfg, {   
                "params": {
                    "backend": "flash_only"
                }
            })
            with pytest.raises(RuntimeError, match="FlashAttention requires CUDA"):
                FlashAttention(**cfg.params)(self.x)


# ---------------------------- #
#   API, BUILD, EGDE CASES
# ---------------------------- #

class TestAttentionSystemAPI:
    @controlled_test(CATEGORY, MODULE)
    def test_interface_compliance(self, test_config):
        dummy_input = torch.randn(2, 64, 32, 32)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dummy_input = dummy_input.to(device)

        for cls in [VanillaAttention, FlashAttention]:    # other attentions added here later
            if cls is FlashAttention and not torch.cuda.is_available():
                continue
            model = cls(dim=64).to(device)     
            assert isinstance(model, BaseAttention)
            out = model(dummy_input)
            assert model(dummy_input).shape == dummy_input.shape

    @controlled_test(CATEGORY, MODULE)
    def test_invalid_varient_raises(self, test_config):
        with pytest.raises(ValueError, match="Unknown attention type"):
            bad_cfg = OmegaConf.create({
                "kind": "invalid",
                "num_heads": "78",
                "dim": 64,
            })
            get_attention(bad_cfg)



