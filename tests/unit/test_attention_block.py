
# CUDA tests not tested for flash attention.

import torch
import pytest
from omegaconf import OmegaConf
from modules.attention.flash_attention import FlashAttention
from modules.attention.registry import get_attention
from modules.attention.base_attention import BaseAttention


# ------------------------ #
#   Shape + Backprop tests
# ------------------------ #

class TestFlashAttention:
    def setup_method(self):
        self.base_cfg = OmegaConf.create({
            "kind": "flash",
            "params": {
                "num_heads": 8,
                "dim": 64,
                "backend": "auto"
            }
        })
        self.x = torch.randn(2, 64, 32, 32, requires_grad=True)

    @pytest.mark.parametrize("backend", ["auto", "fallback_only"])
    def test_output_shape_cpu(self, backend):
        cfg = OmegaConf.create({**self.base_cfg, "backend": backend})
        block = get_attention(cfg)
        out = block(self.x)
        assert out.shape == self.x.shape

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for flash backend")
    @pytest.mark.parametrize("backend", ["auto", "flash_only"])
    def test_output_shape_cuda(self, backend):
        cfg = OmegaConf.merge(self.base_cfg, {
            "params": {
                "backend": backend
            }
        })
        block = get_attention(cfg).cuda()
        out = block(self.x.cuda())
        assert out.shape == self.x.shape


    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for flash backend")
    def test_backprop(self):
        cfg = OmegaConf.merge(self.base_cfg, {
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

    def test_requires_cuda(self):
        if not torch.cuda.is_available():
            cfg = OmegaConf.merge(self.base_cfg, {   
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
    def test_interface_compliance(self):
        dummy_input = torch.randn(2, 64, 32, 32)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dummy_input = dummy_input.to(device)

        for cls in [FlashAttention,]:    # other attentions added here later
            if cls is FlashAttention and not torch.cuda.is_available():
                continue
            model = cls(dim=64, num_heads=4).to(device)
            assert isinstance(model, BaseAttention)
            out = model(dummy_input)
            assert model(dummy_input).shape == dummy_input.shape

    def test_invalid_varient_raises(self):
        with pytest.raises(ValueError, match="Unknown attention type"):
            bad_cfg = OmegaConf.create({
                "kind": "invalid",
                "num_heads": 8,
                "dim": 64,
            })
            get_attention(bad_cfg)



