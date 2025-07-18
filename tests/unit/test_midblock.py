
import torch
import pytest
from modules.mid_block import MidBlock
from omegaconf import OmegaConf

class TestMidBlock:
    @pytest.fixture
    def dummy_cfg(self):
        cfg = {
            "dim": 64,
            "time_embed_dim": 128,
            "midblock": {
                "use_attention": True,
            },
            "resblock": {
                "kind": "vanilla",
                "params": {
                    "norm_type": "group"
                }
            },
            "attention": {
                "kind": "vanilla",
                "params": {
                    "dim": 64,
                    "heads": 4,
                    "dim_head": 32
                }
            }
        }
        return OmegaConf.create(cfg)
    
    @pytest.fixture
    def input_tensor(self, dummy_cfg):
        B, C, H, W = 2, dummy_cfg.dim, 32, 32
        return torch.randn(B, C, H, W)
    
    @pytest.fixture
    def t_emb(self, dummy_cfg):
        B = 2
        return torch.randn(B, dummy_cfg.time_embed_dim)
    
    def test_midblock_with_attention(self, dummy_cfg, input_tensor, t_emb):
        block = MidBlock(
            dim=dummy_cfg.dim,
            time_embed_dim=dummy_cfg.time_embed_dim,
            resblock=dummy_cfg.resblock,
            midblock=dummy_cfg.midblock,
            attention=dummy_cfg.attention
        )
        out = block(input_tensor, t_emb)
        assert out.shape == input_tensor.shape

    def test_midblock_without_attention(self, dummy_cfg, input_tensor, t_emb):
        dummy_cfg["midblock"]["use_attention"] = False
        block = MidBlock(
            dim=dummy_cfg.dim,
            time_embed_dim=dummy_cfg.time_embed_dim,
            midblock=dummy_cfg.midblock,
            resblock=dummy_cfg.resblock,
            attention=dummy_cfg.attention
        )

        out = block(input_tensor, t_emb)
        assert out.shape == input_tensor.shape




