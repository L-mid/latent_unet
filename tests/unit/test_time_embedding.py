
import pytest
import torch
from modules.time_embedding import get_time_embedding
from helpers.test_utils import controlled_test
from omegaconf import OmegaConf


CATEGORY = "unit"
MODULE = "time_embedding"


@pytest.mark.parametrize("kind", ["sinusoidal", "learned", "film"])
@pytest.mark.parametrize("dim", [128, 256])
@controlled_test(CATEGORY, MODULE)
def test_embedding_output_shape(kind, dim, test_config):
    batch_size = 8
    timesteps = torch.randint(0, 1000, (batch_size,))

    # Build mock config expected by get_time_embedding
    cfg = OmegaConf.create({
        "kind": kind,
        "dim": dim, 
    })

    emb_layer = get_time_embedding(cfg)
    emb = emb_layer(timesteps)

    if kind == "film":
        assert isinstance(emb, tuple), f"{kind} should return (scale, shift) tuple"
        scale, shift = emb
        assert scale.shape == (batch_size, dim)
        assert shift.shape == (batch_size, dim)
        assert scale.isfinite().all()
        assert shift.isfinite().all()
    else:
        assert emb.shape == (batch_size, dim), f"{kind} embedding output wrong shape"
        assert emb.isfinite().all(), f"{kind} embedding has NaNs or Infs"


@controlled_test(CATEGORY, MODULE)
def test_sinusoidal_embedding_consistency(test_config):
    from modules.time_embedding import SinusoidalTimeEmbedding
    emb1 = SinusoidalTimeEmbedding(dim=128)
    emb2 = SinusoidalTimeEmbedding(dim=128)

    t = torch.Tensor([0, 1, 2])
    out1 = emb1(t)
    out2 = emb2(t)

    # Sinusoidal is non-parametric => deterministic
    assert torch.allclose(out1, out2, atol=1e-6)

@controlled_test(CATEGORY, MODULE)
def test_learned_embedding_trains(test_config):
    from modules.time_embedding import LearnedTimeEmbedding

    emb = LearnedTimeEmbedding(max_steps=1000, dim=64)
    t = torch.randint(0, 1000, (4,))
    out = emb(t)
    loss = out.sum()
    loss.backward()

    assert emb.embed.weight.grad is not None
    assert emb.embed.weight.grad.shape == emb.embed.weight.shape

















