
import os
import torch
import torch.nn as nn
import tempfile
from utils.vanilla_checkpointing import save_checkpoint, load_checkpoint
from model.unet import UNet
from model.build_unet import build_unet_from_config
from trainer.optim_utils import build_optimizer
from trainer.ema_utils import EMA, EMASchedule
import pytest
from omegaconf import OmegaConf, omegaconf, DictConfig, ListConfig
from utils.vanilla_checkpointing import save_checkpoint_atomic


def tiny_unet_cfg():
    # Adjust to your schema; the idea is "small but real"
    
    return OmegaConf.create({
        "model": {
            "in_channels": 3,
            "base_channels": 16,
            "channel_multipliers": [1, 2], # shallow UNet
        },
        "time_embedding": {"kind": "sinusoidal", "params": {"dim": 32}},
        "resblock": {"kind": "vanilla", "params": {"norm_type": "group"}},
        "attention": {"kind": "none", "params": {"start_layer": 2}}, # keep simplest path
        "midblock": {"use_attention": False},
        "updown": {"num_layers": 2, "expect_skip": True},
        "final_head": {"out_channels": 3},
        "optim": {"optimizer": "adam", "lr": 3e-4, "betas": [0.9, 0.999]},
        "debug": {"enabled": False},
    }) 


class TinyNet(nn.Module):
    def __init__(self, in_ch=3, hidden=8, out_ch=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, hidden, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden, out_ch, 3, padding=1)

    def forward(self, x, t=None):   # t unused, keeps UNet-like signature
        return self.conv2(torch.relu(self.conv1(x)))



def fake_model_factory():
    model = TinyNet()
    cfg = OmegaConf.create({"optim": {"optimizer": "adam", "lr": 3e-4, "betas": [0.9, 0.999]},})
    return model, cfg

def full_fake_model_factory():
    model = TinyNet()
    cfg = OmegaConf.create({"optim": {"optimizer": "adam", "lr": 3e-4, "betas": [0.9, 0.999]},})
    opt = build_optimizer(params=model.parameters(), cfg=cfg) 
    ema = EMA(model)
    return model, opt, ema, cfg

def real_unet_factory():
    from model.build_unet import build_unet_from_config
    cfg = tiny_unet_cfg()
    model = build_unet_from_config(cfg) 
    return model, cfg


@pytest.mark.parametrize("model_factory", [fake_model_factory, real_unet_factory])
def test_checkpoint_roundtrip_parametrized(tmp_path, model_factory):
    model, cfg = model_factory()
    opt = build_optimizer(params=model.parameters(), cfg=cfg) 
    ema = EMA(model)

    x = torch.randn(2, 3, 16, 16)
    out = model(x, getattr(torch, "zeros", lambda *a, **k: None)(2)) if hasattr(model, "forward") else model(x)
    out = out if isinstance(out, torch.Tensor) else out[0]
    out.mean().backward()
    opt.step(); opt.zero_grad()
    ema.update(model)

    path = os.path.join(tmp_path, "checkpoint.pt")

    save_checkpoint(model=model, optimizer=opt, ema=ema, step=1, path=path)

    with torch.no_grad():
        for p in model.parameters():
            p.mul_(0)   # destroy weights


    with torch.serialization.safe_globals([
        omegaconf.ListConfig,
    ]):
        ckpt = torch.load(path, map_location="cpu", weights_only=False) # weights_only=True by default in 2.6

        def walk_types(obj, prefix=""):
            import torch
            from collections.abc import Mapping, Sequence
            bad = []
            def _walk(x, p):
                if isinstance(x, (torch.Tensor, int, float, str, type(None))):
                    return
                if isinstance(x, Mapping):
                    for k, v in x.items(): _walk(v, f"{p}.{k}" if p else str(k))
                if isinstance(x, Sequence) and not isinstance(x, (str, bytes)):
                    for i, v in enumerate(x): _walk(v, f"{p}[{i}]")
                    return
                bad.append((p, type(x)))
            _walk(obj, prefix)
            return bad
        
        offenders = walk_types(ckpt, "ckpt")
        for path_str, typ in offenders:
            print(path_str, "->", typ)

        print("\n Find Omegamconf: \n")

        def find_omegaconf(obj, prefix=""):
            bad = []
            def _walk(x, p):
                if isinstance(x, (DictConfig, ListConfig)):
                    bad.append((p, type(x)))
                    return
                if isinstance(x, dict):
                    for k,v in x.items(): _walk(v, f"{p}.[{k}]" if p else str(k))
                elif isinstance(x, list):
                    for i,v in enumerate(x): _walk(v, f"{p}[{i}]")
            _walk(obj, prefix)
            return bad
        
        print(find_omegaconf(ckpt, "ckpt"))

    load_checkpoint(model=model, optimizer=opt, ema=ema, path=path)


    saved = torch.load(path, map_location="cpu")["model"]
    for k, v in saved.items():
        assert torch.allclose(model.state_dict()[k], v)


def test_checkpoint_save_load_consistency():
    model, optimizer, ema, cfg = full_fake_model_factory()
    dummy_input = torch.randn(2, 3, 64, 64)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "ckpt.pt")
        save_checkpoint(model, optimizer, ema, epoch=5, step=100, loss=0.123, path=path)

        # Modify weights to verify restore
        for p in model.parameters():
            p.data += 1.0
        
        load_checkpoint(path=path, model=model, optimizer=optimizer, ema=ema)
        out = model(dummy_input)

        # Make sure model doesn't produce NaNs
        assert torch.isfinite(out).all(), "Model contains NaNs after load"


def test_checkpoint_restores_epoch_step_loss():
    model, optimizer, ema, cfg = full_fake_model_factory()

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "ckpt.pt")
        save_checkpoint(model, optimizer, ema, epoch=3, step=77, loss=0.42, path=path)

        meta = load_checkpoint(path, model, optimizer, ema)
        assert meta["epoch"] == 3
        assert meta["step"] == 77
        assert abs(meta["loss"] - 0.42) < 1e-6


def test_checkpoint_handles_missing_file():
    try: 
        load_checkpoint("nonexistent.pt", *full_fake_model_factory()[:3])
    except FileNotFoundError:
        assert True
    else:
        assert False, "Expected FileNotFoundError"

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for this test")
def test_checkpoint_cpu_gpu_transfer():
    model, optimizer, ema, cfg = full_fake_model_factory()

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "ckpt.pt")
        save_checkpoint(model, optimizer, ema, epoch=1, step=1, loss=0.0, path=path)


        cpu_model = TinyNet(in_ch=3, out_ch=3).cpu()
        cpu_optimizer = build_optimizer(cpu_model.parameters(), cfg)
        cpu_ema = EMA(cpu_model)

        meta = load_checkpoint(path, cpu_model, cpu_optimizer, cpu_ema, device="cpu")
        assert isinstance(meta, dict)


def test_atomic_checkpoint_interupt(tmp_path, monkeypatch):
    path = tmp_path / "ckpt.pt"
    orig = {"step": 0}
    save_checkpoint_atomic(orig, path)

    # --- Monkeypatch torch.save to simulate interruption ---
    def boom(obj, f, **kwargs):
        raise KeyboardInterrupt("simulated")
    
    monkeypatch.setattr(torch, "save", boom)

    # Attempt to save new checkpoint
    with pytest.raises(KeyboardInterrupt):
        save_checkpoint_atomic({"step": 1}, str(path))

    # After failure, the old checkpoint must still be intact
    ckpt = torch.load(path)
    assert ckpt == orig, "Atmoic save should leave original checkpoint untouched"

    # And there should be no stray temp files left
    leftovers = list(tmp_path.glob("*.tmp*"))
    assert leftovers == [], f"Leftover temp files found: {leftovers}"





