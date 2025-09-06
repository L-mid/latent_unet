


import torch
import pytest
from trainer.train_loop import train_loop
from model.build_unet import build_unet_from_config
from diffusion.ddpm import DDPM
from trainer.losses import get_loss_fn 
from trainer.optim_utils import build_optimizer              
from diffusion.schedule import get_diffusion_schedule
from torch.utils.data import Dataset

from typing import Mapping, Sequence
from types import SimpleNamespace
from omegaconf import OmegaConf
import types

from modules.residual_block import get_resblock

# === NOTES:
"""
SimpleNamespaces are likly not the way to go in future, make a dict and turn it to OmegaConf better.
I like the dummy dataset :)

Really slow load off internet access, import train_loop could cause slow wandb init failure

Does NOT test loggers/checkpointers/extras. Somewhere should.
"""


class DummyDataset(Dataset):

    def __init__(self, num_samples=100, image_shape=(3, 4, 4), num_classes=10):
        self.num_samples = num_samples
        self.image_shape = image_shape
        self.num_samples = num_classes
    
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Random "image"
        img = torch.randn(*self.image_shape)
        # Random "label"
        label = torch.randint(0, self.num_samples, (1,)).item()
        return {
            "image": img,
            "label": label,
            }


def to_plain(obj):
    """Recursively convert SimpleNamespace, DotDict, Mapping, and Sequence into plain dict/list/primitives suitble for OmegaConf.create()"""
    # 1) SimpleNamespace -> dict
    if isinstance(obj, types.SimpleNamespace):
        return {k: to_plain(v) for k, v in vars(obj).items()}
    
    # 2) Mapping (dict, DotDict, other dict-like) -> dict
    if isinstance(obj, Mapping): 
        return {k: to_plain(v) for k, v in obj.items()}
    
    # 3) Sequence (list/tuple), but not str/bytes -> list
    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
        return [to_plain(v) for v in obj]
    
    # 4) Everything else: leave as is (int/float/bool/str/None)
    return obj

def make_tmp_dir():
    import tempfile
    return tempfile.mkdtemp()

    
def make_dummy_cfg():
    tmp_dir = make_tmp_dir()

    return SimpleNamespace(  
        resume_path=None,  
        model=SimpleNamespace(
            image_size=4,
            in_channels=3,
            base_channels=8,
            channel_multipliers=[1, 2],
        ),
        attention=SimpleNamespace(
            kind="vanilla",
            params=SimpleNamespace(
                num_heads=8,
                norm_groups=8,
                start_layer=999,
                window_size=4,
                midblock=False,
                backend="auto"
            ),
        ),
        resblock=SimpleNamespace(
            kind="vanilla",
            params=SimpleNamespace(
                norm_type="group"  
            ),
        ),
        time_embedding=SimpleNamespace(
            kind="sinusoidal",
            params=SimpleNamespace(
                dim=32
            ),
        ),
        midblock=SimpleNamespace(
            use_attention=False,
            num_layers=1,
        ),
        updown=SimpleNamespace(
            use_skip_connections=True,
            num_layers=1,
            expect_skip=True,
        ),
        final_head=SimpleNamespace(
            out_channels=3,
        ),
        schedule=SimpleNamespace(
            schedule="cosine",
            timesteps=4,                     # timesteps set to 4
            beta_start=1e-4,
            beta_end=0.02,
        ),
        optim=SimpleNamespace(
            optimizer="adam",
            lr=3e-4,
            betas=[0.9, 0.999],
        ),
        losses=SimpleNamespace(
            type="mse",
            weight=1.0,
            schedule=SimpleNamespace(
                type="linear",                  # should be renamed loss_schedule maybe
            ),
        ),
        training=SimpleNamespace(
            batch_size=2,
            num_epochs=1,
            amp=True,
            grad_clip=1.0,
            vis_interval=1,    # every N epochs
            ckpt_interval=1,   # every N epochs
        ),
        checkpoint=SimpleNamespace(
            backend="noop",
            save_every=1000,    # 1000 what?
            out_dir=tmp_dir     # might need to str
        ),
        debug=SimpleNamespace(
            enabled=False,
        ),
        logging=SimpleNamespace(
            use_wandb = False,
            project_name = "latent_unet-v1",
            run_name="test_minimal"
        ),
        visualization=SimpleNamespace(
            enabled=False,
            output_dir=tmp_dir,    # make this tmpdir.
        ),
    )

@pytest.mark.skipif(not torch.cuda.is_available(), reason="training loop too slow on cpu")
def test_train_one_step_runs():
    device = "cuda"
    cfg_ns = make_dummy_cfg()
    plain = to_plain(cfg_ns)
    cfg = OmegaConf.create(plain)


    # omegaconf asignment methods (for device as example):

    # 1) direct assignment:
    cfg.device = device

    # 2) OmegaConf.merge:
    cfg = OmegaConf.merge(cfg, {"device": device})

    # 3) CLI-style "dotlist" overrides
    overrides = [f"device={device}"]
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))

    # 4) Update by dotted path
    OmegaConf.update(cfg, "device", device)
    # Can also do deeper paths e.g.: (cfg, "resblock.params.norm_type", batch)


    model = build_unet_from_config(cfg).to(device)
 
    dataset = DummyDataset()

    logs = train_loop(
        cfg=cfg,
        model=model,
        dataset=dataset,
    )

    assert "loss" in logs
    assert isinstance(logs["loss"], float), "loss should be a float"
    # I killed the logger. Assess answer.

def test_dummy_dataset_loading():
    loader = torch.utils.data.DataLoader(DummyDataset(num_samples=3), batch_size=4)
    for batch in loader:
        assert "image" in batch and batch["image"].shape == (4, 3, 4, 4)  
        break








