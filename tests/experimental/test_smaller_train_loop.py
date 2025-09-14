
import torch
import pytest
from omegaconf import OmegaConf
from pathlib import Path

from small_train_loop import train_loop
from model.build_unet import build_unet_from_config
from torch.utils.data import Dataset

from trainer.logger import NoopLogger, ExperimentLogger, build_logger


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


def make_tmp_dir():
    import tempfile
    return tempfile.mkdtemp()


def small_cfg():                # an attempt to simplify the min viable cfg
    tmp_dir = make_tmp_dir()

    cfg = OmegaConf.create({
        "resume_path": f"{tmp_dir}/saved_checkpoints",
        "device": "cpu",  
        "model": {
            "image_size": 32,
            "in_channels": 3,
            "base_channels": 64,
            "channel_multipliers": [1, 2, 3],
        },
        "attention": {
            "kind": "vanilla",
            "params": {
                "num_heads": 8,
                "norm_groups": 8,
                "midblock": True,
                "start_layer": 3,   # or wherever 16x16 is
            },
        },
        "resblock": {
            "kind": "vanilla",
            "params": {
                "norm_type": "group"  
            },
        },
        "time_embedding": {
            "kind": "sinusoidal",
            "params": {
                "dim": 512
            },
        },
        "midblock": {
            "use_attention": False,
            "num_layers": 1,
        },
        "updown": {
            "num_layers": 1,                    # each layer is a resblock
        },
        "final_head": {
            "out_channels": 3,
        },
        "schedule": {
            "schedule_type": "cosine",               # or linear
            "timesteps": 1000,                  
            "beta_start": 1e-4,
            "beta_end": 0.02,
        }, 
        "optim": {
            "optimizer": "adamw",
            "lr": 2e-4,
            "betas": [0.9, 0.999],
        },
        "losses": {
            "type": "mse",
        },
        "training": {
            "batch_size": 64,
            "num_epochs": 150,    
            "vis_interval": 10,   # every N epochs
            "ckpt_interval": 5,   # every N epochs 
        },
        "checkpoint": {
            "backend": "zarr",
            "out_dir": f"{tmp_dir}/saved_checkpoints"    
        },
        "debug": {
            "enabled": False,
        },
        "logging": {
            "use_wandb": False,
            "use_tb": False,
            "project_name": "latent_unet-v1",
            "run_name": "test_minimal"      # maybe add someway to control logging dir? (right now is automatic i think)
        },
        "visualization": {
            "enabled": False,
            "output_dir": f"{tmp_dir}/vis"    
        },
    }
)
    
    return cfg

@pytest.mark.xfail(reason="debbuging asserts tripping correct behavoir in this instance but not others.")
def test_train_step():

    cfg = small_cfg()
    cfg.training.num_epochs = 1

    device = cfg.device
    model = build_unet_from_config(cfg) # might need to device

    image_size = cfg.model.image_size

    data=DummyDataset(image_shape=(3, image_size, image_size)) 
    logger=ExperimentLogger(cfg)

    train_loop(cfg, model, data, logger)


def test_resume_end_to_end():
    tmp_path = make_tmp_dir()
    tmp_path = Path(tmp_path)

    run_dir = tmp_path / "run1"
    ckpt_dir = run_dir / "ckpt"
    viz_dir = run_dir / "viz"
    ckpt_dir.mkdir(parents=True)
    viz_dir.mkdir()

    base_cfg = small_cfg()

    model = build_unet_from_config(base_cfg) 
    image_size = base_cfg.model.image_size
    data=DummyDataset(image_shape=(3, image_size, image_size)) 
    logger=ExperimentLogger(base_cfg)


    # First run: train 1 epoch and save
    cfg = small_cfg()
    cfg.training.num_epochs = 1
    cfg.training.ckpt_interval = 1  

    cfg.resume_path = str(ckpt_dir)
    cfg.checkpoint.out_dir = str(ckpt_dir) # does actually save

    train_loop(cfg, model, data, logger)  # should write meta.json with epoch_next=1


    # Second run: resume and do another epoch
    cfg2 = small_cfg()
    cfg2.training.num_epochs = 1  
    cfg2.training.ckpt_interval = 1   

    cfg2.resume_path = str(ckpt_dir) # should exist

    assert Path(cfg2.resume_path).exists() 
    print(Path(cfg2.resume_path), "| This is in the test, resume path exists and is the same")

    cfg2.checkpoint.out_dir = str(ckpt_dir)  # continue in same place #

    logs = train_loop(cfg2, model, data, logger)
    assert logs["start_epoch"] == 1












