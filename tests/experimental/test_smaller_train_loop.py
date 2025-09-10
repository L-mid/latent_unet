
import torch
from omegaconf import OmegaConf

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
        "resume_path": f"{tmp_dir}/resume_path",
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
            "timesteps": 1000,                  # timesteps set to 4
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
            "out_dir": f"{tmp_dir}/saved_checkpoints"   # should this = the resume checkpt?   
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

def test_train_step():

    cfg = small_cfg()

    device = cfg.device
    model = build_unet_from_config(cfg) # might need to device

    data=None
    logger=None

    train_loop(cfg, model, data, logger)















