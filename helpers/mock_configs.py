
from model.config import load_config    # was UNetConfig, likely when dataclassing


# === NOTES
"""
Not used, not even accurate, but prob a good idea.
As a fixture might be even better.
"""

# ---- Minimal dummy config for unit tests ----
def create_mock_config():
    # Returns a minimal vaild UNetConfig object.

    cfg = {
        "model": {
            "in_channels": 3,
            "out_channels": 3,
            "base_channels": 32,
            "ch_mults": [1, 2],
        },
        "time_embedding": {
            "kind": "sinusoidal",
            "dim": 128
        },
        "resblock": {
            "kind": "vanilla",
            "norm": "group"
        },
        "attention": {
            "kind": "vanilla",
            "start_layer": 1,
        },
        "schedule": {
            "schedule_type": "linear",
            "timesteps": 10,
            "beta_start": 1e-4,
            "beta_end": 0.02,
        },
        "optim": {
            "lr": 1e-4,
            "weight_decay": 0.0,
        },
        "training": {
            "batch_size": 2,
            "grad_clip": 1.0,
            "num_epochs": 1,
            "gradient_checkpointing": False,
            "amp": False
        }
    }
    return cfg



# ---- Variant for bigger model tests ----
def create_large_mock_config():
    # Returns a larger but still fast-to-test config. 
    cfg = {
        "model": {
            "in_channels": 3,
            "out_channels": 3,
            "base_channels": 64,
            "ch_mults": [1, 2, 4],
        },
        "time_embedding": {
            "kind": "sinusoidal",
            "dim": 256
        },
        "resblock": {
            "kind": "film",
            "norm": "batch"
        },
        "attention": {
            "kind": "window",
            "start_layer": 2,
        },
        "schedule": {
            "schedule_type": "cosine",
            "timesteps": 20,
            "beta_start": 1e-4,
            "beta_end": 0.02,
        },
        "optim": {
            "lr": 5e-5,
            "weight_decay": 0.01
        },
        "training": {
            "batch_size": 2,
            "grad_clip": 1.0,
            "num_epochs": 2,
            "gradient_checkpointing": True,
            "amp": True
        }
    }
    return cfg



def accurate_cfg():     # at least at one point was

    cfg = {  
        "resume_path": None,  
        "model": {
            "image_size": 4,
            "in_channels":3,
            "base_channels": 8,
            "channel_multipliers": [1, 2],
        },
        "attention": {
            "kind": "vanilla",
            "params": {
                "num_heads": 8,
                "norm_groups": 8,
                "start_layer": 999,
                "window_size": 4,
                "midblock": False,
                "backend": "auto"
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
                "dim": 32
            },
        },
        "midblock": {
            "use_attention": False,
            "num_layers": 1,
        },
        "updown": {
            "use_skip_connections": True,
            "num_layers": 1,
            "expect_skip": True,
        },
        "final_head": {
            "out_channels": 3,
        },
        "schedule": {
            "schedule": "cosine",
            "timesteps": 4,                     # timesteps set to 4
            "beta_start": 1e-4,
            "beta_end": 0.02,
        },
        "optim": {
            "optimizer": "adam",
            "lr": "3e-4",
            "betas": "[0.9, 0.999]",
        },
        "losses": {
            "type": "mse",
            "weight": 1.0,
            "schedule": {
                "type": "linear",                  # should be renamed loss_schedule maybe
            },
        },
        "training": {
            "batch_size": 2,
            "num_epochs": 1,
            "amp": True,
            "grad_clip": 1.0,
            "vis_interval": 1,    # every N epochs
            "ckpt_interval": 1,   # every N epochs
        },
        "checkpoint": {
            "backend": "noop",
            "save_every": 1000,    # 1000 what?
            "out_dir": None     # might need to str
        },
        "debug": {
            "enabled": False,
        },
        "logging": {
            "use_wandb": False,
            "use_tb": False,
            "project_name": "latent_unet-v1",
            "run_name": "test_minimal"
        },
        "visualization": {
            "enabled": False,
            "output_dir": None,    
        },
    }

    return cfg