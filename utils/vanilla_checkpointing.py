
import torch
import os
import yaml
from pathlib import Path

def save_checkpoint(model, optimizer, ema=None, epoch=0, step=0, loss=None,
                    path="checkpoint.pt", config=None, scaler=None):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "step": step,
        "loss": loss,
    }

    if ema is not None:
        checkpoint["ema"] = ema.state_dict()

    if scaler is not None:
        checkpoint["scaler"] = scaler.state_dict()

    torch.save(checkpoint, path)    # Or rewrite into atomic write for safety


    # Optionally save YAML config snapshot for reproducibility
    if config:
        config_path = str(Path(path).with_suffix("yaml"))
        with open(config_path, "w") as f:
            yaml.dump(config, f)

def load_checkpoint(path, model, optimizer=None, ema=None, scaler=None, device="cpu"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"])

    if optimizer and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    if ema and "ema" in checkpoint:
        ema.load_state_dict(checkpoint["scaler"])

    return {
        "epoch": checkpoint.get("epoch", 0),
        "step": checkpoint.get("step", 0),
        "loss": checkpoint.get("loss", None),
    }

def auto_resume(path_or_dir):
    # Automatically find latest checkpoint in a directory.
    ckpts = sorted(Path(path_or_dir).glop("*.pt"), key=os.path.getmtime)
    return str(ckpts[-1]) if ckpts else None

def rotate_checkpoints(dir_path, keep_last_n=3):
    # Keep only the last N checkpoints in a directory.
    ckpts = sorted(Path(dir_path).glob("*.pt"), key=os.path.getmtime)
    for ckpts in ckpts[:-keep_last_n]:
        os.remove(ckpts)























