
import os, random, math
import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import OmegaConf

from model.config import load_config
from model.build_unet import build_unet_from_config
from data.simple_dataset_loader import load_cifar_dataset
from trainer.train_loop import train_loop
from utils.checkpointing.tensorstore_checkpointing import tensorstore_wrapper


# === NOTES:
"""
Goal: prove the full loop runs: (data, model, loss, backward, optimizer, scheduler, hooks, checkpoint, resume) in seconds. 
Might make a simpler dataloader.
"""

@pytest.mark.smoke
@pytest.mark.timeout(60)    # prevent hangs from blocking CI
def test_end_to_end_train_smoke(tmp_path):

    # --- 0) Load config
    cfg = load_config("configs/unet_config.yaml")
    cfg = OmegaConf.create(cfg)

    cfg.device = 'cpu'  #smoke test stays on CPU for CI speed/stability
    device = cfg.device


    # --- 1) Determinism + fast CPU settings
    seed = 1234
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(1)
    try:
        torch.use_deterministic_algorithms(True)
    except(Exception):
        pass
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Disable external loggers.
    # not implimented. 
    """
    monkeypatch.setenv("WANDB_MODE", "disabled)
    monkeypatch.setenv("WANDB_SILENT", true)
    """
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

    # --- 2) data ---
    cfg.model.image_size = 8
    cfg.batch_size = 2
    dataset = load_cifar_dataset(cfg, subset_size=2, only_return_dataset=True)

    # --- 3) model ---
    cfg.model.base_channels = 8
    cfg.debug.enabled = False
    model = build_unet_from_config(cfg).to(device) 

    # --- 4) One tiny epoch ---
    cfg.training.amp = False
    cfg.gradient_checkpointing = False
    cfg.training.grad_clip = None
    cfg.use_wandb = False; cfg.use_tb = False

    cfg.training.num_epochs = 1
    cfg.training.num_workers = 0
    cfg.training.ckpt_interval = 1

    cfg.model.channel_multipliers = [1, 2]

    cfg.checkpoint.out_dir = "C:/_ckpts/training_baseline_flash"

    model.train()

    

    logs = train_loop(cfg, model, dataset)


    # --- 6) Eval path works ---
    model.eval()
    with torch.no_grad():
        batch = next(iter(dataset))
        x = batch[0].unsqueeze(0).to(device)
        t = torch.randint(low=0, high=cfg.schedule.timesteps, size=(x.size(0),), device=x.device, dtype=torch.long)
        out = model(x, t)           # one day, maybe make a flag to allow t=None
        assert out.shape == x.shape
        assert out.dtype == x.dtype

    # Fresh model/opt to verify load actually restores parameters
    model2 = build_unet_from_config(cfg).to(device)
    opt2 = optim.AdamW(model2.parameters(), lr=1e-3)
    sch2 = optim.lr_scheduler.StepLR(opt2, step_size=1, gamma=0.9)

    from pathlib import Path

    def ckpt_keys_from_fs(root):
        # Collect leaf array node names under "model/"
        base = Path(root) / "model"
        keys = []
        for z in base.rglob("zarr.json"):
            rel = z.parent.relative_to(base)  # e.g., "mid.res2.time_proj.weight"
            keys.append(str(rel).replace("\\", "/"))  # keep dotted names as-is
        return sorted(keys)

    def model_keys(model):
        return sorted(model.state_dict().keys())

    ckpt_root = r"C:/_ckpts/training_baseline_flash"
    ck = ckpt_keys_from_fs(ckpt_root)
    mk = model_keys(model)

    missing = sorted(set(mk) - set(ck))
    unexpected = sorted(set(ck) - set(mk))
    print("Missing in ckpt:", missing[:20])
    print("Unexpected (only in ckpt):", unexpected[:20])
    print(f"counts: model={len(mk)}  ckpt={len(ck)}")




    cfg.resume_path = cfg.checkpoint.out_dir
    #state = tensorstore_wrapper.load_checkpoint(model2, opt2, sch2, path=cfg.resume_path)
    #assert state is not None
    # we had an abort check the handles


    # Validate model weights identical after reload
    for name, orig_tensor in model.state_dict().items():
        reloaded_tensor = model2.state_dict()[name]
        assert torch.allclose(orig_tensor, reloaded_tensor, atol=1e-6), f"Tensor mismatch: {name}"


    # Param parity check with small tolerance
    for (n1, p1), (n2, p2) in zip(model.named_parameters(), model2.named_parameters()):
        assert n1 == n2
        assert torch.allclose(p1, p2, atol=1e-7), f"Mismatch in {n1}"


    # --- 8) End of test sanity ---
    del dataset, model, model2








