
import os, random, math
import numpy as np
import time
import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import OmegaConf

from model.config import load_config
from model.build_unet import build_unet_from_config
from data.simple_dataset_loader import load_cifar_dataset
from trainer.train_loop import train_loop
from utils.checkpointing.zarr_checkpointing import zarr_wrapper

import concurrent.futures as cf
import multiprocessing as mp
import pytest
#from tests.helpers_subproc import enforce_timeout
#from tests._hard_timeout import run_with_hard_timeout
from functools import partial
from pathlib import Path



def tiny_hi():      # makes sure logging is collected
    import logging, time
    logging.getLogger("ckpt").info("hello from child"); time.sleep(0.1)

    # make sure files are made?



# === NOTES:
"""
Goal: prove the full loop runs: (data, model, loss, backward, optimizer, scheduler, hooks, checkpoint, resume) in seconds. 
Might make a simpler dataloader.

OMP_NUM_THREADS=1, MKL_NUM_THREADS=1 (env varibales to set)

do visualizer in the loop next.

"""

@pytest.mark.skip("Not currently used")
@pytest.mark.timeout(0)     # for trace stacks and Lunix CI only
def test_timeout_sanity():
    pass
    # check logging works
    #run_with_hard_timeout(tiny_hi, timeout=20)

    #job = partial(enforce_timeout, holdoff=999)   # fn must be module-level
    #with pytest.raises(TimeoutError):
        #_ = run_with_hard_timeout(job, timeout=0.1)


@pytest.mark.timeout(0)
@pytest.mark.smoke
def test_end_to_end_train_smoke(tmp_path, timeout=60):
    timeout = timeout

    # --- 0) Load config
    cfg = load_config("configs/unet_config.yaml")
    cfg = OmegaConf.create(cfg)

    cfg.device = 'cuda'  #smoke test stays on CPU for CI speed/stability
    device = cfg.device

    """
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
    
    monkeypatch.setenv("WANDB_MODE", "disabled)
    monkeypatch.setenv("WANDB_SILENT", true)
    
    #os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    """

    # --- 2) data ---
    cfg.model.image_size = 32
    cfg.batch_size = 32
    dataset = load_cifar_dataset(cfg, only_return_dataset=True)

    # --- 3) model ---
    cfg.model.base_channels = 64
    cfg.debug.enabled = False
    cfg.model.channel_multipliers = [1, 2, 3, 4]
    model = build_unet_from_config(cfg).to(device) 

    # --- 4) One tiny epoch ---
    cfg.training.amp = False
    cfg.gradient_checkpointing = False
    cfg.training.grad_clip = None
    cfg.logging.use_wandb = True; cfg.logging.use_tb = True

    cfg.training.num_epochs = 999     
    cfg.training.num_workers = 0

    # ckpts and viz and log (per epoch)
    cfg.viz.enabled = True

    cfg.training.ckpt_interval = 30
    cfg.training.viz_interval = 15

    cfg.checkpoint.out_dir = "ckpts"     # "C:/_ckpts" for off onedrive 
    cfg.viz.output_dir = "viz"
    cfg.logging.output_dir = "logs"


    cfg.resume_path = cfg.checkpoint.out_dir  # loops

    model.train()

    # the below is for timing out (good for slowdowns)
    """
    # to run train_loop through timeout
    job = partial(train_loop, cfg, model, dataset)          # passes in args
    logs = run_with_hard_timeout(job, timeout=timeout, stderr_output=True)      # this fn shuts down bad windows async stuff. One day i will debug tensorstore
    """

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
    from trainer.optim_utils import EMA
    model2 = build_unet_from_config(cfg).to(device)
    opt2 = optim.AdamW(model2.parameters(), lr=1e-3)
    sch2 = optim.lr_scheduler.StepLR(opt2, step_size=1, gamma=0.9)
    ema2 = EMA(model2)

    # checkpoint finder helper
    from utils.checkpointing.ckpt_io import resolve_latest_checkpoint
    cfg.resume_path = cfg.checkpoint.out_dir
    correct_path = resolve_latest_checkpoint(cfg.resume_path)

    zarr_wrapper.load_model(model2, opt2, sch2, ema2, path=correct_path)    


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








