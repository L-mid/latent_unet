
import torch
from torch import optim
from torch.cuda.amp import GradScaler
from typing import Any, Dict


# ----------------------------------
# EMA
# ----------------------------------

class EMA:
    def __init__(self, model, decay=0.9999, device=None):
        self.model = model
        self.decay = decay
        self.ema_model = self.clone_model()
        self.ema_model.eval()

    def _clone_model(self):
        import copy
        ema = copy.deepcopy(self.model)
        for p in ema.parameters():
            p.requires_grad = False
        return ema
    
    @torch.no_grad()
    def update(self):
        msd = self.model.state_dict()
        for k, v in self.ema_model.state_dict().items():
            if k in msd:
                msd_k = msd[k].detach()
                v.copy_(v * self.decay + (1. - self.decay) * msd_k)

    def state_dict(self):
        return self.ema_model.state_dict()
    

# --------------------------
# Optimizer Builder
# --------------------------

def build_optimizer(params, cfg):
    if cfg.optim.optimizer == "adam":
        return optim.Adam(params, lr=cfg.optim.lr, betas=cfg.optim.betas)
    elif cfg.optim.optimizer == "adamw":
        return optim.AdamW(params, lr=cfg.optim.lr, betas=cfg.optim.betas, weight_decay=cfg.optim.weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {cfg.optim.optimizer}")
    

# --------------------------
# Scheduler Builder
# --------------------------

def build_scheduler(optimizer, cfg, total_steps):
    if cfg.optim.scheduler == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    elif cfg.optim.lr_scheduler == "linear":
        return optim.lr_scheduler.LinearLR(optimizer, total_iters=total_steps)
    elif cfg.optim.scheduler == "step":
        return optim.lr_scheduler.StepLR(optimizer, step_size=cfg.optim.step_size, gamma=cfg.optim.gamma)
    else: 
        return None


# ---------------------------
# AMP Support
# ---------------------------

def build_amp(cfg):
    if cfg.training.amp:
        return torch.amp.GradScaler()
    return


# ----------------------------
# Grad Clipping Utility
# ----------------------------

def apply_grad_clipping(model, clip_value):
    if clip_value > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
    










