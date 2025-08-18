
import torch
from torch import optim
from torch.cuda.amp import GradScaler
from typing import Any, Dict
from omegaconf import OmegaConf


# ----------------------------------
# EMA
# ----------------------------------

# There is an alternative EMA class!
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
    oc = OmegaConf.to_container(cfg.optim, resolve=True)    # plain dict/list
    name    = str(oc.get("optimizer", "adamw")).lower()
    lr      = float(oc.get("lr", 1e-3))
    betas   = tuple(oc.get("betas", (0.9, 0.999)))      # force tuple
    wd      = float(oc.get("weight_decay", 0.0))
    eps     = float(oc.get("eps", 1e-8))

    if cfg.optim.optimizer == "adam":
        return optim.Adam(params, lr=lr, betas=betas, eps=eps)
    elif cfg.optim.optimizer == "adamw":
        return optim.AdamW(params, lr=lr, betas=betas, weight_decay=wd, eps=eps)
    else:
        raise ValueError(f"Unsupported optimizer: {name}")
    

# --------------------------
# Scheduler Builder
# --------------------------

def build_scheduler(optimizer, cfg):
    sc = OmegaConf.to_container(cfg.schedule, resolve=True)
    kind = str(sc.get("kind", "steplr")).lower()

    if kind == "cosine":
        T_max = int(sc.get("t_max", 1000))
        eta_min = float(sc.get("eta_min", 0.0))
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
    elif kind == "linear":
        total_steps = int(sc.get("total_steps", 1000))
        return optim.lr_scheduler.LinearLR(optimizer, total_iters=total_steps)
    elif kind == "step":
        step_size   = int(sc.get("step_size", 1000))
        gamma       = float(sc.get("gamma", 0.1))
        return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
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
    










