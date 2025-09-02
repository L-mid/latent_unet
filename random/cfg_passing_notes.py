
import torch
import torch.nn as nn

# === NOTES:
"""
Passing full cfg vs passing compartamentalized cfg.

This cfg management is worth looking into for the next repo.
"""



# Signatures:

"""
Where full cfg is temping, it ususally means the component is relying on global settings (device, dtype, seed, amp, dist). 
Better passed via a tiny runtime context object, not the whole config.
"""

def build_unet(unet_cfg): -> nn.Module
def get_loss_fn(loss_cfg): -> nn.Module
def build_optimizer(params, optim_cfg): -> torch.optim.Optimizer 
def build_scheduler(optimizer, sched_cfg): -> torch.optim.lr_scheduler._LRScheduler
def build_ema(model, ema_cfg): -> EMA       # note the internals would take this cfg at the top level


# Train loop "slicing"
cfg = ...  # the OmegaConf DictConfig

# 1) runtime context (explicit, small)
from dataclasses import dataclass
@dataclass(frozen=True)
class RuntimeCtx:
    device: str
    amp: bool
    dtype: str | None = None
    seed: int | None = None

ctx = RuntimeCtx(device=cfg.device, amp=cfg.training.amp)

# 2) build components from slices
model = build_unet(cfg.model).to(ctx.device)
loss_fn = get_loss_fn(cfg.losses)
opt     = build_optimizer(model.parameters(), cfg.optim)
sched   = build_scheduler(opt, cfg.schedule)
ema     = build_ema(model, cfg.get("ema", {}))      # optional block

"""
If some model bits live at the top level (e.g, attention, resblock, time_embedding). 
Prefer to nest them under cfg.model.* long-term so build_unet(cfg.model) is self contained. 
Until, either:
    Pass those sub-configs explicity: build_unet(cfg.model, attn_cfg=cfg.attention, res_cfg=cfg.resblock, temb_cfg=cfg.time_embedding)
    OR: Create a small view/merged dict for model inside the train loop.
"""

# Validation boundries
# Validate each slice at the edge:

def get_loss_fn(loss_cfg):
    # assert required keys exist and types are correct
    typ = loss_cfg.get("type", "mse")
    ...

# Use OmegaConf structured nodes or dataclasses for each slice to catch typos early.


# When is passing full cfg ok?
"""
- tiny scripts or quick experimentations.
- A top level "composition/builder" whose only job is to fan out to sub-builders (even there, it should immediately slice).
But for production-ish code, prefer sliced config + explicit deps.
"""

"""
loss_fn = get_loss_fn(cfg.losses)   ✅ (best practice)
loss_fn = get_loss_fn(cfg)          ❌ (convenient, but tightly coupled)
keep globals (device/amp/seed) in small RuntimeCtx and pass it explicitly to the few places that need it.
"""