
import torch
import torch.nn.functional as F
from torch import nn
from typing import Dict, Optional
import matplotlib.pyplot as plt
from omegaconf import OmegaConf, MISSING


# === NOTES
"""
Some losses configs are key based '[]', not dot based '.'
"""


# Optional perceptual loss libraries
try:
    from lpips import LPIPS
except ImportError:
    LPIPS = None

try:
    from torchvision.models import vgg16
except ImportError:
    vgg16 = None


# ----------------------------------------------
# Loss Registry
# ----------------------------------------------

LOSS_REGISTRY = {}

def register_loss(name):
    def decorator(fn):
        LOSS_REGISTRY[name] = fn
        return fn
    return decorator


SCHEDULE_REGISTRY = {}  # here?

def register_schedule(name):
    def decorator(fn):
        SCHEDULE_REGISTRY[name] = fn
        return fn
    return decorator


# -----------------------------------------------
# Basic Losses
# -----------------------------------------------

@register_loss("mse")
def mse_loss(pred, target, weight=None):
    loss = F.mse_loss(pred, target, reduction='none')
    if weight is not None:
        loss = loss * weight.view(-1, 1, 1, 1)
    return loss.mean()

@register_loss("huber")
def huber_loss(pred, target, weight=None, delta=1.0):
    loss = F.smooth_l1_loss(pred, target, beta=delta, reduction='none')
    if weight is not None:
        loss = loss * weight.view(-1, 1, 1, 1)
    return loss.mean()

@register_loss("p2")
def p2_loss(pred, target, weight=None):
    loss = (pred - target) ** 2
    if weight is not None:
        loss = loss * weight.view(-1, 1, 1, 1)
    return loss.mean()


# -----------------------------------------------
# Perceptual Losses
# -----------------------------------------------

if LPIPS is not None: 
    @register_loss("lpips")
    def lipis_loss(pred, target, **kwargs):
        loss_fn = LPIPS(net='vgg').to(pred.device)
        return loss_fn(pred, target).mean()

@register_loss("vgg")
def vgg_feature_loss(pred, target, **kwargs):
    from torchvision.models import VGG16_Weights
    assert vgg16 is not None, "torchvision VGG16 not available"
    vgg = vgg16(weights=VGG16_Weights.DEFAULT).features.eval().to(pred.device)
    for param in vgg.parameters():
        param.requires_grad = False

    def extract_features(x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        return vgg(x)
    
    f_pred = extract_features(pred)
    f_target = extract_features(target)
    return F.l1_loss(f_pred, f_target)


# ---------------------------------------------
# Dynamic Weight Schedules
# ---------------------------------------------

def linear_ramp_weight(start, end, max_steps, step):
    alpha = min(step / max_steps, 1.0)
    return start * (1 - alpha) + end * alpha

def cosine_scheudle_weight(start, end, max_steps, step):
    import math
    cos_val = (- math.cos(min(step / max_steps, 1.0) * math.pi)) / 2
    return start * (1 - cos_val) + end * cos_val


# ----------------------------------------------
# Combined Loss Module
# ----------------------------------------------

class DiffusionLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.losses = config.losses     # List of dicts with {type, weight, schedule}
        self.max_steps = config.training.num_epochs * config.training.steps_per_epoch

    def get_weight(self, loss_cfg, step):
        sched = loss_cfg.get("schedule", "constant") # ?
        start = loss_cfg.get("start_weight", loss_cfg["weight"])
        end = loss_cfg.get("end_weight", loss_cfg["weight"])

        if sched == "linear":
            return linear_ramp_weight(start, end, self.max_steps, step)
        elif sched == "cosine":
            return cosine_scheudle_weight(start, end, self.max_steps, step)
        else: 
            return loss_cfg["weight"]
        

    def forward(self, pred, target, step=0, visualize=False) -> Dict[str, torch.Tensor]:
        total_loss = 0.0
        loss_dict = {}

        for loss_cfg in self.losses:
            kind = loss_cfg["type"]
            weight = self.get_weight(loss_cfg, step)

            loss_fn = LOSS_REGISTRY[kind]
            loss_val = loss_fn(pred, target)

            weighted_loss = weight * loss_val
            total_loss += weighted_loss

            loss_dict[f"{kind}_loss"] = loss_val.detach()
            loss_dict[f"{kind}_weight"] = weight
            loss_dict[f"{kind}_weighted"] = weighted_loss.detach()

        loss_dict["total_loss"] = total_loss
        if visualize:
            loss_dict["output_sample"] = pred[:4].detach()
            loss_dict["target_sample"] = target[:4].detach()

        return loss_dict
    

# --------------------------------------
# Dynamic Weight Schedules
# --------------------------------------

@register_schedule("constant")
def constant_schedule(t, alpha_t=None):
    return torch.ones_like(t).float()

@register_schedule("inverse_alpha_squared")
def inverse_alpha_squared(t, alpha_t):
    return 1.0 / (alpha_t ** 2 + 1e-7)

@register_schedule("linear") 
def linear_decay(t, alpha_t=None, T=1000):
    return 1.0 - t.float() / T

@register_schedule("cosine_decay")
def cosine_decay(t, T=1000): # might need to add alpha_t
    return 0.5 * (1 + torch.cos(torch.pi * t.float() / T))


# ------------------------------------
# CONFIG-COMPATIBLE LOSS HANDLER
# ------------------------------------

def get_loss_fn(cfg):   
    loss_type = cfg.losses.type.lower()
    
    schedule_cfg = OmegaConf.select(cfg, "losses.schedule")     # safe path acess
    if schedule_cfg is None or schedule_cfg is MISSING:
        # no schedule selection -> skip
        schedule_type = None
    else:
        schedule_type = cfg.losses.schedule.type.lower()

    loss_fn = LOSS_REGISTRY[loss_type]
    if schedule_type: weight_fn = SCHEDULE_REGISTRY[schedule_type] 

    def wrapped(pred, target, t, alpha_t=None):
        if schedule_type: weight = weight_fn(t, alpha_t) 
        else: weight = None
        
        loss = loss_fn(pred, target, weight=weight) 
        return loss, weight
    
    return wrapped


# -------------------------------------
# DEBUG VISUALIZER
# -------------------------------------

def visualize_loss(loss_tensor, title="Loss Heatmap"): 
    if loss_tensor.dim() == 4:
        heatmap = loss_tensor.mean(dim=1).detach().cpu().numpy()
        for i in range(min(4, heatmap.shape[0])):       # Show up to 4 samples
            plt.imshow(heatmap[i], cmap='viridis')
            plt.title(f"{title} [ sample {i}]")
            plt.colourbar()
            plt.show














