
import os
import logging
from . import zarr_core
from typing import Optional, Any, Union
import torch
from pathlib import Path
from utils.checkpointing import ckpt_io
import zarr

# === NOTES
"""
logging is not the logger i created.

"""


logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------------
# High-level Save
# -----------------------------------------------------------------------------------

def save_model(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    ema: Union[Any, None], 
    *,         
    epoch: int,
    step: int,
    path: str,
    keep_last: int = 3,
    extra: dict = None
):
    path = Path(path)
    #path = in_path / f"epoch_{epoch:06d}_step_{step:09d}"
    #logger.info(f"[ZARR] Saving full checkpoint at: {path}")
    
    # High-level save of full model state into Zarr.

    tmp_dir, final_dir = ckpt_io.begin_versioned_save(path, epoch, step)
    
    # open a Zarr group into the TMP dir
      
    group = zarr_core.open_store(tmp_dir, mode="w")


    # Save model parameters
    for name, param in model.state_dict().items():
        zarr_core.write_tensor(group, f"model/{name}", param)

    # Save optimizer state
    if optimizer is not None:
        opt_state = optimizer.state_dict()
        for key, value in opt_state.items():
            sub_group = group.require_group("optimizer")
            zarr_core.write_tensor(sub_group, key, serialize_dict(value)) 

    # Save scheduler state
    if scheduler is not None:
        sched_state = scheduler.state_dict()
        for key, value in sched_state.items():
            sub_group = group.require_group("scheduler")
            zarr_core.write_tensor(sub_group, key, serialize_dict(value)) 

    # Save ema state
    if ema is not None:
        ema_state = ema.state_dict()
        for key, value in ema_state.items():
            sub_group = group.require_group("ema") 
            zarr_core.write_tensor(sub_group, key, serialize_dict(value))
            
    

    group = zarr_core.open_store(tmp_dir, mode="a")
    
    # Save metadata
    group.attrs["epoch"]            = int(epoch) 
    group.attrs["step"]             = int(step)
    group.attrs["epoch_completed"]  = int(epoch)
    group.attrs["epoch_next"]       = int(epoch + 1)    
    group.attrs["global_step"]      = int(step)
    group.attrs["extra"] = extra or {}  

    # mark complete *inside tmp*
    assert tmp_dir.exists(), f"tmp_dir vanished before complete: {tmp_dir}"
    ckpt_io.mark_complete(tmp_dir) 

    # atomic promote tmp -> final
    ckpt_io.finalize_versioned_save(tmp_dir, final_dir) 

    # pointer file at root
    ckpt_io.update_latest_pointer(path, final_dir, epoch, step)  

    # retention
    ckpt_io.keep_last_n(path, n=keep_last)   

    return final_dir


# ------------------------------------------------------------------------------
# High-level Load
# ------------------------------------------------------------------------------

def load_model(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    ema: Union[Any, None],
    path: str,
    strict: bool = True,
) -> dict:
    

    # High-level load of full model state from Zarr.

    #logger.info(f"[ZARR] Loading checkpoint at: {path}")    # path is determined by ckpt finder before load

    group = zarr_core.open_store(path, mode="r")        


    # Load model weights
    state_dict = {}
    model_group = group.require_group("model")
    for name in model_group.array_keys():
        tensor = zarr_core.read_tensor(model_group, name)
        state_dict[name] = tensor

    model.load_state_dict(state_dict, strict=strict)       
    #logger.info(f"[ZARR] Model weights loaded (strict={strict})")


    # --- Optimizer ---
    if optimizer is not None and _has_child_group(group, "optimizer"):
        opt_group = group["optimizer"]
        opt_state = _load_serialized_map(opt_group)
        optimizer.load_state_dict(opt_state)
        #logger.info("[ZARR] Optimizer state loaded")


    # --- Scheduler ---
    if scheduler is not None and _has_child_group(group, "scheduler"):
        sched_group = group["scheduler"]
        sched_state = _load_serialized_map(sched_group)
        scheduler.load_state_dict(sched_state)
        #logger.info("[ZARR] Scheduler state loaded")


    # --- EMA ---
    if ema is not None and _has_child_group(group, "ema"):
        ema_group = group["ema"]
        ema_state = _load_serialized_map(ema_group)
        ema.load_state_dict(ema_state)
        #logger.info("[ZARR] EMA state loaded")


    # Read metadata

    # dict(root.attrs) gives you everything:
    state = dict(group.attrs)
    # (optional) cast numpy scalars to Python ints
    for k, v in list(state.items()):
        try:
            import numpy as np
            if isinstance(v, np.generic):
                state[k] = np.asarray(v).item()
        except Exception:
            pass
    return state
            

# ------------------------------------------------------------------------------------
# Utility: Serialize optimizer/scheduler states
# ------------------------------------------------------------------------------------

def serialize_dict(obj: dict) -> torch.Tensor:
    # Safely serialize a state dict to tensor for Zarr. (make this smarter later)

    import pickle
    buf = pickle.dumps(obj)
    array = torch.tensor(list(buf), dtype=torch.uint8)
    return array

def deserialize_dict(tensor: torch.Tensor) -> dict:
    # Deserialize a state dict from tensor.

    import pickle
    buf = bytes(tensor.tolist())
    obj = pickle.loads(buf)
    return obj


# Serialization helpers
def _has_child_group(group, name: str) -> bool:
    # Works for v2/v3-ish group APIs: membership checks both arrays & groups.
    try:
        return name in group and hasattr(group[name], "attrs")
    except Exception:
        return False

def _load_serialized_map(sub_group):
    out = {}
    for key in sub_group.array_keys():               # keys like "state", "param_groups", etc.
        t = zarr_core.read_tensor(sub_group, key)    # -> torch.uint8 tensor
        out[key] = deserialize_dict(t)               # -> Python object (dict/list/number)
    return out