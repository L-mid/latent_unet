
import os
import logging
from . import zarr_core
from typing import Optional, Any, Union
import torch

# === NOTES
"""
logging is not the logger i created.

dict reading management (serialize and deserialize) is not good.

"""


logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------------
# High-level Save
# -----------------------------------------------------------------------------------

def save_model(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    ema: Union[torch.optim.swa_utils.AveragedModel, Any, None],            # don't have a good name. But put the model in
    epoch: int,
    step: int,
    path: str,
    extra: dict = None
):
    # High-level save of full model state into Zarr.

    logger.info(f"[ZARR] Saving full checkpoint to: {path}")

    group = zarr_core.open_store(path, mode="w")

    # Save model parameters
    for name, param in model.state_dict().items():
        zarr_core.write_tensor(group, f"model/{name}", param)

    # Save optimizer state
    if optimizer is not None:
        opt_sate = optimizer.state_dict()
        for key, value in opt_sate.items():
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


    group = zarr_core.open_store(path, mode="a")
    
    # Save metadata
    group.attrs["epoch"]            = int(epoch) 
    group.attrs["step"]             = int(step)
    group.attrs["epoch_completed"]  = int(epoch)
    group.attrs["epoch_next"]       = int(epoch + 1)    
    group.attrs["global_step"]      = int(step)
    group.attrs["extra"] = extra or {}  



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

    logger.info(f"[ZARR] Loading checkpoint from: {path}")

    group = zarr_core.open_store(path, mode="r")

    # Load model weights
    state_dict = {}
    model_group = group.require_group("model")
    for name in model_group.array_keys():
        tensor = zarr_core.read_tensor(model_group, name)
        state_dict[name] = tensor

    model.load_state_dict(state_dict, strict=strict)        # what does strict do?
    logger.info(f"[ZARR] Model weights loaded (strict={strict})")

    # Load optimizer state
    if optimizer is not None and "optimizer" in group.array_keys():
        opt_state = {}
        opt_group = group["optimizer"]

        for key in opt_group.array_keys(): 
            data = opt_group[key][...]
            if data.shape == ():
                data = data.item()
            opt_state[key] = data

        # Read attrs (if you saved scalars/metadata as attributes)
        for k, v in opt_group.attrs.items():
            opt_group[k] = v

        optimizer.load_state_dict(opt_state)
        logger.info("[ZARR] Optimizer state loaded")


    # Load scheduler state
    if scheduler is not None and "scheduler" in group.array_keys(): 
        sched_state = {}
        sched_group = group["scheduler"]

        for key in sched_group.array_keys():                # iterate stored arrays
            data = sched_group[key][...]                    # -> np.ndarray
            if data.shape == ():
                data = data.item()                          # 0-dim scalar array
            sched_group[key] = data                        

        # Read attrs (if you saved scalars/metadata as attributes)
        for k, v in sched_group.attrs.items():
            sched_group[k] = v

        scheduler.load_state_dict(sched_state)
        logger.info("[ZARR] Scheduler state loaded")


    # Load ema state
    if ema is not None and "ema" in group.array_keys():
        ema_state = {}
        ema_group = group["ema"]

        for key in ema_group.array_keys():
            data = sched_group[key][...]
            if data.shape == ():
                data = data.item()
            ema_group[key] = data

        # Read attrs (if you saved scalars/metadata as attrs)
        for k, v in ema_group.attrs.items():
            ema_group[k] = v
            
        ema.load_state_dict(ema_state)
        logger.info("[ZARR] Schedule state loaded")



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