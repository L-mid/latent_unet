
import torch
import asyncio
import logging
from typing import Optional, Dict, Any

from . import tensorstore_core
from . import schema_utils 
from . import chunk_tuner
from . import metadata_utils
from . import registry

logger = logging.getLogger(__name__)

# === NOTES
"""
Serialization of dicts is a bit odd.

Learned the hard way that tensorstore is acutally (nothing obivously wrong), just really, really slow on CPU.

"""


# ------------------------------------------------------------------------------------
# High-level save
# ------------------------------------------------------------------------------------

logger.setLevel(logging.INFO)     # keep logger.propagate = True (default)

def save_checkpoint(
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        scheduler: Optional[Any],
        epoch: int,
        step: int,
        path: str,                      # root path
        driver: str = "local",
        schema: Optional[Dict] = None,
        metadata: Optional[Dict] = None,
        storage_options: Optional[Dict] = None,     
        chunk_strategy: str = "auto"
):
    # full checkpoint save interface

    async def async_save():
        kvstore = registry.get_kvstore(path, driver=driver, storage_options=storage_options)

        state_dict = model.state_dict()
        shapes = {k: tuple(v.shape) for k, v in state_dict.items()}
        chunks = chunk_tuner.get_chunk_config(shapes, strategy=chunk_strategy)

        # Model weight
        for name, tensor in state_dict.items():
            
            chunk = decide_chunk(name, tensor, chunks)

            await tensorstore_core.write_tensor(kvstore, f"model/{name}", tensor, chunks=chunk)

        # Optimizer state
        if optimizer is not None:
            opt_state = optimizer.state_dict()
            opt_tensor = serialize_dict(opt_state)

            await tensorstore_core.write_tensor(kvstore, "optimizer/state", opt_tensor)

        if scheduler is not None:
            sched_state = scheduler.state_dict()
            sched_tensor = serialize_dict(sched_state)

            await tensorstore_core.write_tensor(kvstore, "scheduler/state", sched_tensor)

        # consider saving ema stuff



        # Store metadata directly into store root (attributes)
        meta = {
            "epoch": epoch,
            "step": step,
            **(metadata or {})
        } 
        metadata_utils.attach_metadata(path, meta) # attach metadata

        logger.info("[TS-WRAPPER] Checkpoint save complete")
        
        # Schema validation post-write
        if schema:
            await schema_utils.validate_schema_async(kvstore, schema) # schema validation

    return asyncio.run(async_save())   # we unlearned it.
        
    


# --------------------------------------------------------------------------------------------
# High-level load
# --------------------------------------------------------------------------------------------

def load_checkpoint(
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        scheduler: Optional[Any],
        path: str,                      # root path
        schema: Optional[Dict] = None,
        storage_options: Optional[Dict] = None,
        strict: bool = True
) -> Dict:
    
    # Full checkpoint load interface.

    state_dict = model.state_dict()
    shapes = {k: tuple(v.shape) for k, v in state_dict.items()}
    
    async def async_load():
        kvstore = registry.get_kvstore(path, storage_options=storage_options)

        # Schema validation before reading
        if schema:
            await schema_utils.validate_schema_async(kvstore, schema)

        loaded = {}

        for name in state_dict.keys():
            loaded[name] = await tensorstore_core.read_tensor(kvstore, f"model/{name}")
        model.load_state_dict(loaded, strict=strict)
        logger.info("[TS-WRAPPER] Model weights loaded")

        model.load_state_dict(loaded, strict=strict)
        logger.info("[TS-WRAPPER] Model weights loaded")


        # Optimizer
        if optimizer is not None:
            opt_tensor = await tensorstore_core.read_tensor(kvstore, "optimizer/state")
            optimizer.load_state_dict(deserialize_dict(opt_tensor))
            logger.info("[TS-WRAPPER] Optimizer state loaded")

        # Scheduler
        if scheduler is not None:
            sched_tensor = await tensorstore_core.read_tensor(kvstore, "scheduler/state")
            scheduler.load_state_dict(deserialize_dict(sched_tensor))
            logger.info("[TS-WRAPPER] Scheduler state loaded")

        meta = metadata_utils.read_metadata(path)    
        logger.info(f"[TS-WRAPPER] Metadata loaded: {meta}")
        return meta
    
    return asyncio.run(async_load()) 


# -------------------------------------------------------------------------------------
# (De)serialization for optimizer/schedueler
# -------------------------------------------------------------------------------------

def serialize_dict(obj: dict) -> torch.Tensor:
    import pickle
    buf = pickle.dumps(obj)
    array = torch.tensor(list(buf), dtype=torch.uint8)
    return array


def deserialize_dict(tensor: torch.Tensor) -> dict:
    import pickle
    buf = bytes(tensor.tolist())
    obj = pickle.loads(buf)
    return obj



# -----------------------------------------------------------------------------
# Utilites 
# -----------------------------------------------------------------------------

def _fallback_chunks(shape, cap=256):
    # Reasonable default: clamp each dim to <= cap, but never 0
    return tuple(max(1, max(cap, int(s))) for s in shape)


def decide_chunk(name, tensor, chunks):     # could be this
    shape = tuple(int(s) for s in tensor.shape)
    chunk = chunks.get(name)
    if chunk is None:
        # Option A: derive by rule (e.g., full row for 2D weights, full len for bias)
        if len(shape) == 1:
            chunk = (shape[0],)
        elif len(shape) == 2:
            # full column chunks are nice for GEMM-shaped weights; tweak as you like
            chunk = (min(1024, shape[0]), shape[1])
        else:
            # generic fallback
            chunk = _fallback_chunks(shape)

    return chunk


# For debugging ONLY (may remove later):

async def write_param(kvstore, name, tensor, chunk=None):
    import asyncio, time, logging
    log = logging.getLogger("ckpt")     # temporary to use this fn


    t0 = time.perf_counter()
    log.info("[TS] write start: %s", name)
    await asyncio.wait_for(
        tensorstore_core.write_tensor(kvstore, f"model/{name}", tensor, chunks=chunk),
        timeout=15,  # ← per-write cap so we learn *which* name stalls
    )
    log.info("[TS] write done: %s in %.3fs", name, time.perf_counter() - t0)


async def async_save_all(kvstore, params):
    # e.g., sequential while you debug; later you can batch with gather
    for name, tensor in params:
        await write_param(kvstore, name, tensor)

