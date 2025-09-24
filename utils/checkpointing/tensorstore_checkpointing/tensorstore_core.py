
import os
import shutil
import asyncio
import logging
from typing import Optional, Tuple, Dict, Union

import numpy as np
import torch
import tensorstore as ts


# === NOTES
"""
Some of the helper functions and API calls are weird, also diff tensors stored separatly?

Deletion does not apply to GCS/S3 deletion routines.

"""


logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------------------

def _as_numpy(x: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)

def _array_chunks(shape: Tuple[int, ...], target: int = 128) -> list:
    """Choose reasonable default chunks (<= target along each axis)."""
    return [int(min(target, max(1, s))) for s in shape]

def _dtype_name(x: np.ndarray) -> str:
    """Standardize dtype string from metadata."""
    return np.dtype(x.dtype).name

def _normalize_kvstore(kvstore: Union[str, Dict]) -> Dict:
    if isinstance(kvstore, str):
        os.makedirs(kvstore, exist_ok=True)
        return {"driver": "file", "path": kvstore}
    return kvstore  # assume already {"driver": "...", "path": "..."}

# -------------------------------------------------------------------------------
# Core async API
# -------------------------------------------------------------------------------

async def write_tensor(
    kvstore: Union[str, Dict],                          # base store (dir path or kvstore dict)
    array_path: str,                                    # subpath inside store, e.g. "model/linear.weight"
    tensor: Union[torch.Tensor, np.ndarray],
    chunks: Optional[Tuple[int, ...]] | None = None,    # legacy: _array_chunks(shape)  (made chunking optionally local to here). Not currently used.
    codecs: Optional[list] = None, 
    delete_existing: bool = True,
):
    arr = _as_numpy(tensor)
    shape = tuple(int(s) for s in arr.shape)
    dtype = arr.dtype.name
    chunk_shape = _array_chunks(shape)                
    codecs = codecs or [{"name": "bytes"}]      # legacy option (slow on CPU): [{"name": "zstd", "configuration": {"level": 5}}]

    # normalize kvstore argument
    if isinstance(kvstore, str):
        kv = {"driver": "file", "path": kvstore}
        # ensure directory exists for local file driver
        os.makedirs(kvstore, exist_ok=True)
    else:
        kv = kvstore    # assumes already a proper kvstore dict

    spec = {
        "driver": "zarr3",
        "kvstore": kv,                        # assumes kv is a full dict {driver: file, path: "local"}
        "path": array_path,
        "create": True,
        "delete_existing": delete_existing,
        "schema": {
            "dtype": dtype,
            "domain": {"shape": list(shape)},
            "chunk_layout": {
                "read_chunk": {"shape": list(chunk_shape)},
                "write_chunk": {"shape": list(chunk_shape)},
                }, 
            "codec": {
                "driver": "zarr3",
                "codecs": codecs,
            },
            "fill_value": 0.0,
        },
    }

    #logger.info(f"[TS] kvstore={kv} | shape={arr.shape} dtype={arr.dtype} chunks={chunks}")
    
    store = await ts.open(spec)
    await store.write(arr)

    logger.info(f"[TS] Write complete: {array_path}")


    return store


async def read_tensor(
        kvstore: Union[str, Dict],                   # base store (dir path or kvstore dict)
        array_path: Union[str, Dict],                # subpath inside store, e.g. "model/linear.weight"
        device: Union[str, torch.device] = "cpu",
        dtype: Optional[torch.dtype] = None,         # optional final cast
        expect_shape: Optional[Tuple[int, ...]] = None,     # optional sanity check
):
    """
    Read a Zarr v3 array written by 'write_tensor' and return a torch.Tensor.
    """
    kv = _normalize_kvstore(kvstore)

    spec = {
        "driver": "zarr3",
        "kvstore": kv,
        "path": array_path,
        "open": True,           # open existing; do not create
        # DO NOT set dtype/schema here, we want whatever was written
    }

    tstore = await ts.open(spec)

    # Read to NumPy
    np_arr = await tstore.read()
    np_arr = np.asarray(np_arr)     # materialize

    if expect_shape is not None and tuple(np_arr.shape) != tuple(expect_shape):
        raise ValueError(
            f"[TS-CORE] Shape mismatch: expected {expect_shape}, got {tuple(np_arr.shape)}"
        )

    # Convert to torch
    torch_arr = torch.from_numpy(np_arr)
    if dtype is not None:
        torch_arr = torch_arr.to(dtype)
    torch_arr = torch_arr.to(device)

    return torch_arr


async def delete_tensor(    # depreated
    root_dir: str,
    name: str,
) -> None:
    return NotImplementedError



# -------------------------------------------------------------------------------------------------
# Friendly sync wrappers (nice for tests & scripts)
# -------------------------------------------------------------------------------------------------

def write_tensor_sync(*args, **kwargs) -> None:
    asyncio.run(write_tensor(*args, **kwargs))

def read_tensor_sync(*args, **kwargs) -> torch.Tensor:
    return asyncio.run(read_tensor(*args, **kwargs))

def delete_tensor_sync(*args, **kwargs) -> None:
    asyncio.run(delete_tensor(*args), **kwargs)












