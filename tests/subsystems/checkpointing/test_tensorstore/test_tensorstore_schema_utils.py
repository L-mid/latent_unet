
import os 
import tempfile
import pytest 
import torch
import asyncio
import numpy as np
import tensorstore as ts
import re


"""
try:
    import zarr
except ImportError:
    zarr = None

if zarr == None: pytest.skip(reason="Tensorstore uses zarr backend")
"""

from utils.tensorstore_checkpointing import (
    tensorstore_core,
    schema_utils,
    registry,
)

# --------------------------------------------------------------------------------
# Create vaild TensorStore datasets for controlled schema tests
# --------------------------------------------------------------------------------

def linear_schema(linear: torch.nn.Linear, dtype="float32"):
    """
    Build schema entries for a given nn.Linear layer.
    Returns a dict mapping param names -> (shape, dtype)
    """
    out_features, in_features = linear.out_features, linear.in_features
    dt = np.dtype(dtype).name   # normalize dtype string
    schema = {
        "weight":   ((out_features, in_features), dt),
        "bias":     ((out_features,), dt) if linear.bias is not None else None,
    }

    print(schema)

    return schema


async def create_valid_store(path): 
    kvstore = registry.get_kvstore(path, driver="local")        # {"driver": "file", "path": C:\\...tmp"}

    # Weight: (out_features, in_features) = (10, 5)
    await tensorstore_core.write_tensor( 
        kvstore=kvstore,
        array_path="model/linear.weight",
        tensor=torch.randn(10, 5, dtype=torch.float32)
    )

    # Bias matches out_features = 10
    await tensorstore_core.write_tensor(
        kvstore=kvstore,
        array_path="model/linear.bias",
        tensor=torch.randn(10, dtype=torch.float32),
    )

    return kvstore


# -----------------------------------------------------------------------------
# Reference vaild schema definition
# -----------------------------------------------------------------------------

vaild_schema = {
    "model": {
        "linear.weight": ((10, 5), "float32"),
        "linear.bias": ((10,), "float32")
    }
}

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

async def delete_array_prefix(kvstore, array_path: str):
    store = kvstore if isinstance(kvstore, ts.KvStore) else await ts.KvStore.open(kvstore)

    prefix = ("/" + array_path.strip("/")).encode() + b"/"      # e.g. b"/model/linear.bias/"
    lo = prefix
    hi = prefix + b"\xff"
    await store.delete_range(ts.KvStore.KeyRange(lo, hi))    # removes metadata + chunks


# -------------------------------------------------------------------------------
# Test: schema passes on vaild store
# -------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_vaild_schema_passes():
    with tempfile.TemporaryDirectory() as tmpdir:
        kvstore = await create_valid_store(tmpdir) 
        await schema_utils.validate_schema_async(kvstore, vaild_schema) 



# -------------------------------------------------------------------------------
# Test: missing dataset triggers failure
# -------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_missing_dataset_detected():
    with tempfile.TemporaryDirectory() as tmpdir:
        kvstore = await create_valid_store(tmpdir)

        keys_before = await (await ts.KvStore.open(kvstore)).list()
        print(keys_before, "[TEST TS_UTILS], keys before deletion")

        # Delete one dataset manually
        await delete_array_prefix(kvstore, "model/linear.bias")

        keys_after = await (await ts.KvStore.open(kvstore)).list()
        print(keys_after, "[TEST TS_UTILS], keys after deletion")

        with pytest.raises(ValueError, match="Domain mismatch at"):
            await schema_utils.validate_schema_async(kvstore, vaild_schema)


# -------------------------------------------------------------------------------------------
# Test: shape mismatch triggers failure
# -------------------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_shape_mismatch_detected():
    with tempfile.TemporaryDirectory() as tmpdir:
        kvstore = await create_valid_store(tmpdir)

        # Recreate dataset with wrong shape
        spec = {
            "driver": "zarr3",
            "kvstore": kvstore,                       
            "path": "model/linear.weight",
            "delete_existing": True,
            "create": True,
            "schema": {
                "dtype": "float32",
                "domain": {"shape": [8, 5]},
            },
        }
        
        tstore = await ts.open(spec)
        await tstore.write(np.random.randn(8, 5).astype(np.float32))

        with pytest.raises(ValueError): 
            await schema_utils.validate_schema_async(kvstore, vaild_schema)


# -----------------------------------------------------------------------------------
# Test: dtype mistmatch triggers failure
# -----------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_dtype_mismathc_detected():
    with tempfile.TemporaryDirectory() as tmpdir:
        kvstore = await create_valid_store(tmpdir)

        # Recreate dataset with wrong shape
        spec = {
            "driver": "zarr3",
            "kvstore": kvstore,                       
            "path": "model/linear.weight",
            "delete_existing": True,
            "create": True,
            "schema": {
                "dtype": "float64",
                "domain": {"shape": [10,]}
            },
        }
        tstore = await ts.open(spec)
        await tstore.write(np.random.randn(10).astype(np.float64))

        with pytest.raises(ValueError):
            await schema_utils.validate_schema_async(kvstore, vaild_schema)




