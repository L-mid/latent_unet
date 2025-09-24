
import torch
import logging
import tensorstore as ts
from typing import Dict, Tuple, Optional
import numpy as np

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------------
# Schema format:
# { group: { tensor_name: (shape_tuple, dtype_str) } }
# ------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------
# Generate schema from model state_dict
# ------------------------------------------------------------------------------------

def generate_schema(model: "torch.nn.Module") -> Dict:
    """
    Auto-generate schema from model state_dict.
    """

    schema = {}
    for name, tensor in model.state_dict().items():
        shape = tuple(tensor.shape)
        dtype = str(tensor.detach().cpu().numpy().dtype)
        schema[name] = (shape, dtype)
    return {"model": schema}

# -------------------------------------------------------------------------------------
# Full Async schema validator
# -------------------------------------------------------------------------------------

async def validate_schema_async(kvstore, schema: Dict):
    # Validates live TensorStore against schema definition.

    logger.info("[TS-SCHEMA] Starting schema validation")

    for top_group, keys in schema.items():
        for key, (expected_shape, expected_dtype) in keys.items():
            dataset_path = f"{top_group}/{key}"

            spec = {
                "driver": "zarr3",
                "kvstore": kvstore,
                "path": dataset_path,
                "open": True, 
                "schema": {
                    "dtype": expected_dtype,
                    "domain": {"shape": list(expected_shape)},
                },
            }

            try:
                tstore = await ts.open(spec)
            except Exception as e:
                raise ValueError(f"[TS-SCHEMA] Domain mismatch at '{dataset_path}': {e}")
            
            actual_shape = tuple(tstore.shape)
            actual_dtype = tstore.dtype.name

            compare_shape_dtype(
                dataset_path,
                expected_shape,
                actual_shape,
                expected_dtype, 
                actual_dtype,

            )


# ---------------------------------------------------------------------------------
# Shape/dtype checker
# ---------------------------------------------------------------------------------

def compare_shape_dtype(
        name: str, 
        expected_shape: Tuple,
        actual_shape: Tuple,
        expected_dtype: str,
        actual_dtype: str
):
    if expected_shape != actual_shape:
        raise ValueError(
            f"[TS-SCHEMA] Shape mismatch for '{name}': expected {expected_shape}, found {actual_shape}"
        )
    if expected_dtype != actual_dtype:
        raise ValueError(
            f"[TS_SCHEMA] Dtype mismatch for '{name}', expected {expected_dtype}, found {actual_dtype}"
        )
    

# -----------------------------------------------------------------------------------
# Pretty print schema (optional debug)
# -----------------------------------------------------------------------------------

def log_schema(schema: Dict):
    logger.info("[TS-SCHEMA] Schema summary:")
    for group, keys in schema.items():
        for name, (shape, dtype) in keys.items():
            logger.info(f"  {group}/{name} | shape={shape} | dtype={dtype}")
                

# ------------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------------

def dtype_to_spec_str(dt): # so far this activly gives out stuff i dont want (dtype("float32")) may delete
    """Normalize dtype to the string Tensorstore expects"""
    if torch is not None and isinstance(dt, torch.dtype):
        # Map common torch dtypes to numpy names
        torch_map = {
            torch.float32: "float32",
            torch.float64: "float64",
            torch.float16: "float16",
            torch.bfloat16: "bfloat16",
            torch.int64: "int64",
            torch.int32: "int32",
            torch.int16: "int16",
            torch.int8: "int8",
            torch.uint8: "uint8",
            torch.bool: "bool",
        }
        return torch_map.get(dt, str(np.dtype(str(dt))))
    if isinstance(dt, np.dtype):
        return dt.name
    if isinstance(dt, str):
        return np.dtype(dt).name    # normalize aliases like "single" -> "float32"
    # Fallback
    return np.dtype(dt).name

def _dtype_name(dt):
    """For comparasions & messages: always a clean string like 'float32'"""
    if isinstance(dt, np.dtype):
        return dt.name
    return np.dtype(dt).name

