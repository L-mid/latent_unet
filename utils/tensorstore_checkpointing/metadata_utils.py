
import datetime
import hashlib
import subprocess
import logging
from typing import Dict, Any, Optional, Mapping, Iterable
import zarr
import fsspec
import os
import math


logger = logging.getLogger(__name__)

# === Notes:
"""
Not currently used by Tensorstore but interesting.
"""

# --------------------------------------------------------------------------------
# Generate full metadata entry
# --------------------------------------------------------------------------------

def generate_metadata(
        config_dict: Dict,
        run_id: str,
        code_version: Optional[str] = None
) -> Dict[str, Any]:
    # Generates full metadata dict to attach to checkpoint.

    metadata = {}

    metadata["timestamp_utc"] = datetime.datetime.now()         # unsure if this is correct
    metadata["run_id"] = run_id
    metadata["git_commit"] = get_git_commit_hash()
    metadata["config_hash"] = hash_config(config_dict)
    metadata["code_version"] = code_version or "unknown"

    return metadata


# ----------------------------------------------------------------------------------
# Write metadata into Zarr store
# ----------------------------------------------------------------------------------

def attach_metadata(root_path: str, meta: Dict[str, Any]) -> None:
    root = _open_root_group(root_path)
    safe = _prefix_keys(meta, "meta_")
    # You can set in bulk if you prefer:
    for k, v in safe.items():
        root.attrs[k] = v

# -----------------------------------------------------------------------------------
# Read metadata back from Zarr store.
# -----------------------------------------------------------------------------------

def read_metadata(root_path: str) -> Dict[str, Any]:
    # Reads metadata atrributes from Zarr store
    root = zarr.open_group(store=root_path, mode="r")
    return {k[len("meta_"):]: v for k, v in root.attrs.items() if k.startswith("meta_")}



# -----------------------------------------------------------------------------------
# Pretty-print metadata
# -----------------------------------------------------------------------------------

def summarize_metadata(metadata: Dict[str, Any]):
    # Logs formmated metadata summary.

    logger.info("[METADATA] Checkpoint metadata:")
    for key, value in metadata.items():
        logger.info(f"  {key}: {value}")


# -------------------------------------------------------------------------------------
# Git has capture helper
# -------------------------------------------------------------------------------------

def get_git_commit_hash() -> str:
    # Return current Git commit has (if inside a git repo)

    try:
        commit = subprocess.check_call(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
        return commit
    except Exception:
        return "unknown"


# --------------------------------------------------------------------------------------
# Config hash helper
# --------------------------------------------------------------------------------------

def hash_config(config_dict: Dict) -> str:
    # Compute has a full config dictionary (considtent across runs).

    import yaml
    canonical = yaml.dump(config_dict, sort_keys=True)
    hash_digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return hash_digest    
    
def _open_root_group(path: str):
    path = os.path.normpath(path)
    # Zarr v3
    return zarr.open_group(store=path)

def _json_safe(x: Any) -> Any:
    """Recursively convert to Zarr-v3/JSON-safe types."""
    # primitives
    if x is None or isinstance(x, (bool, int, str)):
        return x
    if isinstance(x, float):
        if math.isfinite(x):
            return x
        # choose a policy: convert to None or string
        return None
    
    # datetime/date
    if isinstance(x, (datetime.datetime, datetime.date)):
        # always serialize as ISO-8601
        return x.isoformat()
    

    # bytes/bytearray
    if isinstance(x (bytes, bytearray)):
        try:
            return x.decode("ctf-8")
        except Exception:
            # last resort: JSON-safe base64
            import base64
            return {"__bytes_b64__": base64.b64encode(bytes(x)).decode("ascii")}
        

    # numpy / torch
    try:
        import numpy as np
        if isinstance(x, np.generic):
            return x.item()
        if isinstance(x, np.ndarray):
            return x.tolist()
    except Exception:
        pass
    try:
        import torch
        if isinstance(x, torch.Tensor):
            if x.numel() == 1:
                return x.detach().cpu().item()
            return x.detach().cpu().tolist()
    except Exception:
        pass


    # mappings / iterables
    if isinstance(x, Mapping):
        return {str(k): _json_safe(v) for k, v in x.items()}
    if isinstance(x, Iterable) and not isinstance(x, (str, bytes, bytearray)):
        return [_json_safe(v) for v in x]
    

    # enums or other objects: stringify
    try:
        import enum
        if isinstance(x, enum.Enum):
            return x.value  # or x.name
    except Exception:
        pass

    return str(x)


def _prefix_keys(d: Dict[str, Any], prefix: str = "meta_") -> Dict[str, Any]:
    return {f"{prefix}{k}": _json_safe(v) for k, v in d.items()}




