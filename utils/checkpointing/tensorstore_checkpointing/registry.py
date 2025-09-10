
import tensorstore as ts
import logging
from typing import Callable, Dict, Optional

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------------
# Internal driver registry
# --------------------------------------------------------------------------------

_DRIVERS: Dict[str, Callable[[str, Optional[Dict]], Dict]] = {}

# --------------------------------------------------------------------------------
# Public registration API
# --------------------------------------------------------------------------------

def register_driver(name: str, builder: Callable[[str, Optional[Dict]], Dict]):
    # Register a new storage backend

    if name in _DRIVERS:
        logger.warning(f"[REGISTRY] Overwriting existing driver '{name}'")
    _DRIVERS[name] = builder
    logger.info(f"[REGISTRY] Registered TensorStore driver: '{name}'")


def get_kvstore(
    path: str,
    driver: str = "local",
    storage_options: Optional[Dict] = None
) -> Dict:
    
    # Build kvstore spec TensorStore based on registered backend.

    if driver not in _DRIVERS:
        raise ValueError(f"[REGISTRY] Unknown driver '{driver}'")
    
    return _DRIVERS[driver](path, storage_options)


def list_drivers():
    logger.info("[REGISTRY] Available drivers:")
    for name in _DRIVERS.keys():
        logger.info(f"  - {name}")


# -----------------------------------------------------------------------------------
# Built-in driver implementations
# -----------------------------------------------------------------------------------

# Local filesystem
def _local_driver(path: str, storage_options: Optional[Dict]) -> Dict:
    return {"driver": "file", "path": path}

# GDS driver
def _gcs_driver(path: str, storage_options: Optional[Dict]) -> Dict:
    cfg = storage_options or {} 
    return {"driver": "s3", "path": path, **cfg}

# S3 driver
def _s3_driver(path: str, storage_options: Optional[Dict]) -> Dict:
    cfg = storage_options or {}
    return {"driver": "s3", "path": path, **cfg}

# Azure Blob driver
def _azure_driver(path: str, storage_options: Optional[Dict]) -> Dict:
    cfg = storage_options or {}
    return {"driver": "azure", "path": path, **cfg}


# -----------------------------------------------------------------------------------
# Register default drivers on import
# -----------------------------------------------------------------------------------

register_driver("local", _local_driver)
register_driver("gcs", _gcs_driver)
register_driver("s3", _s3_driver)
register_driver("azure", _azure_driver)


















