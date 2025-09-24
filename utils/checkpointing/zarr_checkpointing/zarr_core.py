
import zarr 
import numpy as np
import torch
import os
import logging

# === NOTES
"""
logging is not the logger i created, weird.

Other than that seems alright

"""


logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------------
# Store open/create
# --------------------------------------------------------------------------------

def open_store(path: str, mode: str = "a", storage_options: dict = None) -> zarr.Group:
    """
    Opens (or creates) a Zarr store.

    Args:
        path (str): Filesystem path or cloud URL
        mode (str): "a" (default) = create or open / "w" = overwrite
        storage_options (dict): Passed to fsspec (for remote acess)

    # Returns:
        zarr.Group: root Zarr group object
    """

    if mode == "w" and os.path.exists(path):
        #logger.warning(f"[ZARR] Overwriting existing store at: {path}")
        import shutil
        #shutil.rmtree(path)    # deletes prev path?

    root = zarr.open_group(store=str(path), mode=mode) #
    return root


# -----------------------------------------------------------------------------------
# Tensor write
# -----------------------------------------------------------------------------------

def write_tensor(group: zarr.Group, name: str, tensor: torch.Tensor, chunks: tuple=None):
    """
    Saves a tensor into Zarr group.

    Args:
        group (zarr.Group): Parent group to write into
        name (str): Dataset name
        tensor (torch.Tensor): Tensor to save (moved to CPU automatically)
        chunks (tuple): Optional manual chunk shape
    """

    array_data = np.atleast_1d(tensor.detach().cpu()) #weird it needs conversion but ok

    if name in group:
        logger.warning(f"[ZARR] Overwriting array: {name}")
        del group[name]

    arr = group.create(
        name=name,
        shape=array_data.shape,
        dtype=array_data.dtype,
        chunks=("auto" if chunks is None else chunks),
        overwrite=False,     # could it be that im overwriting?
    ) 
    arr[:] = array_data

    #logger.info(f"[ZARR] Wrote tensor '{name}' with shape {tensor.shape}")
    return arr 

# ------------------------------------------------------------------------------------
# Tensor read 
# ------------------------------------------------------------------------------------

def read_tensor(group: zarr.Group, name: str, device: torch.device = "cpu") -> torch.Tensor:
    # Loads tensor from Zarr group.

    """
    Args:
        group (zarr.Group): Parent group to read from
        name (str): Dataset name
        Device (torch.device): Target device to load into
    
    Returns:
        torch.Tensor: Loaded tensor
    """

    if name not in group:
        raise KeyError(f"[ZARR] Tensor '{name}' not found in store")
    
    np_data = group[name][:]
    tensor = torch.tensor(np_data, device=device)
    
    #logger.info(f"[ZARR] Read tensor '{name}' with shape {tensor.shape}")
    return tensor


# --------------------------------------------------------------------------------
# Tensor deletion
# --------------------------------------------------------------------------------


def delete_array(group: zarr.Group, name: str):
    # Deletes array from Zarr group.

    if name in group:
        del group[name]
        logger.info(f"[ZARR] Deleted array '{name}'")
    else: 
        logger.warning(f"[ZARR] Array '{name}' not found - cannot delete")


