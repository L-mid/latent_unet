
import torch
import gc
import weakref
import logging 
from typing import Optional
from contextlib import Optional, Dict, List, Callable
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# === NOTES:
"""
UNTESTED, unused. 


This has a lot of occurence, BUT:

'logging' (not a native repo logger) is used extensivly, randomly, and served as a placeholder.
It's also used here.
"""

def get_memory_summary(device: Optional[torch.device] = None, detailed: bool = False) -> Dict:
    # Returns memory usage summary for the specified device.
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summary = {}

    if device.type == "cuda":
        device_inx = device.index if device.index is not None else torch.cuda.current_device()
        summary = {
            "device": f"cuda:{device_inx}",
            "allowcated_MB": round(torch.cuda.memory_allocated(device_inx) / 1024**2, 2),
            "reserved_MB": round(torch.cuda.max_memory_reserved(device_inx) / 1024**2, 2),
            "max_allocated_MB": round(torch.cuda.max_memory_allocated(device_inx) / 1024**2, 2),
            "max_allocated_MB": round(torch.cuda.max_memory_reserved(device_inx) / 1024**2, 2),
        }
        if detailed:
            summary["cuda_summary"] = torch.cuda.memory_summary(device_inx)     # bad indent?
    else:
        summary = {"device": str(device), "note": "No GPU memory to track on CPU."}

    return summary


def log_memory_usage(tag: Optional[str] = "", device: Optional[torch.device] = None, level=logging.INFO):
    # Logs current memory usage.
    summary = get_memory_summary(device)
    message = f"[MEMORY][{tag}] {summary}"
    logger.log(level, message)


def auto_clear_cache():
    # Clears unused GPU memory and forces garbage collection.
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


# -- Optional tensor allocation tracking (experimental, why not) --

_allocated_tensors = weakref.WeakSet()

def track_tensor_allocation(tensor: torch.Tensor):
    # Adds tensor to tracking set. Useful for debugging persistent memory leaks.
    _allocated_tensors.add(tensor)

def get_tracked_tensor_count() -> int:
    return len(_allocated_tensors)

def print_tracked_tensor_shapes():
    print("Tracked tensor shapes:")
    for t in _allocated_tensors:
        print(t.shape, t.device, t.dtype)


# -- Memory profiling context --

@contextmanager
def enable_memory_profiling(description: str = "Memory profile", device: Optional[torch.device] = None):
    # Context manager for profiling memory usage.
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.reset_peak_memory_stats(device)
    try:
        yield
    finally:
        used = torch.cuda.max_memory_allocated(device) / 1024**2
        logger.info(f"[PROFILE] {description} - Peak memory usage: {used:.2f} MB")


# -- Multi-GPU & Future Clustering Placeholders -- 

def get_all_gpu_memory_summaries() -> List[Dict]:
    # Returns memory summaries across all visible CUDA devices.
    if not torch.cuda.is_available():
        return []
    
    return [get_memory_summary(torch.device(f"cuda:{i}")) for i in range(torch.cuda.device_count())]


def initialize_distributed_memory_hooks(strategy: str = "ddp"):
    # Placeholder for initializing memory profiling hooks for distributed training.
    logger.info(f"Initializing memory hooks for distributed strategy: {strategy}")
    # Future: Hook ingo DeepSpeed, FullyShardedDDP, HuggingFace Accelarate, etc.


"""
--- Future TODOs --
- Per-layer memory stats (needs torch.fx or custom hooks)
- TensorStore offload monitoring
- Remote memory telemetry push

"""




















