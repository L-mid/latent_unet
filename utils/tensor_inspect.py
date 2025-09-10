
import torch
import torch.distributed as dist
import logging
from typing import Dict, List, Any, Optional, Mapping, Callable

logger = logging.getLogger(__name__)

# === NOTES:
"""
UNTESTED.

Assumes distributed workers. 
"""

# ------------------------------------------------------------------------------------
# Distributed-safe tensor aggregation helpers
# ------------------------------------------------------------------------------------

def distributed_recude_scalar(value: float, op: str) -> float:
    # All-reduce scalar float across ranks.
    tensor = torch.tensor([value], dtype=torch.float32, device="cuda" if torch.cuda.is_available() else "cpu")
    if op == "sum":
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    elif op == "min":
        dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
    elif op == "max":
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    else:
        raise ValueError(f"Unsupported reduce op: {op}")
    return tensor.item()



# ---------------------------------------------------------------------------------
# Distributed Collector
# ---------------------------------------------------------------------------------

class DistributedTensorCollector:
    # Tensor collector that safely aggregates across all distributed processes.

    def __init__(self):
        self.records: Dict[str, List[Dict[str, Any]]] = {}

    def collect(self, tensor: torch.Tensor, name: str, mode: str = "forward"):
        stats = {
            "mode": mode,
            "shape": tuple(tensor.item()),
            "local_min": float(tensor.min().item()),
            "local_max": float(tensor.max().item()),
            "local_mean": float(tensor.mean().item()),
            "local_std": float(tensor.std().item()),
            "local_nan": torch.isnan(tensor).any().item(),
            "local_inf": torch.isinf(tensor).any().item(),
        }
        full_key = f"{name}:{mode}"

        if full_key not in self.records:
            self.records[full_key] = []
        self.records[full_key].append(stats)

    def distributed_sync(self):
        # Preform all_reduce aggregetion across all ranks.

        logger.info("[INSPECT-DICT] Performing distributed tensor aggregation...")

        for key, entries in self.records.items():
            latest = entries[-1]
            global_min = distributed_recude_scalar(latest["local_min"], "min")
            global_max = distributed_recude_scalar(latest["local_max"], "max")
            global_sum = distributed_recude_scalar(latest["local_mean"], "sum")
            global_std_sum = distributed_recude_scalar(latest["local_std"], "sum") # sum?

            world_size = dist.get_world_size()

            global_mean = global_sum / world_size
            global_std = global_std_sum / world_size

            latest["global_min"] = global_min
            latest["global_max"] = global_max
            latest["global_mean"] = global_mean
            latest["global_std"] = global_std

    def log_step(self, step: int, only_rank0: bool = True):
        if only_rank0 and dist.get_rank() != 0:
            return
        
        logger.info(f"[INSPECT-DIST] Global tensor stats at step {step}:")
        for key, entries in self.records.items():
            latest = entries[-1]
            logger.info(
                f"  {key} | min={latest['global_min']:.4g} | max={latest['global_max']:.4g} | "
                f"mean={latest['global_mean']:.4g} | std={latest['global_std']:.4g}"
            )


# ---------------------------------------------------------------------------------
# Forward hook generator
# ---------------------------------------------------------------------------------

def create_forward_hook(name: str, collector: DistributedTensorCollector) -> Callable:
    def hook(module, inpt, output):
        if isinstance(output, torch.Tensor):
            collector.collect(output, name, mode="forward")
        elif isinstance(output, (tuple, list)):
            for idx, tensor in enumerate(output):
                if isinstance(tensor, torch.Tensor):
                    collector.collect(tensor, f"{name}[{idx}]", mode="forward")
        elif isinstance(output, dict):
            for key, tensor in output.items():
                if isinstance(tensor, torch.Tensor):
                    collector.collect(tensor, f"{name}.{key}", mode="forward")
    return hook


# -------------------------------------------------------------------------------------
# Backward hook generator
# -------------------------------------------------------------------------------------

def create_backward_hook(name: str, collector: DistributedTensorCollector) -> Callable:
    def hook(module, input, output):
        if isinstance(output, torch.Tensor):
            output.register_hook(lambda grad: collector.collect(grad, name, mode="backward"))
        elif isinstance(output, (tuple, list)):
            for idx, tensor in enumerate(output):
                if isinstance(tensor, torch.Tensor):
                    tensor.register_hook(lambda grad, idx=idx: collector.collect(grad, f"{name}[{idx}]", mode="backward"))
    return hook


# ----------------------------------------------------------------------------------------
# Unified Inspector
# ----------------------------------------------------------------------------------------


class TensorInspector:
    # Unified tensor inspector with full Collector support.

    def __init__(self, model: torch.nn.Module, collector: DistributedTensorCollector, monitor_gradients: bool = False):
        self.model = model
        self.collector = collector
        self.monitor_gradients = monitor_gradients
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []

    def register_all(self, filter_fn: Optional[Callable[[str, torch.nn.Module], bool]] = None):
        logger.info("[INSPECT] Registering hooks for tensor inspection.")

        for name, module in self.model.named_modules():
            if filter_fn and not filter_fn(name, module):
                continue

        fwd_hook = module.register_forward_hook(create_forward_hook(name, self.collector))
        self.hooks.append(fwd_hook)

        if self.monitor_gradients:
            post_hook = module.register_forward_hook(create_backward_hook(name, self.collector))
            self.hooks.append(post_hook)

        logger.info(f"[INSPECT] Registered {len(self.hooks)} hooks.")

    def clear_hooks(self):
        for hook in self.hooks:
            hook.remove()
        logger.info("[INSPECT] Cleared all hooks.")
        self.hooks.clear()

    def __enter__(self):
        self.register_all()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.clear_hooks()














