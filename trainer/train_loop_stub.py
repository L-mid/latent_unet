
import torch
from utils.failure_injection_utils.failpoints import failpoints

def train_one_epoch(model, loader, device="cpu", logger=None, on_interrupt=None, empty_cache=None):
    model.train()
    seen = 0
    try:
        for batch in loader:
            exc = failpoints.should_raise("train.step")
            if exc: raise exc
            x = batch["image"].to(device)
            try:
                y = model(x)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    if empty_cache: empty_cache()
                    # skip batch
                    continue
                raise
            seen += x.size(0)
        return seen
    except KeyboardInterrupt:
        if on_interrupt: 
            on_interrupt()
        raise