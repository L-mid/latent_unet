
from __future__ import annotations
import copy
from dataclasses import dataclass
from typing import Dict, Any, Iterable, Optional
import torch
from torch import nn

def unwrap_ddp(model: nn.Module) -> nn.Module:
    return getattr(model, "module", model)

@dataclass
class EMASchedule:
    base: float = 0.999
    max: float = 0.9999
    warmup_steps: int = 2_000
    def decay_at(self, step: int) -> float:
        if step <= 0: return self.base
        if step >= self.warmup_steps: return self.max
        t = step / self.warmup_steps
        return self.base + (self.max - self.base) * t
    
class EMA:
    """
    EMA with:
    - scheduled decay (warmpt -> max)
    - CPU shadow storage (saves VRAM)
    - updates only floating params; copies others (e.g., buffers) verbatim
    - DDP/FSDP-friendly (uses state_dict keys)
    """
    def __init__(
            self,
            model: nn.Module,
            schedule: Optional[EMASchedule] = None,
            store_on_cpu: bool = True,
            pin_memory: bool = False,
            update_every: int = 1,
    ):
        self.schedule = schedule or EMASchedule()
        self.update_every = max(1, int(update_every))
        self.num_updates = 0

        model = unwrap_ddp(model).eval()
        shadow = copy.deepcopy(model).eval()

        for p in shadow.parameters(): p.requires_grad_(False)
        for b in shadow.buffers(): b.requires_grad = False  # just in case

        if store_on_cpu:
            shadow.to("cpu", non_blocking=True)
            if pin_memory:
                for t in shadow.state_dict().values():
                    if t.is_floating_point(): t.pin_memory()

        self.shadow = shadow    # holds EMA weights (cpu or device)

    @torch.no_grad()
    def update(self, model: nn.Module, *, step: Optional[int] = None) -> None:
        model = unwrap_ddp(model).eval()

        # Always increment first
        self.num_updates += 1
        effective_step = step if step is not None else self.num_updates

        # Warmup hard-copy
        if effective_step <= self.schedule.warmup_steps:
            self.shadow.load_state_dict(model.state_dict())
            return
        
        # Throttle update frequency
        if (self.num_updates % self.update_every) != 0:
            return
        
        # Compute decay for this step
        decay = self.schedule.decay_at(effective_step)

        msd = model.state_dict()
        esd = self.shadow.state_dict()
        ema_device = next(iter(esd.values())).device

        for k, e in esd.items():
            m = msd[k]
            if torch.is_floating_point(e):
                if m.device != ema_device:
                    m = m.to(ema_device, non_blocking=True)
                e.mul_(decay).add_(m.detach(), alpha=1.0 - decay)
            else: 
                esd[k] = m.detach() if isinstance(m, torch.Tensor) else m

        print("num_updates", self.num_updates, "effective_step:", effective_step, "decay:", decay)


    def to(self, device: str | torch.device) -> "EMA":
        self.shadow.to(device, non_blocking=True)
        return self
    
    def state_dict(self) -> Dict[str, Any]:
        return {
            "shadow": self.shadow.state_dict(),
            "num_updates": self.num_updates,
            "schedule": {
                "base": self.schedule.base,
                "max": self.schedule.max,
                "warmup": self.schedule.warmup_steps,
            },
        }
    
    def load_state_dict(self, sd: Dict[str, Any]) -> None:
        self.shadow.load_state_dict(sd["shadow"])
        self.num_updates = int(sd["num_updates"])
        sch = sd.get("schedule", {})
        self.schedule = EMASchedule(
            base=float(sch.get("base", 0.999)),
            max=float(sch.get("max", 0.9999)),
            warmup_steps=int(sch.get("warmup", 2000)),
        )

class swap_into:
    """
    Context manager to temporarily load EMA weights into a *target* model
    (without allocating a second deep copy at runtime).
    """
    def __init__(self, ema: EMA, model: nn.Module):
        self.ema = ema
        self.model = model
        self.backup: Optional[Dict[{str, torch.Tensor}]] = None

    def __enter__(self):
        self._backup = copy.deepcopy(unwrap_ddp(self.model).state_dict())
        unwrap_ddp(self.model).load_state_dict(self.ema.shadow.state_dict(), strict=True)
        return self.model
    
    def __exit__(self, exc_type, exc, tb):
        unwrap_ddp(self.model).load_state_dict(self._backup, strict=True)
        self._backup = None
    















