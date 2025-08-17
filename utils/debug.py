
import contextlib
import time
import logging
from typing import Optional, Dict

class DebugManager:
    # Central debug manager to control verbosity and logging across modules.

    def __init__(self):
        self.enabled = False
        self.verbose = False
        self.tracked_modules = set()
        self.log_timestamps = False
        self.log_namespace = True

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False

    def toggle(self):
        self.enabled = not self.enabled

    def track(self, module_name: str):
        self.tracked_modules.add(module_name)

    def untrack(self, module_name: str):
        self.tracked_modules.discard(module_name)

    def reset(self):
        self.tracked_modules.clear()

    def is_tracked(self, name: str):
        return name in self.tracked_modules or not self.tracked_modules
    
    def set_verbose(self, flag: bool):
        self.log_timesteps = flag 

    def set_namespace(self, flag: bool):
        self.log_namespace = flag


# Singleton debug manager
debug = DebugManager()

def debug_log(msg: str, name: Optional[str] = None):
    # Central logging helper
    if not debug.enabled:
        return
    
    timestamp = f"[{time.strftime('%H:%M:%S')}]" if debug.log_timestamps else ""
    namespace = f"[{name}]" if name and debug.log_namespace else ""
    print(f"{timestamp}{namespace} {msg}")


@contextlib.contextmanager
def debug_section(name: str, print_enter_exit: bool = True):
    # Context manager for scoped debug logs.
    if debug.enabled and debug.is_tracked(name):
        if print_enter_exit:
            debug_log(">> Enter", name=name)
        start = time.time()
        yield
        elapsed = time.time - start
        debug_log(f"<< Exit ({elapsed:.3f}s)",name=name)
    else:
        yield


def attach_debug_hooks(module, module_name=""):
    # Optional: Register forward/backward hooks for module.

    if not debug.enabled:
        return
    
    def forward_hook(mod, inp, outp):
        debug_log(f"Forward: {mod.__class__.__name__}, Input shape: {inp[0].shape}, Output shape: {outp.shape}, name=module_name")

    module.register_forward_hook(forward_hook)






















