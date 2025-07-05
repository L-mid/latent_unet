
import importlib
from contextlib import contextmanager

@contextmanager
def patch_module_attr(module_path: str, attr_name: str, new_value):
    # Temporarily replace a module attribute (e.g., class, function) during a test.
    module = importlib.import_module(module_path)
    original_value = getattr(module, attr_name)
    setattr(module, attr_name, new_value)
    try:
        yield
    finally:
        setattr(module, attr_name, original_value)


