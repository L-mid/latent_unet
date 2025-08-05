
from .vanilla_attention import VanillaAttention
from .window_attention import WindowAttention
from .flash_attention import FlashAttention

import os
import importlib

package_dir = os.path.dirname(__file__)
for file in os.listdir(package_dir):
    if file.endswith(".py") and file not in ["__init__.py", "registry.py", "base_attention.py"]:
        module_name = f"{__name__}.{file[:-3]}"
        importlib.import_module(module_name)



