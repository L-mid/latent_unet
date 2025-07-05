
import pytest

def should_run_test(config, category, module, test_name):
    """Returns True is the test is enabled based on the test_config.ymal structure."""
    
    test_cfg = config.test_config
    
    # Global override
    if test_cfg.run_all:
        return True
    
    # 2. Category check
    if not hasattr(test_cfg, category):
        raise ValueError(f"[should_run_test] Category '{category}' not found in test.yaml. Check your test control schema.")

    category_cfg = getattr(test_cfg, category)

    # 3. Category-level override
    if getattr(test_cfg, f"run_all_{category}", False):
        return True
    
    # 4. Module check
    if not hasattr(category_cfg, module):
        raise ValueError(f"[should_run_test] Module '{module}' not found under category '{category}' in test_config.yaml.")
    
    module_cfg = getattr(category_cfg, module)

    # 5. Expected fields check
    if not hasattr(module_cfg, "enabled") or not hasattr(module_cfg, "tests"):
        raise ValueError(
            f"[should_run_test] Module '{module}' under category '{category}' must have both 'enabled' and 'tests' fields."
        )
    
    # 6. Run logic
    if module_cfg.enabled:
        if test_name in module_cfg.tests:
            return True
        else:
            raise RuntimeError(
                f"[should_run_test] Test '{test_name}' is not listed in test_config.yaml. Check for typo?"
            )

    return module_cfg.enabled and test_name in module_cfg.tests



# -----------------------------------------------------------------------
# Auto-Skipping Decorator
# -----------------------------------------------------------------------

import pytest
from functools import wraps

def controlled_test(category, module):
    """
    Decorator that auto-skips tests using the test_config fixture,
    based on test name YMAL schema
    """
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            # Look for 'test_config' in positional or keyword args
            test_config = kwargs.get("test_config", None)
            
            # Fallback: extract 'test_config' from args
            if not test_config:
                for arg in args:
                    if isinstance(arg, dict) and "test_config" in arg:
                        test_config = arg["test_config"]
                        break
                    if hasattr(arg, "test_config"): # if test_config is nested
                        test_config = arg.test_config
                        break
            
            if not test_config:
                    raise RuntimeError(
                        f"Test {fn.__name__} must recive 'test_config' fixture - missing or not"
                    )
                
            # If we found the test_config fixture
            if not should_run_test(test_config, category, module, fn.__name__):
                pytest.skip(f"Test {module}.{fn.__name__} disabled via test_config.yaml")

            return fn(*args, **kwargs)
        return wrapper
    return decorator



