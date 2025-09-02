
import logging

logger = logging.getLogger(__name__)


# ===
"""
change name to .registry?

Also, there seems to be a problem in key differentiation?
"""

# Temporary placeholder, we will fully fill this one datasets are written.

DATASET_REGISTRY = {}

def register_dataset(name):
    # Decorator for registering new datasets.

    def decorator(cls):
        DATASET_REGISTRY[name] = cls
        logger.info(f"[DATASET] Registered dataset: {name}")
        # names should be registering here?
        return cls
    return decorator



def build_dataset_from_registry(cfg, transforms, mode="train"):
    # Instantsiate dataset from registry.

    name = cfg.data.name            # A "pass in all cfg" method
    dataset_cls = DATASET_REGISTRY.get(name)

    if dataset_cls is None:
        raise ValueError(f"Unknown dataset name: {name}")
    
    dataset = dataset_cls(cfg, transforms, mode) # is the call for the dataset init
    print(dataset_cls, "[Resgitry]")
    return dataset # it gets to this point

