
from .dataset_registry import build_dataset_from_registry
from .loader import create_dataloader
from .transforms import build_transforms
# sampler builder, might leave out
from . import caption_dataset, basic_dataset

# === NOTES
"""
Data should probably be under utils/ .
"""

def build_dataset(cfg, mode="train"):
    # Build dataset object from config.

    transforms = build_transforms(cfg, mode) # bad cfg!
    dataset = build_dataset_from_registry(cfg, transforms, mode) 
    return dataset # got this out

def build_loader(cfg, mode="train"):  
    # Build DataLoader object from config.

    dataset = build_dataset(cfg, mode) 
    sampler = None

    loader = create_dataloader(
        dataset=dataset,
        batch_size=cfg.data[mode]["batch_size"],   
        shuffle=(sampler is None and mode == "train"),
        sampler=sampler,
        num_workers=cfg.data[mode].get("num_workers", 4),
        collate_fn=None     # Optional, add custom collate_fn logic here.
    )
    return loader 



