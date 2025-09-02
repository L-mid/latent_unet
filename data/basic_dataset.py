
import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset
from .dataset_registry import register_dataset
import logging

logger = logging.getLogger(__name__)


@register_dataset("BasicImageFolder")
class BasicImageFolderDataset(Dataset):
    # Standard ImageFolder-style dataset:
    # root/class_name/image.jpg

    def __init__(self, cfg, transforms, mode="train"): # train is swallowd
        self.root = cfg.data.path
        self.transforms = transforms


        # Build full file list
        self.samples = []
        self.class_to_idx = self._discover_classes()
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

        for class_name, class_idx in self.class_to_idx.items():
            class_dir = os.path.join(self.root, class_name)
            files = glob.glob(os.path.join(class_dir, "*.jpg")) + \
                    glob.glob(os.path.join(class_dir, "*.png")) + \
                    glob.glob(os.path.join(class_dir, "*.jpeg"))
            for f in files:
                self.samples.append((f, class_idx))

        logger.info(f"[DATASET] Loaded {len(self.samples)} samples across {len(self.class_to_idx)} classes")

        # For samplers that expect full label list
        self.class_labels = [label for _, label in self.samples]

    def _discover_classes(self):
        classes = [d for d in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, d))]
        classes.sort()
        class_to_idx = {class_name: idx for idx, class_name in enumerate(classes)}
        logger.info(f"[DATASET] Discovered classes: {class_to_idx}")
        return class_to_idx
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, class_idx = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transforms:
            img = self.transforms(img)
        return {
            "image": img,
            "label": class_idx,
            "path": img_path, 
        }












