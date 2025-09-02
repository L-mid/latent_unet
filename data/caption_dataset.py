
import os
import json
import csv
import logging
from PIL import Image
from torch.utils.data import Dataset
from .dataset_registry import register_dataset
from .tokenizer import build_tokenizer
import importlib

logger = logging.getLogger(__name__)


@register_dataset("CaptionDataset")
class CaptionDataset(Dataset):
    # Paired image-caption(Dataset)
    # Supports JSONL, TSV, or CSV metadata files.

    def __init__(self, cfg, transforms, mode="train"):
        self.cfg = cfg
        self.transforms = transforms
        self.mode = mode

        self.metadata_path = cfg.data.metadata_file
        self.root = cfg.data.image_root
        self.format = cfg.data.metadata_format.lower()

        self.entries = self._load_metadata()

        # Tokenizer for captions
        self.tokenizer = build_tokenizer(cfg) #here 

        logger.info(f"[DATASET] Loaded {len(self.entries)} entries for {mode} mode")

    def _load_metadata(self):
        entries = []
        if self.format == "jsonl":
            with open(self.metadata_path, "r") as f:
                for line in f:
                    obj = json.loads(line)
                    entries.append({"image": obj["image"], "caption": obj["caption"]}) #? formatting?
        elif self.format == "tsv":
            with open(self.metadata_path, "r") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    entries.append({"image": parts[0], "caption": parts[1]})
        elif self.format == "csv":
            with open(self.metadata_path, "r") as f:
                reader = csv.reader(f)
                for row in reader:
                    entries.append({"image": row[0], "caption": row[1]})
        else:
            raise ValueError(f"Unsupported metadata format: {self.format}")
        return entries

    def __len__(self):
        return len(self.entries)
    
    def __getitem__(self, idx):
        entry = self.entries[idx]
        img_path = os.path.join(self.root, entry["image"])
        caption = entry["caption"]

        image = Image.open(img_path).convert("RGB") 
        if self.transforms:
            image = self.transforms(image)

        tokenized = self.tokenizer.encode(caption)
        input_ids = tokenized["input_ids"].squeeze(0)   # remove batch dim

        return {
            "image": image,
            "caption": caption,
            "input_ids": input_ids,
            "path": img_path,
        }