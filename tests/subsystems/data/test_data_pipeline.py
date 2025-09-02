
import os
import tempfile
import shutil
import json
import torch
from PIL import Image
import importlib
from data import build_loader
from omegaconf import OmegaConf
import pytest


# ----------------------------------------------------------------------------------
# Helpers for generating small synthetic datasets
# ----------------------------------------------------------------------------------

def create_dummy_image(path):
    img = Image.new("RGB", (64, 64), color=(255, 0, 0))
    img.save(path)

def create_dummy_basic_dataset(tmp_path):
    root = os.path.join(tmp_path, "basic")
    os.makedirs(os.path.join(root, "cat"))
    os.makedirs(os.path.join(root, "dog"))
    create_dummy_image(os.path.join(root, "cat", "cat1.jpg"))
    create_dummy_image(os.path.join(root, "dog", "dog1.jpg"))
    return root

def create_dummy_caption_dataset(tmp_path):
    img_root = os.path.join(tmp_path, "captions/images")
    meta_file = os.path.join(tmp_path, "captions/metadata.jsonl")
    os.makedirs(img_root)
    create_dummy_image(os.path.join(img_root, "img1.jpg"))  # this image is RGB!

    with open(meta_file, "w") as f:
        json.dump({"image": "img1.jpg", "caption": "A red square."}, f)
        f.write("\n")

    return img_root, meta_file


pytest.importorskip("open_clip", reason="tokenizers/external plugins installed (might miss some if some but not other installations)")
# ---------------------------------------------------------------------------------
# Test: BasicImageFolder pipeline():
# -------------------------------------------------------------------------------

def test_basic_dataset_pipeline():
    tmp_path = tempfile.mkdtemp()
    try:
        root = create_dummy_basic_dataset(tmp_path)

        cfg = OmegaConf.create({
            "data": {
                "name": "BasicImageFolder",  #???
                "path": root,
                "image_size": 64,
                "distributed": False,
                "train": {"batch_size": 2, "num_workers": 0} 
            }
        })

        loader = build_loader(cfg, mode="train")      

        for batch in loader: # it tried
            assert "image" in batch
            assert "label" in batch
            print("Basic dataset batch OK:", batch["image"].shape)
            break

    finally:
        shutil.rmtree(tmp_path)


# ------------------------------------------------------------------------------
# Test: CationDataset pipeline
# -----------------------------------------------------------------------------

def test_caption_dataset_pipeline():
    tmp_path = tempfile.mkdtemp()
    try:
        img_root, meta_file = create_dummy_caption_dataset(tmp_path)

        cfg = OmegaConf.create({
            "data": {
                "name": "CaptionDataset",
                "image_size": 32,
                "image_root": img_root,
                "metadata_file": meta_file,
                "metadata_format": "jsonl",
                "distributed": False,
                "train": {"batch_size": 1, "num_workers": 0},
                "tokenizer": {
                    "name": "clip",
                    "max_length": 77,
                }
            }
        })

        loader = build_loader(cfg, mode="train")    

        for batch in loader:
            assert "image" in batch
            assert "input_ids" in batch
            print("Caption dataset batch OK:", batch["image"].shape)
            break

    finally: 
        shutil.rmtree(tmp_path)




# ------------------------------------------------------------------------------
# Run tests (or integrate into pytest later)
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    print("Running data pipeline tests...")
    test_basic_dataset_pipeline()
    test_caption_dataset_pipeline()
    print("All data pipeline tests passed!")

