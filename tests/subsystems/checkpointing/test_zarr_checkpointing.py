
import os
import tempfile
import torch
import logging
import numpy as np
import pytest
from pathlib import Path

from utils.zarr_checkpointing import (
    zarr_wrapper,
    zarr_core
)

# Dummy toy model for full system test
class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)
        self.bn = torch.nn.BatchNorm1d(5)


def test_full_zarr_checkpoint_roundtrip():
    logging.basicConfig(level=logging.INFO)

    # Setup dummy model, optimizer, scheduler
    model = ToyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    tmp_dir = tempfile.mkdtemp()
    checkpoint_path = Path(tmp_dir) / "zarr_test_store"

    # Perform full save 
    zarr_wrapper.save_model(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=3,
        step=500,
        path=checkpoint_path,
    )
        
    # Create new fresh model to load into
    model_reloaded = ToyModel()
    optimizer_reloaded = torch.optim.Adam(model_reloaded.parameters(), lr=.001)
    scheduler_reloaded = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # Full load
    state = zarr_wrapper.load_model(
        model=model_reloaded,
        optimizer=optimizer_reloaded,
        scheduler=scheduler_reloaded,
        path=checkpoint_path,
        strict=True,
    )

    # Validate tensor equality after roundtrip
    for name, param in model.state_dict().items():
        reloaded_param = model_reloaded.state_dict()[name]
        assert torch.allclose(param, reloaded_param, atol=1e-6), f"Mismatch in param: {name}"

    print("[TEST] Full checkpoint roundtrip suceeded ✅")

    # Cleanup
    import shutil
    shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    test_full_zarr_checkpoint_roundtrip()


