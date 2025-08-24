
import os
import pytest
import tempfile
import torch
import asyncio
import importlib

# -----------------------------------------------------------------------------------
# Dummy test model
# -----------------------------------------------------------------------------------

class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)
        self.bn = torch.nn.BatchNorm1d(5)


# -----------------------------------------------------------------------------------
# Full end-to-end checkpoint test
# ------------------------------------------------------------------------------------


zarr = pytest.importorskip("zarr", reason="zarr not installed")
pytest_asyncio = pytest.importorskip("pytest_asyncio", reason="pytest-asyncio not installed")
def test_tensorstore_checkpoint_roundtrip():
    # Setup dummy model / optimizer / scheduler

    from utils.tensorstore_checkpointing import (
    tensorstore_wrapper,
    schema_utils,
    chunk_tuner,
    registry,
    metadata_utils,
)
    model = DummyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # Generate schema
    schema = schema_utils.generate_schema(model)    

    # generate metadata
    config_dict = {"lr": 0.001, "schedule": "step"}
    metadata = metadata_utils.generate_metadata(
        config_dict=config_dict,
        run_id="unit_test_run"
    )

    # Setup temporary test directory
    tmp_dir = tempfile.mkdtemp()
    checkpoint_path = os.path.join(tmp_dir, "ts_checkpoint")

    # Save checkpoint
    tensorstore_wrapper.save_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=3,
        step=42,
        path=checkpoint_path,   # this goes into kvstore, vaild
        schema=schema,
        metadata=metadata,
        storage_options=None    # Local test
    )

    # Create a new frech model to verify full reload
    model_reloaded = DummyModel()
    optimizer_reloaded = torch.optim.Adam(model_reloaded.parameters(), lr=0.001)
    scheduler_reloaded = torch.optim.lr_scheduler.StepLR(optimizer_reloaded, step_size=5, gamma=0.5)

    # Load checkpoint
    loaded_meta = tensorstore_wrapper.load_checkpoint(
        model=model_reloaded,
        optimizer=optimizer_reloaded,
        scheduler=scheduler_reloaded,
        path=checkpoint_path,   # this goes into kvstore, vaild
        schema=schema,
        strict=True
    )

    # Validate model weights identical after reload
    for name, orig_tensor in model.state_dict().items():
        reloaded_tensor = model_reloaded.state_dict()[name]
        assert torch.allclose(orig_tensor, reloaded_tensor, atol=1e-6), f"Tensor mismatch: {name}"

    print("[TEST] Model tensors identical ✅")

    # Validate metadata survived roundtrip
    assert loaded_meta["epoch"] == 3
    assert loaded_meta["step"] == 42
    print("[TEST] Metadata roundtrip succeeded ✅")

    # Cleanup
    import shutil
    shutil.rmtree(tmp_dir)

if __name__ == "__main__":
    test_tensorstore_checkpoint_roundtrip()




















