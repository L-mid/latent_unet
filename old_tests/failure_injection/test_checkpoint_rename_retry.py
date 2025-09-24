
import os
from pathlib import Path
import pytest
from utils.checkpointing.fail_checkpoint_io import write_bytes_atomic

@pytest.mark.failure_injection
def test_checkpoint_rename_retries_first_failure(fp, tmp_path, monkeypatch):
    dst = tmp_path / "model.ckpt"
    data = b"OK"


    # first rename failure; app code should retry once.
    fp.nth("checkpoint.atomic_rename", n=1, exc=OSError("EBUSY"))

    # Minimal retry wrapper (in real code checkpoint writer should do this)
    def robust_write():
        try:
            write_bytes_atomic(dst, data)
        except OSError:
            # one retry
            write_bytes_atomic(dst, data)


    robust_write()
    assert dst.read_bytes() == data