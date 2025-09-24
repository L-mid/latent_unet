
import os, tempfile
import pytest 
from utils.checkpointing.fail_checkpoint_io import write_bytes_atomic

# === NOTES:
"""
Added. Not implemented/double implimented.
CHECK if implemented in the checkpointers. 
"""

def robust_atomic_rename(dst, data: bytes, retries=1):
    for i in range(retries+1):
        try:
            write_bytes_atomic(dst, data) 
            return 
        except OSError:
            if i == retries:
                raise


@pytest.mark.failure_injection
def test_rename_retries_on_first_failure(fp, tmp_path):
    dst = tmp_path / "a.tmp" 

    fp.nth("checkpoint.atomic_rename", n=1, exc=OSError("simulated EBUSY")) 

    robust_atomic_rename(dst, (b"ok"), retries=1)

    assert dst.read_bytes() == b"ok"


