
import os, errno, tempfile, shutil
from pathlib import Path
from utils.failure_injection_utils.failpoints import failpoints #issue here

# === NOTES:
"""
May already be implemented in the checkpointing proper, CHECK.
"""

class CrashBetweenTmpAndRename(Exception): ...

def write_bytes_atomic(dst: Path, data: bytes):
    dst = Path(dst)
    tmp = dst.with_suffix(dst.suffix + ".tmp")

    # write tmp
    exc = failpoints.should_raise("checkpoint.write_bytes")
    if exc: raise exc
    tmp.write_bytes(data)

    exc = failpoints.should_raise("checkpoint.after_tmp_write")
    if exc: raise exc

    # atomic rename
    exc = failpoints.should_raise("checkpoint.atomic_rename")
    if exc: raise exc
    os.replace(tmp, dst)

def resume_recover(dst: Path):
    """Idempotent cleanup/completion on resume."""
    dst = Path(dst)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    if tmp.exists():
        # choose: complete or delete; here complete
        os.replace(tmp, dst)