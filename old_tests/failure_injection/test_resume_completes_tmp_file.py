
import pytest
from utils.checkpointing.fail_checkpoint_io import write_bytes_atomic, resume_recover, CrashBetweenTmpAndRename

@pytest.mark.failure_injection
def test_resume_completes_tmp_file(fp, tmp_path):
    dst = tmp_path / "weights.bin"
    data = b"DATA"

    # Crash between tmp write and rename
    fp.once("checkpoint.after_tmp_write", exc=CrashBetweenTmpAndRename())

    with pytest.raises(CrashBetweenTmpAndRename):
        write_bytes_atomic(dst, data)

    # Now resume should complete move from .tmp -> final
    resume_recover(dst)
    assert dst.read_bytes() == data

    




















