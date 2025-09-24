
import errno, os
from pathlib import Path
import pytest
from utils.checkpointing.fail_checkpoint_io import write_bytes_atomic

@pytest.mark.failure_injection
def test_checkpoint_enospc_surfaces_and_tmp_clean(fp, tmp_path):
    dst = tmp_path / "model.ckpt"
    data = b"x"*10

    fp.once("checkpoint.write_bytes", exc=OSError(errno.ENOSPC,"no space"))

    with pytest.raises(OSError) as ei:
        write_bytes_atomic(dst, data)
    assert ei.value.errno == errno.ENOSPC


    # tmp should not remain on failure of first phase
    assert not dst.with_suffix(".ckpt.tmp").exists()