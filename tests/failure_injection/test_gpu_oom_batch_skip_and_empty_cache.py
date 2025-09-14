
import pytest, torch
from torch.utils.data import DataLoader
from model.model_stub import Model
from data.dataloader_stub import DummyDS
from trainer.train_loop_stub import train_one_epoch

@pytest.mark.failure_injection
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_oom_skips_batch_and_clears_cache(fp):
    # Make the first forward raise CUDA OOM
    fp.once("gpu.forward", exc=RuntimeError("CUDA out of memory."))

    model = Model().cuda()
    ds = DummyDS(n=4); loader = DataLoader(ds, batch_size=2)
    called = {"ec": 0}
    def empty_cache():
        called["ec"] += 1
        torch.cuda.empty_cache()
    seen = train_one_epoch(model, loader, device="cuda", empty_cache=empty_cache)
    assert called["ec"] == 1
    # Epoch completes; we processed at least one successful batch
    assert seen >= 2

