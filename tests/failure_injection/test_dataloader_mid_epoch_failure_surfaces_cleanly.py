
import pytest, torch
from torch.utils.data import DataLoader
from data.dataloader_stub import DummyDS
from model.model_stub import Model
from trainer.train_loop_stub import train_one_epoch


@pytest.mark.failure_injection
def test_dataloader_failure_crashes_clear_error(fp):
    # Fail on second sample
    ds = DummyDS(n=4, nth=2)
    loader = DataLoader(ds, batch_size=1)
    model = Model()

    with pytest.raises(RuntimeError):
        train_one_epoch(model, loader)