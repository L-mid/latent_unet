
import pytest
from pathlib import Path
from torch.utils.data import DataLoader
from data.dataloader_stub import DummyDS
from model.model_stub import Model
from trainer.train_loop_stub import train_one_epoch
from utils.failure_injection_utils.failpoints import failpoints

@pytest.mark.failure_injection
def test_keyboard_interrupt_triggers_intterupt_checkpoint(fp, tmp_path):
    model = Model()
    ds = DummyDS(n=8); loader = DataLoader(ds, batch_size=2)
    interrupt_flag = {"saved": False}

    def on_interrupt():
        (tmp_path / "interrupt.ckpt").write_text("INT")
        interrupt_flag["saved"] = True

    # Interrupt on 3rd step
    fp.nth("train.step", n=3, exc=KeyboardInterrupt())

    with pytest.raises(KeyboardInterrupt):
        train_one_epoch(model, loader, on_interrupt=on_interrupt)
    
    assert interrupt_flag["saved"]
    assert (tmp_path / "interrupt.ckpt").read_text() == "INT"

