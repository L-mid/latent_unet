
import copy
import math
import types
import torch
import pytest
from torch import nn

# Import your Option B utilities
from trainer.ema_utils import EMA, EMASchedule, swap_into, unwrap_ddp

# ----- Tiny dummy model with one param and one buffer -----

class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(3, 3, bias=False)
        # non-floating buffer to ensure "copy verbatim" path in exercised
        self.register_buffer("counter", torch.tensor(0, dtype=torch.int64))


def clone_tensor(t):
    return t.detach().clone()
    


# ----- Unit tests -----

def test_update_math_basic():
    torch.manual_seed(0)
    model = Tiny()
    ema = EMA(model, schedule=EMASchedule(base=0.9, max=0.9, warmup_steps=0), store_on_cpu=False)

    # Prepare a known change
    w_old = clone_tensor(model.lin.weight.data)
    model.lin.weight.data.fill_(2.0)
    w_new = clone_tensor(model.lin.weight.data)

    # One update with decay=0.9: ema = 0.9*ema + 0.1*model
    ema.update(model, step=1)
    w_ema = ema.shadow.lin.weight.data

    expected = 0.9 * w_old + 0.1 * w_new
    assert torch.allclose(w_ema, expected, atol=1e-6), f"expected={expected}, actual={w_ema}"


def test_warmup_resets_exact_copy():
    torch.manual_seed(0)
    model = Tiny()
    ema = EMA(model, schedule=EMASchedule(base=0.5, max=0.5, warmup_steps=2), store_on_cpu=False)

    # Step 1: should hard copy
    model.lin.weight.data.normal_()
    ema.update(model, step=1)
    assert torch.allclose(ema.shadow.lin.weight, model.lin.weight)

    # Step 2: still hard copy
    model.lin.weight.data.normal_()
    ema.update(model, step=2)
    assert torch.allclose(ema.shadow.lin.weight, model.lin.weight)


def test_scheduled_decay_progression():
    model = Tiny()
    sch = EMASchedule(base=0.8, max=0.99, warmup_steps=100)
    ema = EMA(model, schedule=sch, store_on_cpu=False)

    # Early step closer to base
    d1 = sch.decay_at(10)
    # Late step close to max
    d2 = sch.decay_at(100)
    assert 0.8 < d1 < 0.99
    assert math.isclose(d2, 0.99, rel_tol=0, abs_tol=1e-12)


def test_state_dict_roundtrip():
    model = Tiny()
    ema1 = EMA(model, store_on_cpu=True)
    # mutate a bit
    model.lin.weight.data.uniform_(-0.1, 0.1)
    ema1.update(model, step=10)

    sd = copy.deepcopy(ema1.state_dict())
    ema2 = EMA(model, store_on_cpu=True)    # New instance
    ema2.load_state_dict(sd)

    for (n1, p1), (n2, p2) in zip(ema1.shadow.state_dict().items(), ema2.shadow.state_dict().items()):
        assert n1 == n2
        if torch.is_tensor(p1):
            assert torch.allclose(p1, p2)


def test_cpu_shadow_and_device_isolation():
    # Shadow on CPU, model can live elsewhere
    model = Tiny()
    ema = EMA(model, store_on_cpu=True)
    # Shadow should be on CPU
    for t in ema.shadow.state_dict().values():
        if torch.is_tensor(t):
            assert t.device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_pin_memory_flag_with_cuda():
    model = Tiny().cuda()
    ema = EMA(model, store_on_cpu=True, pin_memory=True)
    # At least one floating tensor should be pinnable
    floats = [t for t in ema.shadow.state_dict().values() if torch.is_tensor(t) and t.is_floating_point()]
    assert len(floats) > 0
    # Some builds report False is pinning is not supported; just ensure call works
    _ = [t.is_pinned() for t in floats] # should not raise


def test_update_every_skips_intermediate_steps():
    model = Tiny()
    ema = EMA(model, update_every=2, schedule=EMASchedule(warmup_steps=0), store_on_cpu=False)
    # Capture initial EMA
    w0 = clone_tensor(ema.shadow.lin.weight)

    # Step 1: skipped (no update)
    model.lin.weight.data.add_(1.0)
    ema.update(model, step=1)
    assert torch.allclose(ema.shadow.lin.weight, w0)

    # Step 2: update happens
    model.lin.weight.data.add_(1.0)
    ema.update(model, step=2)
    assert not torch.allclose(ema.shadow.lin.weight, w0)



def test_buffers_copied_vervatim_for_non_float():
    model = Tiny()
    ema = EMA(model, store_on_cpu=False)

    # change buffer, then update
    model.counter.fill_(7)
    ema.update(model, step=100)

    assert int(ema.shadow.counter.item()) == 7


def test_unwrap_ddp():
    model = Tiny()
    # Fake a DDP-like wrapper that exposes .module
    wrapper = types.SimpleNamespace(module=model)
    assert unwrap_ddp(wrapper) is model
    assert unwrap_ddp(model) is model


def test_swap_into_context_restores_weights():
    torch.manual_seed(0)
    model = Tiny()
    # Make EMA differ from model
    ema = EMA(model, store_on_cpu=False)
    model.lin.weight.data.fill_(3.0)
    ema.update(model, step=50)

    # Backup original to compare later
    original = copy.deepcopy(model.state_dict())

    with swap_into(ema, model):
        # Inside: model should match EMA
        for k, v in model.state_dict().items():
            assert torch.allclose(v, ema.shadow.state_dict()[k])

    # After exit: model restored
    for (k1, v1), (k2, v2) in zip(model.state_dict().items(), original.items()):
        assert k1 == k2
        assert torch.allclose(v1, v2)



















