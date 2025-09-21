
import torch, torch.nn as nn
import pytest
from trainer.optim_utils import EMA
from trainer.ema_utils import EMA as other_EMA

class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 1, bias=True)

    def forward(self, x):
        x = self.lin(x)
        return x

def flat_params(model: nn.Module):
        return torch.cat([p.detach().view(-1) for p in model.parameters()])
    
def assign_all(model: nn.Module, value: float):
        with torch.no_grad():
            for p in model.parameters():
                p.fill_(value)

def clone_like(model: nn.Module):
        c = TinyNet()
        c.load_state_dict(model.state_dict(), strict=True)
        return
    

@pytest.mark.parametrize("beta", [0.0, 0.5, 0.9, 0.999])
@pytest.mark.parametrize("diff_ema", [EMA])
def test_ema_math_scalar(beta, diff_ema):
    # Fake 'model' param as a 1D tensor; many EMA classes accept a module
    x_seq = [1.0, 2.0, 3.0, 4.0]
    # Manual EMA
    ema_manual = 0.0
    for t, x in enumerate(x_seq, start=1):
        ema_manual = beta * ema_manual + (1 - beta) * x

    EMA = diff_ema      # this is parameterized
    
    # Use your EMA on a TinyNet by writing values into its params each step.
    m = TinyNet()
    ema = EMA(m, decay=beta)

    # Init all weights to 0 so they mirror ema_manual start
    assign_all(m, 0.0)
    ema.ema_model.load_state_dict(m.state_dict())
    for x in x_seq:
        assign_all(m, x)
        ema.update()

    # Pull EMA shadow params and compare the average value
    w = ema.get_ema_model() if hasattr(ema, "get_ema_model") else None
    if w is not None:   # some EMAs materialize a model
        got = flat_params(w).mean().item()
    else:
        # or read internal shadow buffers directly:
        shadow = (
            torch.cat([t.view(-1) for t in ema.shadow_params.values()]) 
            if hasattr(ema, "shadow_params") 
            else flat_params(ema.ema_model)
        )
        got = shadow.mean().item()

    assert pytest.approx(ema_manual, rel=1e-6, abs=1e-6) == got



def test_ema_update_no_grad_keeps_graph_clean():
    m = TinyNet()
    for p in m.parameters():
         assert p.requires_grad

    ema = EMA(m, decay=0.9)

    # make a forward pass so params are in areal graph context
    x = torch.randn(2, 4)
    (m(x).sum()).backward() # create grads; EMA must not touch the grapsh

    # Call update outside of no_grad() on purpose; EMA must internally guard it
    ema.update()

    # Ensure EMA params (shadow) are not requiring grad / have no grad_fn
    for p in ema.ema_model.parameters():
        assert p.requires_grad is False
        assert p.grad is None
        assert p.grad_fn is None


def test_ema_state_dict_roundtrip(tmp_path):
    m = TinyNet()
    ema = EMA(m, decay=0.95) 
    assign_all(m, 5.0); ema.update()

    sd = ema.state_dict()
    # Simulate reload 
    ema2 = EMA(TinyNet(), decay=0.95)
    ema2.load_state_dict(sd)

    w1 = ema.ema_model.parameters()
    w2 = ema2.ema_model.parameters()
    for p1, p2 in zip(w1, w2):
        assert torch.allclose(p1, p2)


def test_ema_converges_to_fixed_point():
    m = TinyNet()
    ema = EMA(m, decay=0.99)
    assign_all(m, 4.2)
    ema.ema_model.load_state_dict(m.state_dict())
    for _ in range(200):
        ema.update()
    w = ema.ema_model
    assert torch.allclose(flat_params(w), flat_params(m), atol=1e-5, rtol=0)


