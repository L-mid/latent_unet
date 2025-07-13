
import torch
import pytest
from model.unet import UNet



# -------------------------
# Dummy building blocks
# -------------------------

class DummyTimeEmbed(torch.nn.Module):
    def forward(self, t): return torch.randn(t.size(0), 64, device=t.device)

class DummyBlock(torch.nn.Module):
    def forward(self, x, temb): return x, x     # DownBlock behavior

class DummyMid(torch.nn.Module):
    def forward(self, x, temb): return x

class DummyUp(torch.nn.Module):
    def forward(self, x, skip, temb): return x + skip

class DummyHead(torch.nn.Module):
    def forward(self, x): return x


# -----------------------------
# Basic Functional Tests
# -----------------------------

class TestUNetBasic:
    def setup_method(self):
        self.B, self.C, self.H, self.W = 2, 3, 32, 32
        self.base = 16
        self.x = torch.randn(self.B, self.C, self.H, self.W)
        self.t = torch.randint(0, 100, (self.B,))

        self.unet = UNet(
            in_channels=self.C,
            base_channels=self.base,
            time_embedding=DummyTimeEmbed(),
            downs=torch.nn.ModuleList([DummyBlock(), DummyBlock()]), mid=DummyMid(),
            ups=torch.nn.ModuleList([DummyUp(), DummyUp()]),
            final_head=DummyHead()
        )

    def test_forward_pass_runs(self):
        out = self.unet(self.x, self.t)
        assert out.shape == (self.B, self.base, self.H, self.W)

    def test_device_transfer(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.unet.to(device)
        self.x = self.x.to(device)
        self.t = self.t.to(device)
        out = self.unet(self.x, self.t)
        assert out.device == torch.device(device)
        

# ------------------------------
# Backward & Gradient Tests
# ------------------------------

# Define gradient-flowing test components with safe scope
class GradTimeEmbed(torch.nn.Module):
    def forward(self, t): return torch.randn(t.size(0), 64, requires_grad=True)

class GradBlock(torch.nn.Module):
    def __init__(self, base): 
        super().__init__()
        self.conv = torch.nn.Conv2d(base, base, kernel_size=3, padding=1)
    def forward(self, x, temb): return self.conv(x), x

class GradMid(torch.nn.Module):
    def __init__(self, base): 
        super().__init__()
        self.conv = torch.nn.Conv2d(base, base, kernel_size=3, padding=1)
    def forward (self, x, temb): return self.conv(x)

class GradUp(torch.nn.Module):
    def __init__(self, base): 
        super().__init__()
        self.conv = torch.nn.Conv2d(base, base, kernel_size=3, padding=1)
    def forward(self, x, skip, temb): return self.conv(x + skip)

class GradHead(torch.nn.Module):
    def __init__(self, base): 
        super().__init__()
        self.out = torch.nn.Conv2d(base, 3, kernel_size=1)
    def forward(self, x): return self.out(x)


# Testing Gradients
class TestUNetGradients:
    def setup_method(self):
        self.B, self.C, self.H, self.W = 2, 3, 32, 32
        self.base = 16
        self.x = torch.randn(self.B, self.C, self.H, self.W, requires_grad=True)
        self.t = torch.randint(0, 1000, (self.B,))

        # Store all in self for tests.
        self.unet = UNet(
            in_channels=self.C,
            base_channels=self.base,
            time_embedding=GradTimeEmbed(),
            downs=torch.nn.ModuleList([GradBlock(self.base), GradBlock(self.base)]),
            mid=GradMid(self.base),
            ups=torch.nn.ModuleList([GradUp(self.base), GradUp(self.base)]),
            final_head=GradHead(self.base)
        )

    def test_backward_pass_no_nans(self):
        out = self.unet(self.x, self.t)
        loss = out.mean()
        loss.backward()

        for name, param in self.unet.named_parameters():
            if param.grad is not None:
                assert not torch.isnan(param.grad).any(), f"NaN in grad: {name}"


    def test_nonzero_grads(self):
        out = self.unet(self.x, self.t)
        loss = out.mean()
        loss.backward()

        grads = [p.grad for p in self.unet.parameters() if p.requires_grad]
        assert any(g is not None and g.abs().sum() > 0 for g in grads), "No gradients flowed"


# ----------------------------
# Optional: Hook Tracking
# ----------------------------

class TestUNetHooks:
    def setup_method(self):
        self.B, self.C, self.H, self.W = 2, 3, 32, 32
        self.x = torch.randn(self.B, self.C, self.H, self.W, requires_grad=True)
        self.t = torch.randint(0, 1000, (self.B,))

        self.unet = UNet(
            in_channels=self.C,
            base_channels=16,
            time_embedding=DummyTimeEmbed(),
            downs=torch.nn.ModuleList([DummyBlock(), DummyBlock()]),
            mid=DummyMid(),
            ups=torch.nn.ModuleList([DummyUp(), DummyUp()]),
            final_head=DummyHead()
        )

    def test_hooks_are_triggered(self):
        hits = []

        def hook_fn(mod, grad_in, grad_out):
            hits.append(mod)

        for mod in self.unet.modules():
            if any(p.requires_grad for p in mod.parameters()):
                mod.register_full_backward_hook(hook_fn)

        out = self.unet(self.x, self.t)
        out.mean().backward()

        assert len(hits) > 0, "No hooks triggered - check grad flow"








