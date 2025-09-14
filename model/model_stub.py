
import torch
from utils.failure_injection_utils.failpoints import failpoints

class Model(torch.nn.Module):
    def forward(self, x):
        exc = failpoints.should_raise("gpu.forward")
        if exc: raise exc
        return x * 2 # (not good!)