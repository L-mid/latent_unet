
import torch
from torch.utils.data import Dataset, DataLoader
from utils.failure_injection_utils.failpoints import failpoints

class DummyDS(Dataset):
    def __init__(self, n=8, c=3, h=8, w=8, fail_on_idx=None, nth=None):
        self.n = n; self.c=c; self.h=h; self.w=w
        self._count = 0
        self.fail_on_idx = fail_on_idx
        self.nth = nth

    def __len__(self): return self.n

    def __getitem__(self, idx):
        exc = failpoints.should_raise("dataloader.__gettiem__")
        if exc: raise exc
        if self.fail_on_idx is not None and idx == self.fail_on_idx:
            raise RuntimeError("decode failed [dataloader]")
        if self.nth is not None:
            self._count += 1
            if self._count == self.nth:
                raise RuntimeError("decode failed [dataloader]")
        return {"image": torch.randn(self.c, self.h, self.w)}       # ha baddd