
import torch.nn as nn

class MidBlock():
    def __init__(self, dim, time_embed_dim, resblock, attention):
        super().__init__()
        self.layer = nn.Identity()

    def forward(self, x):
        return self.layer(x)
    