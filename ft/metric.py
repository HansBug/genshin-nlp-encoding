from typing import Mapping

import torch
from torch import nn
from treevalue import FastTreeValue


class Accuracy(nn.Module):
    def __init__(self, n_classes: int, mode: str = 'native'):
        nn.Module.__init__(self)
        self.n_classes = n_classes
        self.mode = mode
        assert self.mode in {'native', 'mean'}

    def forward(self, logits, labels):
        values = (logits.argmax(axis=-1) == labels).type(torch.float32)
        if self.mode == 'native':
            return values
        elif self.mode == 'mean':
            return values.mean()


class MultiHeadAccuracy(nn.Module):
    def __init__(self, column_n_classes: Mapping[str, int], mode='native'):
        nn.Module.__init__(self)
        mapping = {
            name: Accuracy(n_classes, mode)
            for name, n_classes in column_n_classes.items()
        }
        self.losses = nn.ModuleDict(mapping)
        self.losses_tv = FastTreeValue(mapping)

    def forward(self, logits, labels):
        return self.losses_tv(logits, labels)
