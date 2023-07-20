from typing import Tuple, Mapping

from torch import nn
from treevalue import FastTreeValue


class MLP(nn.Module):
    def __init__(self, in_features: int, out_features: int, layers: Tuple[int, ...] = (1024,)):
        nn.Module.__init__(self)
        self.in_features = in_features
        self.out_features = out_features
        self.layers = layers
        ios = [self.in_features, *self.layers, self.out_features]
        self.mlp = nn.Sequential(
            *(
                nn.Linear(in_, out_, bias=True)
                for in_, out_ in zip(ios[:-1], ios[1:])
            )
        )

    def forward(self, x):
        return self.mlp(x)


class MultiHeadMLP(nn.Module):
    def __init__(self, in_features: int, out_features: Mapping[str, int], layers: Tuple[int, ...] = (1024,)):
        nn.Module.__init__(self)
        self.in_features = in_features
        self.out_features = out_features
        self.layers = layers
        self.mlps = {
            o_name: MLP(in_features, o_feat, layers)
            for o_name, o_feat in self.out_features.items()
        }

    def forward(self, x):
        return FastTreeValue({
            name: mlp(x)
            for name, mlp in self.mlps.items()
        })
