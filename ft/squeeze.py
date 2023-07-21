from typing import Type

from torch import nn

_KNOWN_SQUEEZERS = {}


def register_squeezer(name: str, model_class: Type[nn.Module]):
    _KNOWN_SQUEEZERS[name] = model_class


def create_squeezer(name: str):
    return _KNOWN_SQUEEZERS[name]()


class LinearSqueezer(nn.Module):
    def forward(self, x):
        return x.reshape(*x.shape[:-2], -1)


class LastSqueezer(nn.Module):
    def forward(self, x):
        return x[..., -1, :]


class MeanSqueezer(nn.Module):
    def forward(self, x):
        return x.mean(axis=-2)


register_squeezer('linear', LinearSqueezer)
register_squeezer('last', LastSqueezer)
register_squeezer('mean', MeanSqueezer)
