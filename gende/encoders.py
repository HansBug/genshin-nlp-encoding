from typing import Callable

from torch import nn

_REGISTERED_ENCODERS = {}


def register_encoder(name: str, encoder_builder: Callable[..., nn.Module]):
    _REGISTERED_ENCODERS[name] = encoder_builder


def create_encoder(name: str, *args, **kwargs) -> nn.Module:
    return _REGISTERED_ENCODERS[name](*args, **kwargs)
