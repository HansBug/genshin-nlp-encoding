from typing import Type

from torch import nn

_KNOWN_MODELS = {}


def register_model(name: str, model_class: Type[nn.Module], *args, **kwargs):
    _KNOWN_MODELS[name] = (model_class, args, kwargs)


def create_model(name: str, *args, **kwargs) -> nn.Module:
    model_class, args_, kwargs_ = _KNOWN_MODELS[name]
    return model_class(*args_, *args, **kwargs_, **kwargs)
