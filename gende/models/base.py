from typing import Type

import torch
from torch import nn

_KNOWN_MODELS = {}


def register_model(name: str, model_class: Type[nn.Module], *args, **kwargs):
    _KNOWN_MODELS[name] = (model_class, args, kwargs)


def create_model(name: str, **kwargs) -> nn.Module:
    model_class, args_, kwargs_ = _KNOWN_MODELS[name]
    model = model_class(*args_, **kwargs_, **kwargs)
    model.__arguments__ = {'name': name, **kwargs}
    return model


def save_model_to_ckpt(model, file):
    arguments = getattr(model, '__arguments__', {}) or {}
    info = getattr(model, '__info__', {}) or {}
    torch.save({
        'state_dict': model.state_dict(),
        'arguments': arguments,
        'info': info,
    }, file)


def load_model_from_ckpt(file):
    data = torch.load(file, map_location='cpu')
    arguments = data['arguments'].copy()
    name = arguments.pop('name')
    model = create_model(name, **arguments)
    existing_keys = set(model.state_dict())
    state_dict = {key: value for key, value in data['state_dict'].items() if key in existing_keys}
    model.load_state_dict(state_dict)
    model.__info__ = data.get('info', {}) or {}

    return model
