import json
import os
import re
import shutil
from typing import Optional, List, Tuple

import numpy as np
import torch
from PIL import Image
from ditk import logging

from .models import load_model_from_ckpt


def export_model_from_workdir(workdir, export_dir, name: Optional[str] = None) -> List[Tuple[str, str]]:
    model_filename = os.path.join(workdir, 'ckpts', 'best.ckpt')
    name = name or os.path.basename(os.path.abspath(workdir))

    model = load_model_from_ckpt(model_filename)
    _info = model.__info__

    metrics = {}
    plots = {}
    for key, value in _info.items():
        if isinstance(value, (int, float, str, type(None))) or \
                (isinstance(value, (torch.Tensor, np.ndarray)) and not value.shape):
            if isinstance(value, (torch.Tensor, np.ndarray)):
                value = value.tolist()

            metrics[key] = value
        elif isinstance(value, Image.Image):
            plots[key] = value
        else:
            logging.warn(f'Argument {key!r} is a {type(value)}, unable to export.')

    os.makedirs(export_dir, exist_ok=True)
    files = []

    ckpt_file = os.path.join(export_dir, f'{name}.ckpt')
    logging.info(f'Copying checkpoint to {ckpt_file!r}')
    shutil.copyfile(model_filename, ckpt_file)
    files.append((ckpt_file, 'model.ckpt'))

    meta_file = os.path.join(export_dir, f'{name}_meta.json')
    logging.info(f'Exporting meta-information of model to {meta_file!r}')
    with open(meta_file, 'w') as f:
        json.dump(model.__arguments__, f, sort_keys=True, indent=4, ensure_ascii=False)
    files.append((meta_file, 'meta.json'))

    metrics_file = os.path.join(export_dir, f'{name}_metrics.json')
    logging.info(f'Recording metrics to {metrics_file!r}')
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, sort_keys=True, indent=4, ensure_ascii=False)
    files.append((metrics_file, 'metrics.json'))

    for key, value in plots.items():
        norm_key = re.sub(r"[\W_]+", "_", key)
        plt_file = os.path.join(export_dir, f'{name}_plot_{norm_key}.png')
        logging.info(f'Save plot figure {key} to {plt_file!r}')
        value.save(plt_file)
        files.append((plt_file, f'plot_{norm_key}.png'))

    return files
