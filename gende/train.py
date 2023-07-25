import os
from typing import Optional, List

import numpy as np
import torch
from accelerate import Accelerator
from ditk import logging
from hbutils.random import global_seed
from torch.optim import lr_scheduler
from torch.utils.data import random_split, DataLoader
from tqdm.auto import tqdm
from treevalue import FastTreeValue, union

from .dataset import MarkedTextDataset
from .loss import MultiHeadFocalLoss
from .metric import MultiHeadAccuracy
from .models import create_model
from .plot import plt_export, plt_confusion_matrix
from .session import TrainSession

_DEFAULT_TEXT_COLUMN = 'desc_en'
_DEFAULT_DATA_COLUMNS = [
    '生效阶段', '伤害相关', '回复血量',
    '元素骰子', '元素类型', '元素充能', '角色专用',
]
_torch_cat = FastTreeValue.func(subside=True)(torch.cat)


def train(workdir: str, model_name: str,
          datasource: str, text_column: str = _DEFAULT_TEXT_COLUMN, data_columns: List[str] = None,
          max_epochs: int = 500, learning_rate: float = 0.001, weight_decay: float = 1e-3, batch_size: int = 16,
          eval_epoch: int = 1, val_ratio: float = 0.2, token_size: int = 256,
          seed: Optional[int] = None):
    if seed is not None:
        # native random, numpy, torch and faker's seeds are includes
        # if you need to register more library for seeding, see:
        # https://hansbug.github.io/hbutils/main/api_doc/random/state.html#register-random-source
        logging.info(f'Globally set the random seed {seed!r}.')
        global_seed(seed)

    os.makedirs(workdir, exist_ok=True)
    accelerator = Accelerator(
        # mixed_precision=self.cfgs.mixed_precision,
        step_scheduler_with_optimizer=False,
    )

    logging.info(f'Loading from datasource {datasource!r}, '
                 f'text column: {text_column!r}, data columns: {data_columns!r}')
    dataset = MarkedTextDataset(
        datasource,
        text_column=text_column,
        data_columns=data_columns or _DEFAULT_DATA_COLUMNS,
        token_size=token_size,
    )
    test_cnt = int(len(dataset) * val_ratio)
    train_cnt = len(dataset) - test_cnt
    train_dataset, val_dataset = random_split(dataset, [train_cnt, test_cnt])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    test_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    logging.info(f'Creating model {model_name!r}, with {dataset.column_n_classes}')
    model = create_model(model_name, head_n_classes=dataset.column_n_classes)
    loss_fn = MultiHeadFocalLoss(dataset.column_n_classes)
    acc_fn = MultiHeadAccuracy(dataset.column_n_classes)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = lr_scheduler.OneCycleLR(
        optimizer, max_lr=learning_rate,
        steps_per_epoch=len(train_dataloader), epochs=max_epochs,
        pct_start=0.15, final_div_factor=20.
    )

    model, optimizer, train_dataloader, test_dataloader, scheduler, loss_fn, acc_fn = \
        accelerator.prepare(model, optimizer, train_dataloader, test_dataloader, scheduler, loss_fn, acc_fn)

    session = TrainSession(workdir, key_metric='acc/min')
    for epoch in range(1, max_epochs + 1):
        train_loss = 0.0
        train_total = 0
        train_accs = []
        train_pred, train_true = [], []
        for i, (inputs, labels_) in enumerate(tqdm(train_dataloader)):
            inputs, labels_ = FastTreeValue(inputs), FastTreeValue(labels_)
            inputs = inputs.to(accelerator.device)
            labels_ = labels_.to(accelerator.device)
            tvb = inputs['attention_mask'].size(0)

            optimizer.zero_grad()
            outputs = model(inputs)
            train_total += tvb
            train_accs.append(acc_fn(outputs, labels_))
            train_pred.append(outputs.argmax(axis=-1))
            train_true.append(labels_)

            loss = loss_fn(outputs, labels_)
            # loss.backward()
            accelerator.backward(loss)
            optimizer.step()
            train_loss += loss.item() * tvb
            scheduler.step()

        # metrics
        train_accs = _torch_cat(train_accs, axis=-1).mean()
        train_metrics = {'loss': train_loss / train_total}
        accs = []
        for key, value in train_accs.items():
            current_acc = value.detach().item()
            train_metrics[f'acc/{key}'] = current_acc
            accs.append(current_acc)
        train_metrics['acc/min'] = np.min(accs)
        train_metrics['acc/mean'] = np.mean(accs)

        # plots
        train_pred = _torch_cat(train_pred, axis=-1)
        train_true = _torch_cat(train_true, axis=-1)
        train_plots = {}
        for key, (t_pred, t_true) in union(train_pred, train_true).items():
            train_plots[f'cm/{key}'] = plt_export(
                plt_confusion_matrix,
                t_true, t_pred, dataset.column_labels[key],
                title=f'Train Confusion Epoch {epoch} - {key}',
                figsize=(6, 6),
            )
        session.tb_train_log(epoch, {**train_metrics, **train_plots})

        if epoch % eval_epoch == 0:
            model.eval()
            with torch.no_grad():
                test_loss = 0.0
                test_total = 0
                test_accs = []
                test_pred, test_true = [], []
                for i, (inputs, labels_) in enumerate(tqdm(test_dataloader)):
                    inputs, labels_ = FastTreeValue(inputs), FastTreeValue(labels_)
                    inputs = inputs.to(accelerator.device)
                    labels_ = labels_.to(accelerator.device)
                    tvb = inputs['attention_mask'].size(0)

                    outputs = model(inputs)
                    test_total += tvb
                    test_accs.append(acc_fn(outputs, labels_))
                    test_pred.append(outputs.argmax(axis=-1))
                    test_true.append(labels_)

                    loss = loss_fn(outputs, labels_)
                    test_loss += loss.item() * tvb

                # metrics
                test_accs = _torch_cat(test_accs, axis=-1).mean()
                test_metrics = {'loss': test_loss / test_total}
                accs = []
                for key, value in test_accs.items():
                    current_acc = value.detach().item()
                    test_metrics[f'acc/{key}'] = current_acc
                    accs.append(current_acc)
                test_metrics['acc/min'] = np.min(accs)
                test_metrics['acc/mean'] = np.mean(accs)

                # plots
                test_pred = _torch_cat(test_pred, axis=-1)
                test_true = _torch_cat(test_true, axis=-1)
                test_plots = {}
                for key, (t_pred, t_true) in union(test_pred, test_true).items():
                    test_plots[f'cm/{key}'] = plt_export(
                        plt_confusion_matrix,
                        t_true, t_pred, dataset.column_labels[key],
                        title=f'Test Confusion Epoch {epoch} - {key}',
                        figsize=(6, 6),
                    )
                session.tb_eval_log(epoch, model, {**test_metrics, **test_plots})
