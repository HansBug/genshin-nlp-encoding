import os
from typing import Optional, List

import torch
from accelerate import Accelerator
from ditk import logging
from hbutils.random import global_seed
from torch.optim import lr_scheduler
from torch.utils.data import random_split, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from treevalue import FastTreeValue

from .dataset import MarkedTextDataset
from .loss import MultiHeadFocalLoss
from .metric import MultiHeadAccuracy
from .models import create_model

_DEFAULT_TEXT_COLUMN = 'desc_en'
_DEFAULT_DATA_COLUMNS = [
    '生效阶段', '伤害相关', '回复血量',
    '元素骰子', '元素类型', '元素充能', '角色专用',
]
_torch_cat = FastTreeValue.func(subside=True)(torch.cat)


def train(workdir: str, model_name: str,
          datasource: str, text_column: str = _DEFAULT_TEXT_COLUMN, data_columns: List[str] = None,
          max_epochs: int = 500, learning_rate: float = 0.001, weight_decay: float = 1e-3, batch_size: int = 16,
          eval_epoch: int = 5, val_ratio: float = 0.2,
          seed: Optional[int] = None):
    if seed is not None:
        # native random, numpy, torch and faker's seeds are includes
        # if you need to register more library for seeding, see:
        # https://hansbug.github.io/hbutils/main/api_doc/random/state.html#register-random-source
        logging.info(f'Globally set the random seed {seed!r}.')
        global_seed(seed)

    os.makedirs(workdir, exist_ok=True)
    tb_writer = SummaryWriter(workdir)
    ckpt_dir = os.path.join(workdir, 'ckpts')
    os.makedirs(ckpt_dir, exist_ok=True)
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
    )
    test_cnt = int(len(dataset) * val_ratio)
    train_cnt = len(dataset) - test_cnt
    train_dataset, val_dataset = random_split(dataset, [train_cnt, test_cnt])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    logging.info(f'Creating model {model_name!r}, with {dataset.column_n_classes}')
    model = create_model(model_name, dataset.column_n_classes)
    loss_fn = MultiHeadFocalLoss(dataset.column_n_classes)
    acc_fn = MultiHeadAccuracy(dataset.column_n_classes)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = lr_scheduler.OneCycleLR(
        optimizer, max_lr=learning_rate,
        steps_per_epoch=len(train_dataloader), epochs=max_epochs,
        pct_start=0.15, final_div_factor=20.
    )

    model, optimizer, train_dataloader, test_dataloader, scheduler, loss_fn = \
        accelerator.prepare(model, optimizer, train_dataloader, test_dataloader, scheduler, loss_fn)

    for epoch in range(1, max_epochs + 1):
        logging.info(f'Epoch {epoch} start')
        train_loss = 0.0
        train_total = 0
        train_accs = []
        for i, (inputs, labels_) in enumerate(tqdm(train_dataloader)):
            inputs, labels_ = FastTreeValue(inputs), FastTreeValue(labels_)
            inputs = inputs.to(accelerator.device)
            labels_ = labels_.to(accelerator.device)

            optimizer.zero_grad()
            outputs = model(inputs)
            train_total += labels_.shape[0]
            train_accs.append(acc_fn(outputs, labels_))

            loss = loss_fn(outputs, labels_)
            # loss.backward()
            accelerator.backward(loss)
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            scheduler.step()

        train_accs = _torch_cat(train_accs, axis=-1).mean()
        tb_writer.add_scalar('train/loss', train_loss / train_total)
        for key, value in train_accs.items():
            tb_writer.add_scalar(f'train/{key}', value)

        if epoch % eval_epoch == 0:
            model.eval()
            with torch.no_grad():
                test_loss = 0.0
                test_total = 0
                test_accs = []
                for i, (ids, inputs, labels_) in enumerate(tqdm(test_dataloader)):
                    inputs, labels_ = FastTreeValue(inputs), FastTreeValue(labels_)
                    inputs = inputs.to(accelerator.device)
                    labels_ = labels_.to(accelerator.device)

                    outputs = model(inputs)
                    test_total += labels_.shape[0]
                    test_accs.append(acc_fn(outputs, labels_))

                    loss = loss_fn(outputs, labels_)
                    test_loss += loss.item() * inputs.size(0)

                test_accs = _torch_cat(test_accs, axis=-1).mean()
                tb_writer.add_scalar('test/loss', test_loss / test_total)
                for key, value in test_accs.items():
                    tb_writer.add_scalar(f'test/{key}', value)

                ckpt_file = os.path.join(ckpt_dir, f'model-{epoch}.ckpt')
                logging.info(f'Save epoch {epoch!r} to {ckpt_file!r}')
                torch.save(model.state_dict(), ckpt_file)
