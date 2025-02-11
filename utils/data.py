# Copyright (c) 2025-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os
import os.path as osp
import random
import time

import torch
from attrdict import AttrDict
from tqdm import tqdm

from data import samplers
from utils.paths import eval_data_dir


def get_sampler(cfg: AttrDict, mode: str):
    if cfg.task == 'selective_copy':
        sampler = samplers[cfg.task](
            sequence_length=cfg.sequence_length,
            num_tokens_memorize=cfg.num_tokens_memorize,
            seed=cfg[f'{mode}_seed']
        )
    elif cfg.task in ['parity_check', 'even_pairs', 'cycle_nav', 'bucket_sort', 'majority', 'majority_count', 'missing_duplicate']:
        sampler = samplers[cfg.task](
            seed=cfg[f'{mode}_seed'],
        )
    else:
        raise ValueError
    return sampler


def get_eval_path(cfg: AttrDict, mode: str):
    path = osp.join(eval_data_dir, cfg.task)

    if cfg.task == 'selective_copy':
        filename = f"seq_len-{cfg.sequence_length}_"
    else:
        filename = ""

    seed = cfg[f'{mode}_seed']
    filename += f"seed-{seed}_{mode}"
    filename += ".tar"
    return path, filename


def get_loss(cfg):
    loss_types = {
        'selective_copy': torch.nn.CrossEntropyLoss(),
        'parity_check': torch.nn.CrossEntropyLoss(),
        'even_pairs': torch.nn.CrossEntropyLoss(),
        'cycle_nav': torch.nn.CrossEntropyLoss(),
        'bucket_sort': torch.nn.CrossEntropyLoss(),
        'majority': torch.nn.CrossEntropyLoss(),
        'majority_count': torch.nn.CrossEntropyLoss(),
        'missing_duplicate': torch.nn.CrossEntropyLoss(),
    }
    return loss_types[cfg.task]


def get_metric(cfg, ravg):
    """
        ravg: RunningAverage (see utils/log.py)
    """
    metric_names = {
        'selective_copy': 'acc',
        'parity_check': 'acc',
        'even_pairs': 'acc',
        'cycle_nav': 'acc',
        'bucket_sort': 'acc',
        'majority': 'acc',
        'majority_count': 'acc',
        'missing_duplicate': 'acc',
    }
    metric_name = metric_names[cfg.task]

    metric = ravg.get(
        metric_name
    )
    return metric


def gen_evalset(cfg: AttrDict, mode: str):
    print(f"Generating Evaluation Sets")

    sampler = get_sampler(cfg, mode=mode)
    batches = []
    for i in tqdm(range(cfg.eval_num_batches), ascii=True):
        batches.append(sample_data(cfg, sampler, mode))

    torch.manual_seed(time.time())
    torch.cuda.manual_seed(time.time())

    path, filename = get_eval_path(cfg, mode)
    if not osp.isdir(path):
        os.makedirs(path)
    torch.save(batches, osp.join(path, filename))


def compute_outs(cfg, y_pred, batch, loss):
    outs = AttrDict()
    if cfg.task == 'selective_copy':
        y_pred = y_pred[:, -cfg.num_tokens_memorize:]
    elif cfg.task in ['parity_check', 'even_pairs', 'cycle_nav', 'majority', 'majority_count', 'missing_duplicate']:
        y_pred = y_pred[:, -1:]
    elif cfg.task in ['bucket_sort']:
        length = y_pred.shape[1] // 2
        y_pred = y_pred[:, -length:]
    else:
        raise ValueError

    y_pred = y_pred.flatten(0, 1)
    y_target = batch.y.flatten(0, 1)
    outs.loss = loss(y_pred, y_target)
    outs.acc = (y_pred.argmax(-1) == y_target).float().mean()

    return outs


def sample_data(cfg, sampler, mode, device='cuda'):
    if mode == 'train':
        batch_size = cfg.train_batch_size // cfg.accum_iter
    else:  # Eval
        batch_size = cfg.eval_batch_size

    if cfg.task == 'selective_copy':
        batch = sampler.sample(batch_size=batch_size, device=device)
    elif cfg.task in ['parity_check', 'even_pairs', 'cycle_nav', 'bucket_sort', 'majority', 'majority_count', 'missing_duplicate']:
        if mode == 'train':  # Train
            length = random.randint(1, 40)
        else:  # Eval
            length = random.randint(40, 256)  # following xLSTM's setup

        batch = sampler.sample(batch_size=batch_size,
                               length=length, device=device)
    else:
        raise ValueError
    return batch
