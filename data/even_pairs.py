# Copyright (c) 2025-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
from attrdict import AttrDict

VOCAB_SIZE = 3
EMPTY = VOCAB_SIZE - 1  # <EMPTY> value


class EvenPairsSampler:
    def __init__(self, seed=None):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    def sample(self, batch_size, length, device="cpu"):
        batch = AttrDict()

        tokens = torch.randint(
            low=0,
            high=VOCAB_SIZE - 1,
            size=(batch_size, length),
        )
        unequal_pairs = torch.logical_xor(tokens[:, :-1], tokens[:, 1:])

        batch.x = torch.cat(
            (tokens, (torch.ones(batch_size, 1) * EMPTY).int()), dim=-1
        )
        batch.y = unequal_pairs.sum(-1, keepdim=True) % 2

        # Convert x to one_hot
        batch.x = batch.x.to(device)
        batch.y = batch.y.to(device)

        return batch
