# Copyright (c) 2025-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
from attrdict import AttrDict

VOCAB_SIZE = 11
EMPTY = VOCAB_SIZE - 1  # <EMPTY> value
OUT_SIZE = VOCAB_SIZE - 1


class BucketSortSampler:
    def __init__(self, seed=None):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    def sample(self, batch_size, length, device="cpu"):

        tokens = torch.randint(
            low=0,
            high=VOCAB_SIZE - 1,
            size=(batch_size, length),
        )
        sorted_tokens, _ = torch.sort(tokens, dim=-1)

        batch = AttrDict()
        batch.x = torch.cat(
            (tokens, (torch.ones(batch_size, length) * EMPTY).int()), dim=-1
        )
        batch.y = sorted_tokens

        # Convert x to one_hot
        batch.x = batch.x.to(device)
        batch.y = batch.y.to(device)

        return batch
