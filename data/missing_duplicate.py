# Copyright (c) 2025-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
from attrdict import AttrDict
import random

VOCAB_SIZE = 10
EMPTY = VOCAB_SIZE - 1  # <EMPTY> value
OUT_SIZE = VOCAB_SIZE - 1


class MissingDuplicateSampler:
    def __init__(self, seed=None):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    def sample(self, batch_size, length, device="cpu"):
        batch = AttrDict()

        # Randomly generate tokens (skipping the SOS and EOS token values)
        tokens = torch.randint(
            low=0,
            high=VOCAB_SIZE - 1,
            size=(batch_size, length),
        )

        missing_index = random.randint(0, length-1)

        batch.x = torch.cat(
            (tokens, tokens[:, :missing_index], (torch.ones(batch_size, 1) * EMPTY).int()), dim=-1
        )
        batch.y = tokens[:, missing_index:missing_index+1]

        # Convert x to one_hot
        batch.x = batch.x.to(device)
        batch.y = batch.y.to(device)

        return batch
