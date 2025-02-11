# Copyright (c) 2025-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
from attrdict import AttrDict

VOCAB_SIZE = 4
EMPTY = VOCAB_SIZE - 1  # <EMPTY> value
CYCLE_LENGTH = 5


class CycleNavSampler:
    def __init__(self, seed=None):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    def sample(self, batch_size, length, device="cpu"):
        batch = AttrDict()

        # Randomly generate tokens (skipping the SOS and EOS token values)
        actions = torch.randint(
            low=0,
            high=VOCAB_SIZE - 1,
            size=(batch_size, length),
        )
        batch.x = torch.cat(
            (actions, (torch.ones(batch_size, 1) * EMPTY).int()), dim=-1
        )
        batch.y = (actions - 1).sum(dim=-1, keepdim=True) % CYCLE_LENGTH
        # batch.y = (actions).sum(dim=-1, keepdim=True) % CYCLE_LENGTH

        # Convert x to one_hot
        batch.x = batch.x.to(device)
        batch.y = batch.y.to(device)

        return batch
