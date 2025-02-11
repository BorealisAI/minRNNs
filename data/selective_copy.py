# Copyright (c) 2025-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
from attrdict import AttrDict

VOCAB_SIZE = 16
EMPTY = VOCAB_SIZE - 1  # <EMPTY> value
PRED = 0  # <PRED> value


class SelectiveCopyTaskSampler:
    def __init__(self, sequence_length, num_tokens_memorize, seed=None):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        self.sequence_length = sequence_length
        self.num_tokens_memorize = num_tokens_memorize

    def sample(self, batch_size=16, device="cpu"):
        batch = AttrDict()

        # Randomly generate tokens (skipping the SOS and EOS token values)
        tokens = torch.randint(
            low=1,
            high=VOCAB_SIZE - 1,
            size=(batch_size, self.num_tokens_memorize),
        )

        inds = torch.stack([
            torch.randperm(self.sequence_length)[:self.num_tokens_memorize]
            for _ in range(batch_size)
        ], 0)
        inds, _ = inds.sort(-1)

        # default is <EMPTY> tokens
        sequence = EMPTY * torch.ones(
            (batch_size, self.sequence_length), dtype=torch.long)
        
        # insert tokens to copy
        sequence.scatter_(-1, inds, tokens)

        # add <PRED> tokens
        markers = PRED * \
            torch.ones((batch_size, self.num_tokens_memorize),
                       dtype=torch.long)

        batch.x = torch.cat((sequence, markers), dim=1)
        batch.y = tokens

        batch.x = batch.x.to(device)
        batch.y = batch.y.to(device)

        return batch
