# Copyright (c) 2025-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn.functional as F
from attrdict import AttrDict

VOCAB_SIZE = 64
EMPTY = VOCAB_SIZE - 1  # <EMPTY> value
OUT_SIZE = VOCAB_SIZE - 1


class MajorityCountSampler:
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

        count = F.one_hot(tokens, num_classes=OUT_SIZE)
        majority_count, _ = count.sum(dim=1).max(dim=-1, keepdim=True)

        batch.x = torch.cat(
            (tokens, (torch.ones(batch_size, 1) * EMPTY).int()), dim=-1
        )
        batch.y = majority_count

        # Convert x to one_hot
        batch.x = batch.x.to(device)
        batch.y = batch.y.to(device)

        return batch
