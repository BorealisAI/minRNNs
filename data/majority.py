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


class MajoritySampler:
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

        num_classes = VOCAB_SIZE - 1
        count = F.one_hot(tokens, num_classes=num_classes)
        total_count = count.sum(dim=1)

        majority = (total_count + 0.001 *
                    torch.arange(num_classes).unsqueeze(0)).argmax(dim=-1, keepdim=True)

        batch.x = torch.cat(
            (tokens, (torch.ones(batch_size, 1) * EMPTY).int()), dim=-1
        )
        batch.y = majority

        # Convert x to one_hot
        batch.x = batch.x.to(device)
        batch.y = batch.y.to(device)

        return batch
