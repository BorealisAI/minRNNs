# Copyright (c) 2025-present, Royal Bank of Canada.
# Copyright (c) 2024, Phil Wang
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#####################################################################################
# Code leverages module implementations
# from https://github.com/lucidrains/minGRU-pytorch by Phil Wang which is licensed under MIT License.
# You may obtain a copy of the License at 
# 
# https://github.com/lucidrains/minGRU-pytorch/blob/main/LICENSE
#
####################################################################################


import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Module


class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return F.normalize(x, dim=-1) * self.scale * (self.gamma + 1)


class BatchNorm(Module):
    def __init__(self, dim, momentum=0.9):
        super().__init__()
        self.norm = nn.BatchNorm1d(dim, momentum=momentum)

    def forward(self, x):
        if len(x.shape) == 3:
            B, L, D = x.shape
            return self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            raise ValueError


class GEGLU(Module):
    def __init__(
        self,
        dim,
        mult_bias=True
    ):
        super().__init__()
        self.mult_bias = nn.Parameter(torch.ones(dim)) if mult_bias else 1.

    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.gelu(gate) * x * self.mult_bias


def FeedForward(dim, mult=4, dropout=0.1):
    dim_inner = int(dim * mult)
    return nn.Sequential(
        nn.Linear(dim, dim_inner * 2),
        GEGLU(dim_inner),
        nn.Linear(dim_inner, dim),
        nn.Dropout(dropout)
    )


class CausalDepthWiseConv1d(Module):
    def __init__(self, dim, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.net = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=kernel_size, groups=dim),
            nn.Conv1d(dim, dim, kernel_size=1)
        )

    def forward(self, x):
        x = x.transpose(1, 2)  # b n d -> b d n
        x = F.pad(x, (self.kernel_size - 1, 0), value=0.)
        x = self.net(x)
        return x.transpose(1, 2)  # b d n -> b n d
