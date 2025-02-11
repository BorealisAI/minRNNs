# Copyright (c) 2025-present, Royal Bank of Canada.
# Copyright (c) 2024, Phil Wang
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#####################################################################################
# Model implementation is based on the implementation
# from https://github.com/lucidrains/minGRU-pytorch by Phil Wang which is licensed under MIT License.
# You may obtain a copy of the License at 
# 
# https://github.com/lucidrains/minGRU-pytorch/blob/main/LICENSE
#
####################################################################################


from torch import nn
from torch.nn import Module, ModuleList

from models.minRNNs import sequence_modules
from models.modules import (BatchNorm, CausalDepthWiseConv1d, FeedForward,
                            RMSNorm)
from utils.misc import exists

norms = {
    'BatchNorm': BatchNorm,
    'RMSNorm': RMSNorm
}


class Model(Module):
    def __init__(
        self,
        *,
        module,
        num_tokens,
        d_in,
        d_out,
        dim,
        depth,
        dropout,
        ff_mult,
        conv_kernel_size,
        enable_conv,
        enable_ff,
        norm_type,
        rnn_config
    ):
        super().__init__()
        if num_tokens is not None:
            self.token_emb = nn.Embedding(num_tokens, dim)
        else:
            self.token_emb = None
            self.layer_in = nn.Linear(d_in, dim)

        self.layers = ModuleList([])

        Norm = norms[norm_type]
        RNN_Module = sequence_modules[module]
        for _ in range(depth):
            self.layers.append(ModuleList([
                CausalDepthWiseConv1d(
                    dim, conv_kernel_size) if enable_conv else None,
                Norm(dim),
                RNN_Module(**rnn_config),
                Norm(dim),
                FeedForward(dim, mult=ff_mult,
                            dropout=dropout) if enable_ff else None,
            ]))

        self.norm = Norm(dim)
        self.to_out = nn.Linear(dim, d_out, bias=False)

    def forward(
        self,
        x,
        return_states=False,
        prev_states=None
    ):
        if exists(self.token_emb):
            x = self.token_emb(x)
        else:
            x = self.layer_in(x)

        next_prev_states = []

        if not exists(prev_states):
            prev_states = [None for _ in range(len(self.layers))]

        for (conv, norm, minlstm, ff_norm, ff), prev_state in zip(self.layers, prev_states):
            # conv
            if exists(conv):
                x = conv(x) + x
            
            # minRNN
            min_rnn_out, next_prev_state = minlstm(
                norm(x),
                prev_state=prev_state
            )
            x = min_rnn_out + x
            next_prev_states.append(next_prev_state)

            # feedforward
            if exists(ff):
                x = ff(ff_norm(x)) + x

        logits = self.to_out(self.norm(x))

        if not return_states:
            return logits

        return logits, next_prev_states
