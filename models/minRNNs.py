# Copyright (c) 2025-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Module

from models.associative_scan import associative_scan_log
from models.helpers import g, log_g
from utils.misc import exists


class minGRU(Module):
    def __init__(self,
                 dim,
                 expansion_factor,
                 use_init_hidden_state,
                 **kwargs):
        super().__init__()
        dim_inner = int(dim * expansion_factor)
        self.to_hidden_and_gate = Linear(dim, dim_inner * 2)
        self.to_out = Linear(dim_inner, dim)

        self.init_hidden_state = nn.Parameter(
            torch.randn(dim_inner), requires_grad=True) if use_init_hidden_state else None

    def forward(self, x, prev_state=None):
        seq_len = x.shape[1]
        hidden, gate = self.to_hidden_and_gate(x).chunk(2, dim=-1)

        if seq_len == 1:
            # handle sequential
            tilde_h = g(hidden)
            gate = gate.sigmoid()

            if exists(prev_state):
                (prev_hidden, _) = prev_state
                out = (1-gate) * prev_hidden + gate * tilde_h
            elif exists(self.init_hidden_state):
                prev_hidden = g(self.init_hidden_state)
                out = (1-gate) * prev_hidden + gate * tilde_h
            else:
                out = gate * tilde_h

            (next_hidden, next_log_hidden) = (out[:, -1:], out[:, -1:].log())
        else:
            log_coeffs = -F.softplus(gate)

            log_z = -F.softplus(-gate)
            log_tilde_h = log_g(hidden)
            log_values = log_z + log_tilde_h

            if exists(prev_state) or exists(self.init_hidden_state):
                if exists(prev_state):
                    (_, prev_log_hidden) = prev_state
                else:
                    prev_log_hidden = log_g(self.init_hidden_state)
                prev_log_hidden = prev_log_hidden.repeat((log_values.shape[0], 1, 1))
                log_values = torch.cat((prev_log_hidden, log_values), dim=1)
                log_coeffs = F.pad(log_coeffs, (0, 0, 1, 0))

            log_out = associative_scan_log(
                log_coeffs, log_values, return_log=True)

            out = torch.exp(log_out[:, -seq_len:])

            (next_hidden, next_log_hidden) = (out[:, -1:], log_out[:, -1:])

        out = self.to_out(out)
        return out, (next_hidden, next_log_hidden)


class minLSTM(Module):
    def __init__(self,
                 dim,
                 expansion_factor,
                 forget_bias_init_scale,
                 use_coeff_norm,
                 use_init_hidden_state,
                 **kwargs
                 ):
        super().__init__()

        dim_inner = int(dim * expansion_factor)
        self.to_hidden_and_gate = Linear(dim, dim_inner * 3)
        self.use_coeff_norm = use_coeff_norm

        # Rescale the forget gate's bias init value
        if exists(forget_bias_init_scale):
            to_add = torch.zeros(self.to_hidden_and_gate.bias.shape)
            to_add[dim_inner: 2*dim_inner] = torch.ones(dim_inner)*forget_bias_init_scale
            self.to_hidden_and_gate.bias = nn.Parameter(
                to_add + self.to_hidden_and_gate.bias)

        self.to_out = Linear(dim_inner, dim)

        self.init_hidden_state = nn.Parameter(
            torch.randn(dim_inner), requires_grad=True) if use_init_hidden_state else None
        

    def forward(self, x, prev_state=None):
        seq_len = x.shape[1]
        hidden, forget_gate, input_gate = self.to_hidden_and_gate(
            x).chunk(3, dim=-1)

        if seq_len == 1:
            # handle sequential
            tilde_h = g(hidden)
            input_gate = input_gate.sigmoid()
            forget_gate = forget_gate.sigmoid()

            if self.use_coeff_norm:
                norm = forget_gate + input_gate
                forget_gate, input_gate = forget_gate/norm, input_gate/norm

            if exists(prev_state):
                (prev_hidden, _) = prev_state
                out = forget_gate * prev_hidden + input_gate * tilde_h
            elif exists(self.init_hidden_state):
                prev_hidden = g(self.init_hidden_state)
                out = forget_gate * prev_hidden + input_gate * tilde_h
            else:
                out = input_gate * tilde_h

            (next_hidden, next_log_hidden) = (out[:, -1:], out[:, -1:].log())
        else:
            # parallel
            if self.use_coeff_norm:
                diff = F.softplus(-forget_gate) - F.softplus(-input_gate)
                log_coeffs = -F.softplus(diff)
                log_val_coeffs = -F.softplus(-diff)
            else:
                log_coeffs = -F.softplus(-forget_gate)
                log_val_coeffs = -F.softplus(-input_gate)

            log_tilde_h = log_g(hidden)
            log_values = log_val_coeffs + log_tilde_h

            if exists(prev_state) or exists(self.init_hidden_state):
                if exists(prev_state):
                    (_, prev_log_hidden) = prev_state
                else:
                    prev_log_hidden = log_g(self.init_hidden_state)
                prev_log_hidden = prev_log_hidden.repeat((log_values.shape[0], 1, 1))
                log_values = torch.cat((prev_log_hidden, log_values), dim=1)
                log_coeffs = F.pad(log_coeffs, (0, 0, 1, 0))

            log_out = associative_scan_log(
                log_coeffs, log_values, return_log=True)

            out = torch.exp(log_out[:, -seq_len:])

            (next_hidden, next_log_hidden) = (out[:, -1:], log_out[:, -1:])

        out = self.to_out(out)
        return out, (next_hidden, next_log_hidden)


sequence_modules = {
    'minGRU': minGRU,
    'minLSTM': minLSTM,
}
