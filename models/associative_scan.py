# Copyright (c) 2025-present, Royal Bank of Canada.
# Copyright (c) 2023, GlassRoom
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#####################################################################################
# Scan implementation is based on the implementation
# from https://github.com/glassroom/heinsen_sequence by GlassRoom which is licensed under MIT License.
# You may obtain a copy of the License at 
# 
# https://github.com/glassroom/heinsen_sequence/blob/main/LICENSE
#
####################################################################################


def associative_scan_log(log_coeffs, log_values, return_log=False):
    a_star = log_coeffs.cumsum(dim=1)
    log_h0_plus_b_star = (log_values - a_star).logcumsumexp(dim=1)
    log_h = a_star + log_h0_plus_b_star

    if return_log:
        return log_h
    else:
        return log_h.exp()
