# Copyright (c) 2025-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


def add_arguments(parser):
    # Task Specific Arguments
    parser.add_argument("--sequence_length", type=int, default=None)
    parser.add_argument("--num_tokens_memorize", type=int, default=None)

    return parser
