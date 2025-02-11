# Copyright (c) 2025-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from data.selective_copy import SelectiveCopyTaskSampler
from data.parity_check import ParityCheckSampler
from data.even_pairs import EvenPairsSampler
from data.cycle_nav import CycleNavSampler
from data.bucket_sort import BucketSortSampler
from data.majority import MajoritySampler
from data.majority_count import MajorityCountSampler
from data.missing_duplicate import MissingDuplicateSampler

samplers = {
    'selective_copy': SelectiveCopyTaskSampler,
    'parity_check': ParityCheckSampler,
    'even_pairs': EvenPairsSampler,
    'cycle_nav': CycleNavSampler,
    'bucket_sort': BucketSortSampler,
    'majority': MajoritySampler,
    'majority_count': MajorityCountSampler,
    'missing_duplicate': MissingDuplicateSampler,
}
