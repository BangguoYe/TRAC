# coding=utf-8
# Copyright 2026 Bangguo Ye, Yuanwei Zhang, Xiaoqun Zhang
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

HIDDEN_SIZE_TO_TENSOR_SHAPE={
    768: [8, 8, 12],
    1024: [8, 8, 16],
    1280: [8, 8, 20],
    4096: [16, 16, 16],
    5120: [16, 16, 20],
}

HIDDEN_SIZE_TO_TENSOR_RANK={
    768: [1, 16, 24],
    1024: [1, 16, 24],
    1280: [1, 24, 48],
    4096: [1, 24, 48],
    5120: [1, 30, 60],
}

HIDDEN_SIZE_TO_TENSOR_RANK_A={
    768: [1, 8, 12],
    1024: [1, 8, 12],
    1280: [1, 12, 24],
    4096: [1, 12, 24],
    5120: [1, 15, 30],
}

HIDDEN_SIZE_TO_TENSOR_RANK_B={
    768: [1, 24, 36],
    1024: [1, 24, 36],
    1280: [1, 36, 72],
    4096: [1, 36, 72],
    5120: [1, 45, 90],
}

PARAM_NAME = ['q', 'v'] # ['q', 'k', 'v', 'proj', 'fc1', 'fc2']

PARAM_STRUCTURE = {'trainable_dim': {'A': [0], 'B': [0]},
                   'random_dim': {'A': None, 'B': None},
                   'shared_trainable_dim': {'A': [2], 'B': [2]},
                   'shared_random_dim': {'A': [1], 'B': [1]},}
