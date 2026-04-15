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

from .tensor_cfg import (
    HIDDEN_SIZE_TO_TENSOR_RANK, 
    HIDDEN_SIZE_TO_TENSOR_RANK_A, 
    HIDDEN_SIZE_TO_TENSOR_RANK_B, 
    HIDDEN_SIZE_TO_TENSOR_SHAPE,
    PARAM_NAME,
    PARAM_STRUCTURE
    )

from .layers import Linear, LoraLayer
from .trac import TracConfig, TracModel
from .peft_model import PeftModel

from .mapping import get_peft_config, get_peft_model

from .utils import (
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    mark_lora_layernorm_cls_trainable,
    compute_trainable_param
)