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

import re
from typing import List, Optional, Union
from enum import Enum
from dataclasses import asdict, dataclass, field

import torch

from .tensor_rep import create_shared_params, get_module_name
from .layers import LoraLayer, Linear

from .utils import PeftConfig, PeftType, mark_lora_layernorm_cls_trainable
from .tensor_cfg import HIDDEN_SIZE_TO_TENSOR_SHAPE, HIDDEN_SIZE_TO_TENSOR_RANK_A, HIDDEN_SIZE_TO_TENSOR_RANK_B, PARAM_STRUCTURE

@dataclass
class TracConfig(PeftConfig):
    """
    This is the configuration class to store the configuration of a [`~peft_local_tensor.Lora`].

    Args:
        r (`int`): Lora attention dimension
        target_modules (`Union[List[str],str]`): The names of the modules to apply Lora to.
        lora_alpha (`float`): The alpha parameter for Lora scaling.
        lora_dropout (`float`): The dropout probability for Lora layers.
        merge_weights (`bool`):
            Whether to merge the weights of the Lora layers with the base transformer model in `eval` mode.
        fan_in_fan_out (`bool`): Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        enable_lora ( `List[bool]`): Used with `lora.MergedLinear`.
        bias (`str`): Bias type for Lora. Can be 'none', 'all' or 'lora_only'
        modules_to_save (`List[str]`):List of modules apart from LoRA layers to be set as trainable
            and saved in the final checkpoint.
    """

    r: int = field(default=8, metadata={"help": "Lora attention dimension"})
    hidden_size: int = field(default=768, metadata={"help": "Hidden attention dimension"})
    mlp_hidden_dim: int = field(default=3072, metadata={"help": "Hidden MLP dimension , 4 * hidden_size for BERT, ViT, GPT2, flexible for Llama"})
    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to replace with Lora."
            "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
        },
    )
    
    tensor_rank: int = field(default=8, metadata={"help": "Tensor rank for decomposition"})
    tensor_shape: dict = field(
        default_factory=lambda: HIDDEN_SIZE_TO_TENSOR_SHAPE, 
        metadata={"help": "Default tensor shape mapping"}
    )
    tensor_rank_A: dict = field(
        default_factory=lambda: HIDDEN_SIZE_TO_TENSOR_RANK_A, 
        metadata={"help": "Default tensor rank A mapping"}
    )
    tensor_rank_B: dict = field(
        default_factory=lambda: HIDDEN_SIZE_TO_TENSOR_RANK_B, 
        metadata={"help": "Default tensor rank B mapping"}
    )
    tensor_init: str = field(default='TTNorm', metadata={"help": "Initialization method for tensors"})
    param_structure: dict = field(
        default_factory=lambda: PARAM_STRUCTURE, 
        metadata={"help": "Default parameter structure"}
    )
    
    lora_alpha: int = field(default=None, metadata={"help": "Lora alpha"})
    lora_dropout: float = field(default=None, metadata={"help": "Lora dropout"})
    merge_weights: bool = field(
        default=False, metadata={"help": "Merge weights of the original model and the Lora model"}
    )
    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "Set this to True if the layer to replace stores weight like (fan_in, fan_out)"},
    )
    enable_lora: Optional[List[bool]] = field(default=None, metadata={"help": "Used with `lora.MergedLinear`."})
    bias: str = field(default="none", metadata={"help": "Bias type for Lora. Can be 'none', 'all' or 'lora_only'"})
    modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "List of modules apart from LoRA layers to be set as trainable and saved in the final checkpoint. "
            "For example, in Sequence Classification or Token Classification tasks, "
            "the final layer `classifier/score` are randomly initialized and as such need to be trainable and saved."
        },
    )
    
    scale_shared_tt_cores: bool = field(default=True, metadata={"help": "Use scalar for shared parameters"})
    vector_activation: str = field(default='none', metadata={"help": "Activation function for vectors"})
    scale_init: str = field(default='ones', metadata={"help": "Initialization for scale: 'zeros' or 'ones'"})
    use_fast_tt: bool = field(default=False, metadata={"help": "Whether to use fast Tensor Train implementation"})

    def __post_init__(self):
        self.peft_type = PeftType.LORA


class TracModel(torch.nn.Module):
    """
    Creates Low Rank Adapter (Lora) model from a pretrained transformers model.

    Args:
        model ([`transformers.PreTrainedModel`]): The model to be adapted.
        config ([`LoraConfig`]): The configuration of the Lora model.

    Returns:
        `torch.nn.Module`: The Lora model.

    """

    def __init__(self, config, model):
        super().__init__()
        self.peft_config = config
        self.model = model
        self._find_and_replace()
        mark_lora_layernorm_cls_trainable(self.model, self.peft_config.bias)
        self.forward = self.model.forward

    def _find_and_replace(self):
        is_target_modules_in_base_model = False
        key_list = [key for key, _ in self.model.named_modules()]
        shared_params = create_shared_params(peft_config=self.peft_config,
                                             backbone_model=self.model, 
                                             target_modules=self.peft_config.target_modules, 
                                             hidden_size=self.peft_config.hidden_size, 
                                             mlp_hidden_dim=self.peft_config.mlp_hidden_dim, 
                                             matrix_rank=self.peft_config.r)

        for key in key_list:
            if isinstance(self.peft_config.target_modules, str):
                target_module_found = re.fullmatch(self.peft_config.target_modules, key)
            else:
                target_module_found = any(key.endswith(target_key) for target_key in self.peft_config.target_modules)
            if target_module_found:
                if not is_target_modules_in_base_model:
                    is_target_modules_in_base_model = True
                parent, target, target_name = self._get_submodules(key)
                module_name = get_module_name(target_name)
                bias = target.bias is not None

                if isinstance(target, torch.nn.Linear) and self.peft_config.enable_lora is None:
                    new_module = Linear(self.peft_config, target.in_features, target.out_features, r=self.peft_config.r, module_name=module_name, \
                                        shared_params=shared_params, lora_alpha=self.peft_config.lora_alpha, lora_dropout=self.peft_config.lora_dropout, \
                                        fan_in_fan_out=self.peft_config.fan_in_fan_out, merge_weights=self.peft_config.merge_weights)
                    
                self._replace_module(parent, target_name, new_module, target)
        if not is_target_modules_in_base_model:
            raise ValueError(
                f"Target modules {self.peft_config.target_modules} not found in the base model. "
                f"Please check the target modules and try again."
            )

    def _get_submodules(self, key):
        parent = self.model.get_submodule(".".join(key.split(".")[:-1]))
        target_name = key.split(".")[-1]
        target = self.model.get_submodule(key)
        return parent, target, target_name

    def _replace_module(self, parent_module, child_name, new_module, old_module):
        setattr(parent_module, child_name, new_module)
        new_module.weight = old_module.weight
        if old_module.bias is not None:
            new_module.bias = old_module.bias
        if getattr(old_module, "state", None) is not None:
            new_module.state = old_module.state
            new_module.to(old_module.weight.device)

        # dispatch to correct device
        for name, module in new_module.named_modules():
            if "lora_" in name:
                module.to(old_module.weight.device)

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)

    @property
    def modules_to_save(self):
        return None

    def get_peft_config_as_dict(self, inference: bool = False):
        config = {k: v.value if isinstance(v, Enum) else v for k, v in asdict(self.peft_config).items()}
        if inference:
            config["inference_mode"] = True
        return config

    def _set_adapter_layers(self, enabled=True):
        for module in self.model.modules():
            if isinstance(module, LoraLayer):
                module.disable_adapters = False if enabled else True

    def enable_adapter_layers(self):
        self._set_adapter_layers(enabled=True)

    def disable_adapter_layers(self):
        self._set_adapter_layers(enabled=False)

