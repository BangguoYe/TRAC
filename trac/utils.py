# coding=utf-8
# This file contains code derived from the HuggingFace PEFT project.
# 
# Copyright 2023-present the HuggingFace Inc. team.
# Copyright (c) 2026 Bangguo Ye, Yuanwei Zhang, Xiaoqun Zhang
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

import enum
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Optional, Union

from huggingface_hub import hf_hub_download
from transformers.utils import PushToHubMixin

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

WEIGHTS_NAME = "adapter_model.bin"
CONFIG_NAME = "adapter_config.json"


# ==============================================================================
# ENUMS
# Defines the different types of PEFT methods and tasks supported.
# ==============================================================================

class PeftType(str, enum.Enum):
    PROMPT_TUNING = "PROMPT_TUNING"
    P_TUNING = "P_TUNING"
    PREFIX_TUNING = "PREFIX_TUNING"
    LORA = "LORA"
    BOTTLENECK = "BOTTLENECK"



class TaskType(str, enum.Enum):
    SEQ_CLS = "SEQ_CLS"
    SEQ_2_SEQ_LM = "SEQ_2_SEQ_LM"
    CAUSAL_LM = "CAUSAL_LM"
    TOKEN_CLS = "TOKEN_CLS"


# ==============================================================================
# CONFIGURATION CLASSES
# Base and specific configuration classes for PEFT models.
# ==============================================================================

@dataclass
class PeftConfigMixin(PushToHubMixin):
    r"""
    This is the base configuration class for PEFT adapter models. It contains all the methods that are common to all
    PEFT adapter models. This class inherits from `transformers.utils.PushToHubMixin` which contains the methods to
    push your model to the Hub. The method `save_pretrained` will save the configuration of your adapter model in a
    directory. The method `from_pretrained` will load the configuration of your adapter model from a directory.

    Args:
        peft_type (Union[[`~peft_local_tensor.utils.config.PeftType`], `str`]): The type of Peft method to use.
    """
    peft_type: Optional[PeftType] = field(default=None, metadata={"help": "The type of PEFT model."})

    @property
    def __dict__(self):
        return asdict(self)

    def to_dict(self):
        return self.__dict__

    def save_pretrained(self, save_directory, **kwargs):
        r"""
        This method saves the configuration of your adapter model in a directory.

        Args:
            save_directory (`str`):
                The directory where the configuration will be saved.
            **kwargs:
                Additional keyword arguments passed along to the `transformers.utils.PushToHubMixin.push_to_hub`
                method.
        """
        if os.path.isfile(save_directory):
            raise AssertionError(f"Provided path ({save_directory}) should be a directory, not a file")

        os.makedirs(save_directory, exist_ok=True)

        output_dict = self.__dict__
        output_path = os.path.join(save_directory, CONFIG_NAME)

        # save it
        with open(output_path, "w") as writer:
            writer.write(json.dumps(output_dict, indent=2, sort_keys=True))

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        r"""
        This method loads the configuration of your adapter model from a directory.

        Args:
            pretrained_model_name_or_path (`str`):
                The directory or the hub-id where the configuration is saved.
            **kwargs:
                Additional keyword arguments passed along to the child class initialization.
        """
        if os.path.isfile(os.path.join(pretrained_model_name_or_path, CONFIG_NAME)):
            config_file = os.path.join(pretrained_model_name_or_path, CONFIG_NAME)
        else:
            try:
                config_file = hf_hub_download(pretrained_model_name_or_path, CONFIG_NAME)
            except Exception:
                raise ValueError(f"Can't find config.json at '{pretrained_model_name_or_path}'")

        loaded_attributes = cls.from_json_file(config_file)

        config = cls(**kwargs)

        for key, value in loaded_attributes.items():
            if hasattr(config, key):
                setattr(config, key, value)

        return config

    @classmethod
    def from_json_file(cls, path_json_file, **kwargs):
        r"""
        Loads a configuration file from a json file.

        Args:
            path_json_file (`str`):
                The path to the json file.
        """
        with open(path_json_file, "r") as file:
            json_object = json.load(file)

        return json_object


@dataclass
class PeftConfig(PeftConfigMixin):
    """
    This is the base configuration class to store the configuration of a :class:`~peft_local_tensor.PeftModel`.

    Args:
        peft_type (Union[[`~peft_local_tensor.utils.config.PeftType`], `str`]): The type of Peft method to use.
        task_type (Union[[`~peft_local_tensor.utils.config.TaskType`], `str`]): The type of task to perform.
        inference_mode (`bool`, defaults to `False`): Whether to use the Peft model in inference mode.
    """

    base_model_name_or_path: str = field(default=None, metadata={"help": "The name of the base model to use."})
    peft_type: Union[str, PeftType] = field(default=None, metadata={"help": "Peft type"})
    task_type: Union[str, TaskType] = field(default=None, metadata={"help": "Task type"})
    inference_mode: bool = field(default=False, metadata={"help": "Whether to use inference mode"})


@dataclass
class PromptLearningConfig(PeftConfig):
    """
    This is the base configuration class to store the configuration of a Union[[`~peft_local_tensor.PrefixTuning`],
    [`~peft_local_tensor.PromptEncoder`], [`~peft_local_tensor.PromptTuning`]].

    Args:
        num_virtual_tokens (`int`): The number of virtual tokens to use.
        token_dim (`int`): The hidden embedding dimension of the base transformer model.
        num_transformer_submodules (`int`): The number of transformer submodules in the base transformer model.
        num_attention_heads (`int`): The number of attention heads in the base transformer model.
        num_layers (`int`): The number of layers in the base transformer model.
    """

    num_virtual_tokens: int = field(default=None, metadata={"help": "Number of virtual tokens"})
    token_dim: int = field(
        default=None, metadata={"help": "The hidden embedding dimension of the base transformer model"}
    )
    num_transformer_submodules: Optional[int] = field(
        default=None, metadata={"help": "Number of transformer submodules"}
    )
    num_attention_heads: Optional[int] = field(default=None, metadata={"help": "Number of attention heads"})
    num_layers: Optional[int] = field(default=None, metadata={"help": "Number of transformer layers"})


# ==============================================================================
# STATE DICT UTILITIES
# Functions for extracting and loading the state dictionary of PEFT models.
# ==============================================================================

def get_peft_model_state_dict(model, state_dict=None):
    """
    Get the state dict of the Peft model.

    Args:
        model ([`PeftModel`]): The Peft model. When using torch.nn.DistributedDataParallel, DeepSpeed or FSDP,
        the model should be the underlying model/unwrapped model (i.e. model.module).
        state_dict (`dict`, *optional*, defaults to `None`):
            The state dict of the model. If not provided, the state dict of the model
        will be used.
    """
    if state_dict is None:
        # state_dict = model.full_state_dict()
        if hasattr(model, 'full_state_dict'):
            state_dict = model.full_state_dict()
        else:
            state_dict = model.state_dict()
    if True:
        # to_return = lora_state_dict(model, bias=model.peft_config.bias)
        # adapted from `https://github.com/microsoft/LoRA/blob/main/loralib/utils.py`
        # to directly with the state dict which is necessary when using DeepSpeed or FSDP
        # bias = model.peft_config.bias
        bias = "none"
        if bias == "none":
            # 目前随机冻结部分的参数也需要保存，但理论上只需要保存生成其的伪随机种子就可以了，后续可以通过工程上的优化来实现
            to_return = {k: state_dict[k] for k in state_dict if (("lora_" in k) or ('classifier' in k))}
        elif bias == "all":
            to_return = {k: state_dict[k] for k in state_dict if "lora_" in k or "bias" in k}
        elif bias == "lora_only":
            to_return = {}
            for k in state_dict:
                if "lora_" in k:
                    to_return[k] = state_dict[k]
                    bias_name = k.split("lora_")[0] + "bias"
                    if bias_name in state_dict:
                        to_return[bias_name] = state_dict[bias_name]
        else:
            raise NotImplementedError
        
    return to_return


def set_peft_model_state_dict(model, peft_model_state_dict):
    """
    Set the state dict of the Peft model.

    Args:
        model ([`PeftModel`]): The Peft model.
        peft_model_state_dict (`dict`): The state dict of the Peft model.
    """

    model.load_state_dict(peft_model_state_dict, strict=False)
    if model.peft_config.peft_type != PeftType.LORA and model.peft_config.peft_type != PeftType.BOTTLENECK:
        model.prompt_encoder.embedding.load_state_dict(
            {"weight": peft_model_state_dict["prompt_embeddings"]}, strict=True
        )
    return model


# ==============================================================================
# MODEL UTILITIES
# Helper functions for model modifications and mathematical operations.
# ==============================================================================

def _set_trainable(model):
    if model.modules_to_save is not None:
        for name, param in model.named_parameters():
            if any(module_name in name for module_name in model.modules_to_save):
                param.requires_grad = True


def transpose(weight, fan_in_fan_out):
    return weight.T if fan_in_fan_out else weight


def mark_lora_layernorm_cls_trainable(model: nn.Module, bias: str = "none") -> None:
    for n, p in model.named_parameters():
        if ("lora_" in n and 'random_' not in n) or ('classifier' in n):
            p.requires_grad = True
        else:
            p.requires_grad = False
    if bias == "none":
        pass
    elif bias == "all":
        for n, p in model.named_parameters():
            if "bias" in n:
                p.requires_grad = True
    elif bias == "lora_only":
        from .layers import LoraLayer
        
        for m in model.modules():
            if isinstance(m, LoraLayer) and hasattr(m, "bias") and m.bias is not None:
                m.bias.requires_grad = True
    else:
        raise NotImplementedError


def compute_trainable_param(model):
    if isinstance(model, DistributedDataParallel):
        model = model.module

    total_param = 0
    peft_trainable_param = 0
    for n, p in model.named_parameters():
        total_param += torch.numel(p)
        if p.requires_grad:
            # Exclude task-specific heads and normalization layers from the PEFT parameter count.
            # These parameters share the same trainable settings across all PEFT methods (e.g., LoRA),
            # and their sizes vary across different model architectures. Therefore, we exclude them 
            # to focus on the core PEFT parameters.
            if ('head' not in n) and ('classifier' not in n) and ('norm' not in n) and ('LayerNorm' not in n):
                peft_trainable_param += torch.numel(p)

    return total_param, peft_trainable_param
