# coding=utf-8
# Copyright 2023-present the HuggingFace Inc. team.
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

import os
from contextlib import contextmanager

import torch
from accelerate import dispatch_model, infer_auto_device_map
from accelerate.hooks import AlignDevicesHook, add_hook_to_module, remove_hook_from_submodules
from accelerate.utils import get_balanced_memory
from huggingface_hub import hf_hub_download
from transformers.utils import PushToHubMixin

from .trac import TracModel
from .utils import (
    WEIGHTS_NAME,
    PeftConfig,
    PeftType,
    _set_trainable,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)

class PeftModel(PushToHubMixin, torch.nn.Module):
    """
    Parameter-Efficient Fine-Tuning Model. Base model specifically tailored for TRAC, 
    an algorithm based on LoRA.
    
    Note: This implementation is streamlined for TRAC. If you need to use other PEFT 
    methods such as LoRA, Prompt Tuning, Prefix Tuning, P-Tuning, or Adapters, please use the 
    official `peft` library from Hugging Face Hub.

    Args:
        model ([`PreTrainedModel`]): The base transformer model used for Peft.
        peft_config ([`PeftConfig`]): The configuration of the Peft model.

    **Attributes**:
        - **base_model** ([`PreTrainedModel`]) -- The base transformer model used for Peft.
        - **peft_config** ([`PeftConfig`]) -- The configuration of the Peft model.
        - **modules_to_save** (`list` of `str`) -- The list of sub-module names to save when
        saving the model.
    """

    def __init__(self, model, peft_config: PeftConfig, use_peft_state_dict: bool=True):
        super().__init__()
        self.peft_config = peft_config
        self.base_model = model
        self.config = self.base_model.config
        self.modules_to_save = None

        # Replace the standard LoRA module with the TRAC module.
        if self.peft_config.peft_type == PeftType.LORA:
            self.base_model = TracModel(peft_config, model)

        if getattr(self.peft_config, "modules_to_save", None) is not None:
            self.modules_to_save = self.peft_config.modules_to_save
            _set_trainable(self)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.base_model_torch_dtype = getattr(model, "dtype", None)
        
        self.use_peft_state_dict = use_peft_state_dict
        self.full_state_dict = self.state_dict
        self.peft_state_dict = get_peft_model_state_dict(self)
        if self.use_peft_state_dict == True:
            self.state_dict = self._peft_state_dict

    def _peft_state_dict(self):
        self.peft_state_dict = get_peft_model_state_dict(self)
        return self.peft_state_dict
    
    def save_pretrained(self, save_directory, weight_name=WEIGHTS_NAME, **kwargs):
        r"""
        Args:
        This function saves the adapter model and the adapter configuration files to a directory, so that it can be
        re-loaded using the `LoraModel.from_pretrained` class method, and also used by the `LoraModel.push_to_hub`
        method.
            save_directory (`str`):
                Directory where the adapter model and configuration files will be saved (will be created if it does not
                exist).
            **kwargs:
                Additional keyword arguments passed along to the `push_to_hub` method.
        """
        if os.path.isfile(save_directory):
            raise ValueError(f"Provided path ({save_directory}) should be a directory, not a file")
        os.makedirs(save_directory, exist_ok=True)

        # save only the trainable weights
        output_state_dict = get_peft_model_state_dict(self, kwargs.get("state_dict", None))
        torch.save(output_state_dict, os.path.join(save_directory, weight_name))
        # os.makedirs(os.path.join(save_directory, weight_name), exist_ok=True)
        # torch.save(output_state_dict, f'{os.path.join(save_directory, weight_name)}/{weight_name}.pt')

        # save the config and change the inference mode to `True`
        if self.peft_config.base_model_name_or_path is None:
            self.peft_config.base_model_name_or_path = self.base_model.model.__dict__.get("name_or_path", None)
            
        inference_mode = self.peft_config.inference_mode
        self.peft_config.inference_mode = True
        self.peft_config.save_pretrained(save_directory)
        self.peft_config.inference_mode = inference_mode

    @classmethod
    def from_pretrained(cls, model, model_id, **kwargs):
        r"""
        Args:
        Instantiate a `LoraModel` from a pretrained Lora configuration and weights.
            model (`transformers.PreTrainedModel`):
                The model to be adapted. The model should be initialized with the `from_pretrained` method. from
                `transformers` library.
            model_id (`str`):
                The name of the Lora configuration to use. Can be either:
                    - A string, the `model id` of a Lora configuration hosted inside a model repo on
                        huggingface Hub
                    - A path to a directory containing a Lora configuration file saved using the
                        `save_pretrained` method, e.g., ``./my_lora_config_directory/``.
        """
        from .mapping import MODEL_TYPE_TO_PEFT_MODEL_MAPPING, PEFT_TYPE_TO_CONFIG_MAPPING
        
        # load the config
        config = PEFT_TYPE_TO_CONFIG_MAPPING[PeftConfig.from_pretrained(model_id).peft_type].from_pretrained(model_id)

        if getattr(model, "hf_device_map", None) is not None:
            remove_hook_from_submodules(model)

        if config.task_type not in MODEL_TYPE_TO_PEFT_MODEL_MAPPING.keys():
            model = cls(model, config)
        else:
            model = MODEL_TYPE_TO_PEFT_MODEL_MAPPING[config.task_type](model, config)

        # load weights if any
        if os.path.exists(os.path.join(model_id, WEIGHTS_NAME)):
            filename = os.path.join(model_id, WEIGHTS_NAME)
        else:
            try:
                filename = hf_hub_download(model_id, WEIGHTS_NAME)
            except:  # noqa
                raise ValueError(
                    f"Can't find weights for {model_id} in {model_id} or in the Hugging Face Hub. "
                    f"Please check that the file {WEIGHTS_NAME} is present at {model_id}."
                )

        adapters_weights = torch.load(
            filename, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        # load the weights into the model
        model = set_peft_model_state_dict(model, adapters_weights)
        if getattr(model, "hf_device_map", None) is not None:
            device_map = kwargs.get("device_map", "auto")
            max_memory = kwargs.get("max_memory", None)
            no_split_module_classes = model._no_split_modules
            if device_map != "sequential":
                max_memory = get_balanced_memory(
                    model,
                    max_memory=max_memory,
                    no_split_module_classes=no_split_module_classes,
                    low_zero=(device_map == "balanced_low_0"),
                )
            if isinstance(device_map, str):
                device_map = infer_auto_device_map(
                    model, max_memory=max_memory, no_split_module_classes=no_split_module_classes
                )
            model = dispatch_model(model, device_map=device_map)
            hook = AlignDevicesHook(io_same_device=True)
            if model.peft_config.peft_type == PeftType.LORA:
                add_hook_to_module(model.base_model.model, hook)

        return model

    def print_trainable_parameters(self):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in self.named_parameters():
            num_params = param.numel()
            # if using DS Zero 3 and the weights are initialized empty
            if num_params == 0 and hasattr(param, "ds_numel"):
                num_params = param.ds_numel

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
        )

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.base_model, name)

    def forward(self, *args, **kwargs):
        """
        Forward pass of the model.
        """
        return self.get_base_model()(*args, **kwargs)

    @contextmanager
    def disable_adapter(self):
        """
        Disables the adapter module.
        """
        self.base_model.disable_adapter_layers()
        yield
        self.base_model.enable_adapter_layers()

    def get_base_model(self):
        """
        Returns the base model.
        """
        return self.base_model.model


class PeftModelForSequenceClassification(PeftModel):
    """
    Peft model for sequence classification tasks, specifically tailored for TRAC.

    Note: This implementation is streamlined for TRAC. If you need to use other PEFT 
    methods such as LoRA, Prompt Tuning, Prefix Tuning, P-Tuning, or Adapters, please use the 
    official `peft` library from Hugging Face Hub.

    Args:
        model ([`PreTrainedModel`]): Base transformer model
        peft_config ([`PeftConfig`]): Peft config.

    **Attributes**:
        - **config** ([`PretrainedConfig`]) -- The configuration object of the base model.
        - **cls_layer_name** (`str`) -- The name of the classification layer.

    Example::

        >>> from transformers import AutoModelForSequenceClassification 
        >>> from peft_local_tensor import PeftModelForSequenceClassification, get_peft_config 
        >>> config = {
                'peft_type': 'LORA', 'task_type': 'SEQ_CLS', 'inference_mode': False, 'r': 8, 
                'lora_alpha': 32, 'lora_dropout': 0.1, 'target_modules': ['query', 'value']
            }
        >>> peft_config = get_peft_config(config) 
        >>> model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased") 
        >>> peft_model = PeftModelForSequenceClassification(model, peft_config) 
        >>> peft_model.print_trainable_parameters()
    """

    def __init__(self, model, peft_config: PeftConfig):
        super().__init__(model, peft_config)
        self.modules_to_save = ["classifier", "score"]

        for name, _ in self.base_model.named_children():
            if any(module_name in name for module_name in self.modules_to_save):
                self.cls_layer_name = name
                break

        # to make sure classifier layer is trainable
        _set_trainable(self)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        return self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )
    

class PeftModelForCausalLM(PeftModel):
    """
    Peft model for Causal LM, specifically tailored for TRAC.

    Note: This implementation is streamlined for TRAC. If you need to use other PEFT 
    methods such as LoRA, Prompt Tuning, Prefix Tuning, P-Tuning, or Adapters, please use the 
    official `peft` library from Hugging Face Hub.

    Args:
        model ([`PreTrainedModel`]): Base transformer model
        peft_config ([`PeftConfig`]): Peft config.

    Example::

        >>> from transformers import AutoModelForCausalLM 
        >>> from peft_local_tensor import PeftModelForCausalLM, get_peft_config
        >>> config = {
                'peft_type': 'LORA', 'task_type': 'CAUSAL_LM', 'inference_mode': False, 'r': 8, 
                'lora_alpha': 32, 'lora_dropout': 0.1, 'target_modules': ['c_attn']
            }
        >>> peft_config = get_peft_config(config) 
        >>> model = AutoModelForCausalLM.from_pretrained("gpt2-large") 
        >>> peft_model = PeftModelForCausalLM(model, peft_config) 
        >>> peft_model.print_trainable_parameters()
    """

    def __init__(self, model, peft_config: PeftConfig):
        super().__init__(model, peft_config)
        self.base_model_prepare_inputs_for_generation = self.base_model.prepare_inputs_for_generation

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        return self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

    def generate(self, **kwargs):
        self.base_model.prepare_inputs_for_generation = self.prepare_inputs_for_generation
        try:
            outputs = self.base_model.generate(**kwargs)
        except:
            self.base_model.prepare_inputs_for_generation = self.base_model_prepare_inputs_for_generation
            raise
        else:
            self.base_model.prepare_inputs_for_generation = self.base_model_prepare_inputs_for_generation
            return outputs

    def prepare_inputs_for_generation(self, *args, **kwargs):
        model_kwargs = self.base_model_prepare_inputs_for_generation(*args, **kwargs)

        return model_kwargs


class PeftModelForSeq2SeqLM(PeftModel):
    """
    Peft model for Seq2Seq LM, specifically tailored for TRAC.

    Note: This implementation is streamlined for TRAC. If you need to use other PEFT 
    methods such as LoRA, Prompt Tuning, Prefix Tuning, P-Tuning, or Adapters, please use the 
    official `peft` library from Hugging Face Hub.

    Args:
        model ([`PreTrainedModel`]): Base transformer model
        peft_config ([`PeftConfig`]): Peft config.

    Example::

        >>> from transformers import AutoModelForSeq2SeqLM 
        >>> from peft_local_tensor import PeftModelForSeq2SeqLM, get_peft_config
        >>> config = {
                'peft_type': 'LORA', 'task_type': 'SEQ_2_SEQ_LM', 'inference_mode': False, 'r': 8, 
                'target_modules': ['q', 'v'], 'lora_alpha': 32, 'lora_dropout': 0.1, 
                'merge_weights': False, 'fan_in_fan_out': False, 'bias': 'none'
            }
        >>> peft_config = get_peft_config(config) 
        >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base") 
        >>> peft_model = PeftModelForSeq2SeqLM(model, peft_config) 
        >>> peft_model.print_trainable_parameters()
    """

    def __init__(self, model, peft_config: PeftConfig):
        super().__init__(model, peft_config)
        self.base_model_prepare_inputs_for_generation = self.base_model.prepare_inputs_for_generation
        self.base_model_prepare_encoder_decoder_kwargs_for_generation = (
            self.base_model._prepare_encoder_decoder_kwargs_for_generation
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        decoder_inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        return self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            decoder_inputs_embeds=decoder_inputs_embeds,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

    def generate(self, **kwargs):
        self.base_model.prepare_inputs_for_generation = self.prepare_inputs_for_generation
        self.base_model._prepare_encoder_decoder_kwargs_for_generation = (
            self._prepare_encoder_decoder_kwargs_for_generation
        )
        try:
            outputs = self.base_model.generate(**kwargs)
        except:
            self.base_model.prepare_inputs_for_generation = self.base_model_prepare_inputs_for_generation
            self.base_model._prepare_encoder_decoder_kwargs_for_generation = (
                self.base_model_prepare_encoder_decoder_kwargs_for_generation
            )
            raise
        else:
            self.base_model.prepare_inputs_for_generation = self.base_model_prepare_inputs_for_generation
            self.base_model._prepare_encoder_decoder_kwargs_for_generation = (
                self.base_model_prepare_encoder_decoder_kwargs_for_generation
            )
            return outputs

    def prepare_inputs_for_generation(self, *args, **kwargs):
        model_kwargs = self.base_model_prepare_inputs_for_generation(*args, **kwargs)

        return model_kwargs


class PeftModelForTokenClassification(PeftModel):
    """
    Peft model for token classification tasks, specifically tailored for TRAC.

    Note: This implementation is streamlined for TRAC. If you need to use other PEFT 
    methods such as LoRA, Prompt Tuning, Prefix Tuning, P-Tuning, or Adapters, please use the 
    official `peft` library from Hugging Face Hub.

    Args:
        model ([`PreTrainedModel`]): Base transformer model
        peft_config ([`PeftConfig`]): Peft config.

    **Attributes**:
        - **config** ([`PretrainedConfig`]) -- The configuration object of the base model.
        - **cls_layer_name** (`str`) -- The name of the classification layer.

    Example::

        >>> from transformers import AutoModelForTokenClassification 
        >>> from peft_local_tensor import PeftModelForTokenClassification, get_peft_config 
        >>> config = {
                'peft_type': 'LORA', 'task_type': 'TOKEN_CLS', 'inference_mode': False, 'r': 8, 
                'lora_alpha': 32, 'lora_dropout': 0.1, 'target_modules': ['query', 'value']
            }
        >>> peft_config = get_peft_config(config) 
        >>> model = AutoModelForTokenClassification.from_pretrained("bert-base-cased") 
        >>> peft_model = PeftModelForTokenClassification(model, peft_config) 
        >>> peft_model.print_trainable_parameters()
    """

    def __init__(self, model, peft_config: PeftConfig):
        super().__init__(model, peft_config)
        self.modules_to_save = ["classifier", "score"]

        for name, _ in self.base_model.named_children():
            if any(module_name in name for module_name in self.modules_to_save):
                self.cls_layer_name = name
                break

        # to make sure classifier layer is trainable
        _set_trainable(self)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        return self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )