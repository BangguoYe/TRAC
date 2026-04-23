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

import warnings

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

from .tensor_rep import ConfigClass, TensorTrain1D
from .tensor_cfg import(HIDDEN_SIZE_TO_TENSOR_SHAPE,
                        HIDDEN_SIZE_TO_TENSOR_RANK_A,
                        HIDDEN_SIZE_TO_TENSOR_RANK_B,
                        PARAM_STRUCTURE,
    )

from .utils import transpose


def apply_activation(x: torch.Tensor, activation: str = 'none', scalar=1.) -> torch.Tensor:
    """
    Applies the specified activation function to the input tensor.

    Args:
        x (torch.Tensor): The input tensor.
        activation (str): The type of activation function (case-insensitive).
            Supported options include: 'none', 'linear', 'relu', 'sigmoid', 
            'tanh', 'softmax', etc.
        scalar (float): A scaling factor used primarily for the softmax function.

    Returns:
        torch.Tensor: The activated tensor.
    """
    act = activation.lower()

    if act in ['none', 'linear']:
        return x
    elif act == 'relu':
        return F.relu(x)
    elif act == 'sigmoid':
        return torch.sigmoid(x)
    elif act == 'tanh':
        return torch.tanh(x)
    elif act == 'softmax':
        # Softmax usually requires specifying a dimension; modify as needed for specific tasks
        return scalar * F.softmax(x, dim=0)
    else:
        raise ValueError(f"Unsupported activation function: {activation}")
    


# Below code is based on https://github.com/microsoft/LoRA/blob/main/loralib/layers.py and modified to work with our methods

#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

class LoraLayer:
    """Base class for LoRA-like layers, managing rank, dropout, and weight merging state."""
    def __init__(
        self,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights
        self.disable_adapters = False


class TracLayer(nn.Module):
    """Base class for TRAC-like layers."""
    def __init__(
        self,
        peft_config,
        in_features: int,
        out_features: int,
        r: int = 0,
        module_name: str = None,
        shared_params: dict = None,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        merge_weights: bool = False,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = self.lora_alpha / self.r

        # Optional dropout
        self.lora_dropout = lora_dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x

        # Actual trainable parameters
        if r > 0:
            self.target_sdv = np.sqrt(1/(self.in_features+self.out_features))

            _param_structure = peft_config.param_structure if peft_config.param_structure is not None else PARAM_STRUCTURE

            tensor_config_A = ConfigClass(
                        lora_matrix_role="A",
                        hidden_size=self.in_features,
                        matrix_rank=r,
                        tensor_shape=peft_config.tensor_shape[self.in_features] if peft_config.tensor_shape is not None else HIDDEN_SIZE_TO_TENSOR_SHAPE[self.in_features],
                        tensor_ranks=peft_config.tensor_rank_A[self.in_features] if peft_config.tensor_rank_A is not None else HIDDEN_SIZE_TO_TENSOR_RANK_A[self.in_features],
                        scale_shared_tt_cores=peft_config.scale_shared_tt_cores,
                        zero_init=False,
                        tensor_init=peft_config.tensor_init,
                        scale_init=peft_config.scale_init,
                        target_sdv=self.target_sdv)
            
            self.A = TensorizedLinearModule(tensor_config_A, _param_structure['trainable_dim']['A'], _param_structure['random_dim']['A'],\
                                             shared_params[module_name]['A']['trainable'], shared_params[module_name]['A']['random'],\
                                             is_input_side=True, vector_activation=peft_config.vector_activation, use_fast_tt=peft_config.use_fast_tt)

            tensor_config_B = ConfigClass(
                        lora_matrix_role="B",
                        hidden_size=self.out_features,
                        matrix_rank=r,
                        tensor_shape=peft_config.tensor_shape[self.out_features] if peft_config.tensor_shape is not None else HIDDEN_SIZE_TO_TENSOR_SHAPE[self.out_features],
                        tensor_ranks=peft_config.tensor_rank_B[self.out_features] if peft_config.tensor_rank_B is not None else HIDDEN_SIZE_TO_TENSOR_RANK_B[self.out_features],
                        scale_shared_tt_cores=peft_config.scale_shared_tt_cores,
                        zero_init=True,
                        tensor_init=peft_config.tensor_init,
                        scale_init=peft_config.scale_init,
                        target_sdv=self.target_sdv)
            
            self.B = TensorizedLinearModule(tensor_config_B, _param_structure['trainable_dim']['B'], _param_structure['random_dim']['B'],\
                                             shared_params[module_name]['B']['trainable'], shared_params[module_name]['B']['random'],\
                                             vector_activation=peft_config.vector_activation, use_fast_tt=peft_config.use_fast_tt)

    def forward(self, x):
        return self.B(self.A(self.lora_dropout(x))) * self.scaling


class Linear(nn.Linear, LoraLayer):
    """
    TRAC-adapted Linear layer that replaces standard nn.Linear.
    Injects tensorized low-rank adaptation matrices (A and B) into the forward pass.
    """
    def __init__(
        self,
        peft_config,
        in_features: int,
        out_features: int,
        r: int = 0,
        module_name: str = None,
        shared_params: dict = None,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = False,
        **kwargs,
    ):
        
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoraLayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)
        
        self.fan_in_fan_out = fan_in_fan_out

        # Actual trainable parameters
        if r > 0:
            self.target_sdv = np.sqrt(1/(in_features + out_features))

            _param_structure = peft_config.param_structure if peft_config.param_structure is not None else PARAM_STRUCTURE

            tensor_config_A = ConfigClass(
                        lora_matrix_role="A",
                        hidden_size=in_features,
                        matrix_rank=r,
                        tensor_shape=peft_config.tensor_shape[in_features] if peft_config.tensor_shape is not None else HIDDEN_SIZE_TO_TENSOR_SHAPE[in_features],
                        tensor_ranks=peft_config.tensor_rank_A[in_features] if peft_config.tensor_rank_A is not None else HIDDEN_SIZE_TO_TENSOR_RANK_A[in_features],
                        scale_shared_tt_cores=peft_config.scale_shared_tt_cores,
                        zero_init=False,
                        tensor_init=peft_config.tensor_init,
                        scale_init=peft_config.scale_init,
                        target_sdv=self.target_sdv)
            
            self.lora_A = TensorizedLinearModule(tensor_config_A, _param_structure['trainable_dim']['A'], _param_structure['random_dim']['A'],\
                                             shared_params[module_name]['A']['trainable'], shared_params[module_name]['A']['random'], \
                                             is_input_side=True, vector_activation=peft_config.vector_activation, use_fast_tt=peft_config.use_fast_tt)

            tensor_config_B = ConfigClass(
                        lora_matrix_role="B",
                        hidden_size=out_features,
                        matrix_rank=r,
                        tensor_shape=peft_config.tensor_shape[out_features] if peft_config.tensor_shape is not None else HIDDEN_SIZE_TO_TENSOR_SHAPE[out_features],
                        tensor_ranks=peft_config.tensor_rank_B[out_features] if peft_config.tensor_rank_B is not None else HIDDEN_SIZE_TO_TENSOR_RANK_B[out_features],
                        scale_shared_tt_cores=peft_config.scale_shared_tt_cores,
                        zero_init=True,
                        tensor_init=peft_config.tensor_init,
                        scale_init=peft_config.scale_init,
                        target_sdv=self.target_sdv)
            
            self.lora_B = TensorizedLinearModule(tensor_config_B, _param_structure['trainable_dim']['B'], _param_structure['random_dim']['B'],\
                                             shared_params[module_name]['B']['trainable'], shared_params[module_name]['B']['random'],
                                             vector_activation=peft_config.vector_activation, use_fast_tt=peft_config.use_fast_tt)

            self.scaling = self.lora_alpha / self.r

            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False

        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)

    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)
        self.lora_A.train(mode)
        self.lora_B.train(mode)
        if not mode and self.merge_weights and not self.merged:
            # Merge the weights and mark it
            if self.r > 0:
                self.weight.data += (
                    transpose(self.lora_B.weight @ self.lora_A.weight, self.fan_in_fan_out) * self.scaling
                )
            self.merged = True
        elif self.merge_weights and self.merged:
            # Make sure that the weights are not merged
            if self.r > 0:
                self.weight.data -= (
                    transpose(self.lora_B.weight @ self.lora_A.weight, self.fan_in_fan_out) * self.scaling
                )
            self.merged = False

    def eval(self):
        nn.Linear.eval(self)
        self.lora_A.eval()
        self.lora_B.eval()

    def forward(self, x: torch.Tensor):
        previous_dtype = self.weight.dtype
            
        if x.dtype != previous_dtype:
            print(f'--- previous_dtype = self.weight.dtype: {self.weight.dtype} --- x.dtype: {x.dtype}')
            x.to(previous_dtype)

        if self.disable_adapters:
            if self.r > 0 and self.merged:
                matmul_output = self.lora_B.weight @ self.lora_A.weight
                self.weight.data -= transpose(matmul_output.to(previous_dtype), self.fan_in_fan_out) * self.scaling
                self.merged = False

            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        elif self.r > 0 and not self.merged:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias.to(previous_dtype))
            expected_dtype = result.dtype
            if self.r > 0:
                with autocast():
                    # print(f'shape of a {self.lora_A(self.lora_dropout(x)).shape}')
                    result += self.lora_B(self.lora_A(self.lora_dropout(x))).to(expected_dtype) * self.scaling
        else:
             result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

        if result.dtype != previous_dtype:
            result = result.to(previous_dtype)

        return result


class TensorizedLinearModule(nn.Module):
    """
    Wraps the TensorTrain1D module to function as a linear transformation matrix (A or B).
    Handles the forward pass, including tensor contraction and optional fast TT algorithms.
    """
    def __init__(self,
                config,
                trainable_dim: list=None,
                random_dim: list=None,
                shared_trainable_tensor: dict=None,
                shared_random_tensor: dict=None,
                is_input_side=False,
                vector_activation: str='none',
                use_fast_tt=False,
    ):
        super(TensorizedLinearModule,self).__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.tensor_shape = config.tensor_shape
        self.is_input_side = is_input_side
        self.vector_activation = vector_activation
        self.use_fast_tt = use_fast_tt
        self.tt_1d = TensorTrain1D(
                config=config,
                trainable_dim=trainable_dim,
                random_dim=random_dim,
                shared_trainable_tensor=shared_trainable_tensor,
                shared_random_tensor=shared_random_tensor,
                )

    def forward(self, x):

        if not self.use_fast_tt:
            out = self.tt_1d.factors[0]
            if (self.tt_1d.tt_dim_scale_factors is None) and (self.tt_1d.tt_rank_scale_factors is None):
                for i in range(1, len(self.tt_1d.factors)):
                    out = torch.tensordot(out, self.tt_1d.factors[i], [[-1], [0]])

            elif (self.tt_1d.tt_dim_scale_factors is not None) or (self.tt_1d.tt_rank_scale_factors is not None):
                for i in range(1, len(self.tt_1d.factors)):
                    if i in self.tt_1d.shared_trainable_dim:
                        rank_act = apply_activation(self.tt_1d.tt_rank_scale_factors[i], self.vector_activation, len(self.tt_1d.tt_rank_scale_factors[i]))
                        dim_act = apply_activation(self.tt_1d.tt_dim_scale_factors[i], self.vector_activation, len(self.tt_1d.tt_dim_scale_factors[i]))
                        
                        scaled_factor = rank_act.view(-1, 1, 1) * self.tt_1d.factors[i]
                        scaled_factor = dim_act.view(1, -1, 1) * scaled_factor
                        
                        out = torch.tensordot(out, scaled_factor, [[-1], [0]])
                    else:
                        out = torch.tensordot(out, self.tt_1d.factors[i], [[-1], [0]])

            else:
                warnings.warn("No preset tensor calculation method was used, so the output is the same as the input (considered as identity mapping).")
                return x
            
            if self.is_input_side == True:
                output = x @ out.reshape(self.config.hidden_size, self.config.matrix_rank)
            else:
                output = x @ (out.reshape(self.config.hidden_size, self.config.matrix_rank)).t()
        
        else:
            # The current fast tensor routine is primarily compatible with the specific tensor structure
            # implied by the three TT (Tensor-Train) cores. To extend it to higher-order tensors or other
            # tensor decomposition formats, you will need to adjust the summation pattern in the `torch.einsum` expressions below.
            current_factors = [factor for factor in self.tt_1d.factors]
            
            if self.is_input_side == True:
                bsz = x.shape[0]
                length = x.shape[1]
                x_tensor = x.reshape([bsz, length] + self.tensor_shape)

                for i in range(1, len(self.tt_1d.factors)):
                    if i in self.tt_1d.shared_trainable_dim:
                        rank_act = apply_activation(self.tt_1d.tt_rank_scale_factors[i], self.vector_activation, len(self.tt_1d.tt_rank_scale_factors[i]))
                        dim_act = apply_activation(self.tt_1d.tt_dim_scale_factors[i], self.vector_activation, len(self.tt_1d.tt_dim_scale_factors[i]))
                        
                        temp_factor = rank_act.view(-1, 1, 1) * self.tt_1d.factors[i]
                        current_factors[i] = dim_act.view(1, -1, 1) * temp_factor

                output = torch.einsum('abc, cde, efg, hibdf->hiag', 
                                    current_factors[0], 
                                    current_factors[1], 
                                    current_factors[2], 
                                    x_tensor)
                output = output.squeeze(2)
                
            else:
                bsz = x.shape[0]
                length = x.shape[1]
                
                for i in range(1, len(self.tt_1d.factors)):
                    if i in self.tt_1d.shared_trainable_dim:
                        rank_act = apply_activation(self.tt_1d.tt_rank_scale_factors[i], self.vector_activation, len(self.tt_1d.tt_rank_scale_factors[i]))
                        dim_act = apply_activation(self.tt_1d.tt_dim_scale_factors[i], self.vector_activation, len(self.tt_1d.tt_dim_scale_factors[i]))
                        
                        temp_factor = rank_act.view(-1, 1, 1) * self.tt_1d.factors[i]
                        current_factors[i] = dim_act.view(1, -1, 1) * temp_factor

                output = torch.einsum('abc, cde, efg, hig->hiabdf', 
                                    current_factors[0], 
                                    current_factors[1], 
                                    current_factors[2], 
                                    x)
                output = output.squeeze(2)
                output = output.reshape([bsz, length, self.hidden_size])

        return output
