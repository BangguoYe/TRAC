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

import math
import torch
import torch.nn as nn
import numpy as np

from .tensor_cfg import(HIDDEN_SIZE_TO_TENSOR_SHAPE,
                        HIDDEN_SIZE_TO_TENSOR_RANK,
                        HIDDEN_SIZE_TO_TENSOR_RANK_A,
                        HIDDEN_SIZE_TO_TENSOR_RANK_B,
                        PARAM_NAME,
                        PARAM_STRUCTURE,
    )


# ==============================================================================
# PEFT CORE (TENSORIZED) MODULE
# Class definition for the core tensorized PEFT module.
# ==============================================================================

class ConfigClass():
    """A simple utility class to convert dictionary kwargs into object attributes."""
    def __init__(self,
                **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


# Core class: Implements the Tensor-Train decomposition in the TRAC method, 
# consisting of various types of tensor cores (trainable, frozen, shared).
class TensorTrain1D(nn.Module):
    """
    Represents an adaptation module as a sequence of Tensor-Train cores.
    Supports flexible configurations where specific cores can be fully trainable, 
    frozen (randomly initialized), or shared across multiple layers to maximize parameter efficiency.
    """
    def __init__(self, 
                config: ConfigClass,
                trainable_dim: list=None,
                random_dim: list=None,
                shared_trainable_tensor: dict=None,
                shared_random_tensor: dict=None,
                ):
        super(TensorTrain1D, self).__init__()

        self.config = config
        self.trainable_dim = trainable_dim
        self.random_dim = random_dim
        self.shared_trainable_dim = None
        self.shared_random_dim = None

        self.lora_matrix_role = config.lora_matrix_role
        self.hidden_size = config.hidden_size
        self.matrix_rank = config.matrix_rank
        
        self.tensor_shape = config.tensor_shape
        self.tensor_order = len(config.tensor_shape)
        tensor_shape_product = math.prod(self.tensor_shape)
        if self.hidden_size != tensor_shape_product:
            raise ValueError(f"Shape mismatch: self.hidden_size ({self.hidden_size}) does not match the product of tensor_shape ({tensor_shape_product})")

        if isinstance(config.tensor_ranks, int):
            self.tensor_ranks = [1] + [config.tensor_ranks] * (self.tensor_order - 1) + [self.matrix_rank]
        elif isinstance(config.tensor_ranks, list):
            self.tensor_ranks = config.tensor_ranks + [self.matrix_rank]
        else:
            raise TypeError(f"Expected 'self.tensor_ranks' to be an int or list, but got {type(self.tensor_ranks)}")
        
        self.scale_shared_tt_cores = config.scale_shared_tt_cores
        self.zero_init = config.zero_init
        self.tensor_init = config.tensor_init
        self.scale_init = config.scale_init
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
        if (random_dim is None) and (shared_trainable_tensor is None) and (shared_random_tensor is None):
            self.factors = []
            self.all_train = True
        else:
            self.all_train = False
            self.factors = [None] * self.tensor_order
            if shared_trainable_tensor is not None:
                self.shared_trainable_dim = list(shared_trainable_tensor.keys())
            if shared_random_tensor is not None:
                self.shared_random_dim = list(shared_random_tensor.keys())
                
        if self.scale_shared_tt_cores:
            self.tt_dim_scale_factors = nn.ParameterList()
            self.tt_rank_scale_factors = nn.ParameterList()
        else:
            self.tt_dim_scale_factors = None
            self.tt_rank_scale_factors = None
        
        self._build_factors(shared_trainable_tensor, shared_random_tensor)
        self._build_tt_dim_scale_factors()
        self._build_tt_rank_scale_factors()

    def get_factors(self):
        return self.factors, self.tt_dim_scale_factors, self.tt_rank_scale_factors
    
    def _build_factors(self,
                shared_trainable_tensor=None,
                shared_random_tensor=None
                ):
        if self.all_train:
            for i in range(0, self.tensor_order):
                r0 = self.tensor_ranks[i]
                n = self.tensor_shape[i]
                r1 = self.tensor_ranks[i+1]

                setattr(self, f'U{i}', self._build_trainable_tensor(r0, n, r1))
                self.factors.append(getattr(self, f'U{i}'))

            # Similar to the initialization of matrix B in LoRA: initialize the last tensor core 
            # to zero, ensuring the PEFT model is equivalent to the pre-trained model at initialization.
            if self.zero_init:
                nn.init.zeros_(self.factors[-1])
        else:
            if self.trainable_dim is not None:
                for idx in self.trainable_dim:
                    r0 = self.tensor_ranks[idx]
                    n = self.tensor_shape[idx]
                    r1 = self.tensor_ranks[idx+1]

                    setattr(self, f'U{idx}', self._build_trainable_tensor(r0, n, r1))
                    self.factors[idx] = getattr(self, f'U{idx}')
            
            if self.random_dim is not None:
                for idx in self.random_dim:
                    r0 = self.tensor_ranks[idx]
                    n = self.tensor_shape[idx]
                    r1 = self.tensor_ranks[idx+1]

                    self.factors[idx] = torch.randn(r0, n, r1).to(self.device) / math.sqrt(r1) * self.config.target_sdv ** (1 / self.tensor_order)

                    # Note: If the randomized parameters need to appear in the module list/named parameters 
                    # but do not participate in training, use the following code instead:
                    # self.factors[idx] = nn.Parameter(torch.randn(r0, n, r1) / math.sqrt(r1)\
                    #                                         * self.config.target_sdv ** (1 / self.tensor_order))
                    # self.factors[idx].requires_grad = False

            if self.shared_trainable_dim is not None:
                for idx in self.shared_trainable_dim:
                    self.factors[idx] = shared_trainable_tensor[idx]

            if self.shared_random_dim is not None:
                for idx in self.shared_random_dim:
                    self.factors[idx] = shared_random_tensor[idx]

            # Similar to the initialization of matrix B in LoRA: initialize the last tensor core 
            # to zero, ensuring the PEFT model is equivalent to the pre-trained model at initialization.
            if self.zero_init:
                if self.factors[-1].requires_grad:
                    nn.init.zeros_(self.factors[-1])
                else:
                    print("Warning: The last factor does not require gradients, not setting it to zero.")


    def _build_tt_dim_scale_factors(self):
        """Builds lightweight scaling factors for the physical dimensions (dim) of the shared TT cores."""
        # Return directly if TT cores are not shared or if shared trainable dimensions are not specified
        if not self.scale_shared_tt_cores or self.shared_trainable_dim is None:
            return

        for i in range(self.tensor_order):
            if i in self.shared_trainable_dim:
                n = self.tensor_shape[i]
                
                # Determine the initialization method based on self.scale_init
                if self.scale_init == 'ones':
                    dim_scale_factor = nn.Parameter(torch.ones(n))
                elif self.scale_init == 'zeros':
                    dim_scale_factor = nn.Parameter(torch.zeros(n))
                else:
                    raise ValueError(f"Unsupported scale_init: '{self.scale_init}'. Expected 'ones' or 'zeros'.")
            else:
                dim_scale_factor = None

            self.tt_dim_scale_factors.append(dim_scale_factor)


    def _build_tt_rank_scale_factors(self):
        """Builds lightweight scaling factors for the rank dimensions (rank) of the shared TT cores."""
        # Return directly if TT cores are not shared or if shared trainable dimensions are not specified
        if not self.scale_shared_tt_cores or self.shared_trainable_dim is None:
            return

        for i in range(self.tensor_order):
            if i in self.shared_trainable_dim:
                r0 = self.tensor_ranks[i]
                
                # Determine the initialization method based on self.scale_init
                if self.scale_init == 'ones':
                    rank_scale_factor = nn.Parameter(torch.ones(r0))
                elif self.scale_init == 'zeros':
                    rank_scale_factor = nn.Parameter(torch.zeros(r0))
                else:
                    raise ValueError(f"Unsupported scale_init: '{self.scale_init}'. Expected 'ones' or 'zeros'.")
            else:
                rank_scale_factor = None

            self.tt_rank_scale_factors.append(rank_scale_factor)

    # Specify the initialization method while building the tensor
    def _build_trainable_tensor(self, r0, n, r1):
        # Extract the common logic for calculating fan_in in Kaiming initialization based on the LoRA matrix role
        # Assuming self.lora_matrix_role is either 'A' or 'B'
        if self.tensor_init in ['KaimingNorm', 'KaimingUnif']:
            if self.lora_matrix_role == 'A':
                fan_in = self.hidden_size
            elif self.lora_matrix_role == 'B':
                fan_in = self.matrix_rank
            else:
                raise ValueError(f"Unknown lora_matrix_role: {self.lora_matrix_role}. Expected 'A' or 'B'.")
            
            kaiming_sigma = math.sqrt(2 / fan_in)

        # Initialize based on the selected tensor_init branch
        if self.tensor_init == 'TTNorm':
            U = nn.Parameter(torch.randn(r0, n, r1) / math.sqrt(r1) * self.config.target_sdv ** (1 / self.tensor_order))
        
        elif self.tensor_init == 'TTUnif': 
            U = nn.Parameter(torch.empty(r0, n, r1))
            sigma = 1 / math.sqrt(r1) * self.config.target_sdv ** (1 / self.tensor_order)
            nn.init.uniform_(U, a=-sigma, b=sigma)

        elif self.tensor_init == 'KaimingNorm':
            U = nn.Parameter(torch.randn(r0, n, r1) * kaiming_sigma)

        elif self.tensor_init == 'KaimingUnif':
            U = nn.Parameter(torch.empty(r0, n, r1))
            nn.init.uniform_(U, a=-kaiming_sigma, b=kaiming_sigma)

        else:
            # Default branch: fallback to TTNorm logic
            U = nn.Parameter(torch.randn(r0, n, r1) / math.sqrt(r1) * self.config.target_sdv ** (1 / self.tensor_order))
            print(f"Warning: Initialization method '{self.tensor_init}' is not recognized. "
                  f"Using Gaussian initialization (TTNorm logic) as default.")

        return U


# ==============================================================================
# SHARED TENSOR-TRAIN CORES
# Utilities to create the shared Tensor-Train (TT) cores.
# ==============================================================================

def build_tensor(r0, n, r1, target_sdv, tensor_order, lora_matrix_role, hidden_size, matrix_rank, tensor_init='TTNorm'):
    """Build and initialize a single tensor core."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Extract the common logic for calculating fan_in in Kaiming initialization based on the LoRA matrix role
    # Assuming lora_matrix_role is either 'A' or 'B'
    if tensor_init in ['KaimingNorm', 'KaimingUnif']:
        if lora_matrix_role == 'A':
            fan_in = hidden_size
        elif lora_matrix_role == 'B':
            fan_in = matrix_rank
        else:
            raise ValueError(f"Unknown lora_matrix_role: '{lora_matrix_role}'. Expected 'A' or 'B'.")
        
        kaiming_sigma = math.sqrt(2 / fan_in)

    # Initialize based on the selected tensor_init branch
    if tensor_init == 'TTNorm':
        U = nn.Parameter(torch.randn(r0, n, r1).to(device) / math.sqrt(r1) * target_sdv ** (1 / tensor_order))

    elif tensor_init == 'TTUnif': 
        U = nn.Parameter(torch.empty(r0, n, r1))
        sigma = 1 / math.sqrt(r1) * target_sdv ** (1 / tensor_order)
        nn.init.uniform_(U, a=-sigma, b=sigma)

    elif tensor_init == 'KaimingNorm':
        U = nn.Parameter(torch.randn(r0, n, r1) * kaiming_sigma)

    elif tensor_init == 'KaimingUnif':
        U = nn.Parameter(torch.empty(r0, n, r1))
        nn.init.uniform_(U, a=-kaiming_sigma, b=kaiming_sigma)

    else:
        # Default branch: fallback to TTNorm logic
        U = nn.Parameter(torch.randn(r0, n, r1) / math.sqrt(r1) * target_sdv ** (1 / tensor_order))
        print(f"Warning: Initialization method '{tensor_init}' is not recognized. "
              f"Using Gaussian initialization (TTNorm logic) as default.")

    return U


def create_shared_tensor(backbone_model, param_name, shared_dim, tensor_shape, tensor_ranks, target_sdv, tensor_order, is_trainable,\
                        zero_init, hidden_size, matrix_rank, tensor_init):
    """Creates and registers shared tensor cores across multiple adaptation layers."""

    shared_tensor = {}
    if shared_dim is not None:
        if is_trainable==True:
            for idx in shared_dim:
                r0 = tensor_ranks[idx]
                n = tensor_shape[idx]
                r1 = tensor_ranks[idx+1]
                
                if backbone_model is None:
                    print(f'trainable backbone_model is None')
                    shared_tensor[idx] = build_tensor(r0, n, r1, target_sdv, tensor_order, zero_init, hidden_size, matrix_rank, tensor_init)
                else:
                    backbone_model.register_parameter(f'{param_name}_{idx}', build_tensor(r0, n, r1, target_sdv, tensor_order, zero_init, hidden_size, matrix_rank, tensor_init))            
                    shared_tensor[idx] = getattr(backbone_model, f'{param_name}_{idx}')
        else:
            for idx in shared_dim:
                r0 = tensor_ranks[idx]
                n = tensor_shape[idx]
                r1 = tensor_ranks[idx+1]

                if backbone_model is None:
                    print(f'random backbone_model is None')
                    shared_tensor[idx] = build_tensor(r0, n, r1, target_sdv, tensor_order, zero_init, hidden_size, matrix_rank, tensor_init)
                else:
                    backbone_model.register_parameter(f'{param_name}_{idx}', build_tensor(r0, n, r1, target_sdv, tensor_order, zero_init, hidden_size, matrix_rank, tensor_init))        
                    shared_tensor[idx] = getattr(backbone_model, f'{param_name}_{idx}')
    else:
        shared_tensor = None
        
    return shared_tensor


def create_shared_params(peft_config, backbone_model, target_modules, hidden_size:int=768, mlp_hidden_dim:int=3072, matrix_rank:int=8):
    """Parses the configuration and constructs the shared parameters dictionary for target modules."""
    shared_params = {}
    
    param_name = [get_module_name(s) for s in target_modules]
    for name in param_name:
        if name in ['q', 'k', 'v', 'proj']:
            in_features = hidden_size
            out_features = hidden_size
        elif name in ['fc1']:
            in_features = hidden_size
            out_features = mlp_hidden_dim
        elif name in ['fc2']:
            in_features = mlp_hidden_dim
            out_features = hidden_size

        target_sdv = np.sqrt(1/(in_features+out_features))

        if peft_config.tensor_shape is not None:
            tensor_shape_A = peft_config.tensor_shape[in_features] + [matrix_rank]
            tensor_shape_B = peft_config.tensor_shape[out_features] + [matrix_rank]
        else:
            tensor_shape_A = HIDDEN_SIZE_TO_TENSOR_SHAPE[in_features] + [matrix_rank]
            tensor_shape_B = HIDDEN_SIZE_TO_TENSOR_SHAPE[out_features] + [matrix_rank]
        
        if peft_config.tensor_rank_A is not None:
            tensor_ranks_A = peft_config.tensor_rank_A[in_features] + [matrix_rank]
        else:
            tensor_ranks_A = HIDDEN_SIZE_TO_TENSOR_RANK_A[in_features] + [matrix_rank]

        if peft_config.tensor_rank_B is not None:
            tensor_ranks_B = peft_config.tensor_rank_B[out_features] + [matrix_rank]
        else:
            tensor_ranks_B = HIDDEN_SIZE_TO_TENSOR_RANK_B[out_features] + [matrix_rank]

        if peft_config.param_structure is not None:
            _param_structure = peft_config.param_structure
        else:
            _param_structure = PARAM_STRUCTURE

        tensor_order_A = len(tensor_shape_A)
        tensor_order_B = len(tensor_shape_B)

        shared_trainable_tensor_A = create_shared_tensor(backbone_model, f'shared_trainable_param_U_{name}_lora_A', _param_structure['shared_trainable_dim']['A'], tensor_shape_A, \
                                                         tensor_ranks_A, target_sdv, tensor_order_A, is_trainable=True, \
                                                        zero_init=False, hidden_size=in_features, matrix_rank=matrix_rank, tensor_init=peft_config.tensor_init)
        shared_trainable_tensor_B = create_shared_tensor(backbone_model, f'shared_trainable_param_U_{name}_lora_B', _param_structure['shared_trainable_dim']['B'], tensor_shape_B, \
                                                         tensor_ranks_B, target_sdv, tensor_order_B, is_trainable=True, \
                                                        zero_init=True, hidden_size=in_features, matrix_rank=matrix_rank, tensor_init=peft_config.tensor_init)
        shared_random_tensor_A = create_shared_tensor(backbone_model, f'shared_random_param_U_{name}_lora_A', _param_structure['shared_random_dim']['A'], tensor_shape_A, \
                                                      tensor_ranks_A, target_sdv, tensor_order_A, is_trainable=False, \
                                                        zero_init=False, hidden_size=in_features, matrix_rank=matrix_rank, tensor_init=peft_config.tensor_init)
        shared_random_tensor_B = create_shared_tensor(backbone_model, f'shared_random_param_U_{name}_lora_B', _param_structure['shared_random_dim']['B'], tensor_shape_B, \
                                                      tensor_ranks_B, target_sdv, tensor_order_B, is_trainable=False, \
                                                        zero_init=True, hidden_size=in_features, matrix_rank=matrix_rank, tensor_init=peft_config.tensor_init)

        shared_params[name] = {'A': {'trainable': shared_trainable_tensor_A, 'random': shared_random_tensor_A},
                            'B': {'trainable': shared_trainable_tensor_B, 'random': shared_random_tensor_B}}
        
    return shared_params


# ==============================================================================
# PARAMETER NAME MAPPING
# Utilities for handling parameter-name remapping and key translation in models.
# ==============================================================================

# Standardizes the naming of matrices in the transformer architecture, as different models use different conventions.
# The current implementation is primarily designed for BERT, ViT, LLaMA, etc.
# For other architectures, compatibility can be achieved by extending this function based on their parameter names.
def get_module_name(target_name):
    """Maps specific layer parameter names to a unified internal representation (e.g., 'q', 'k', 'v', 'proj')."""
    if ('q' in target_name) or ('q_proj' in target_name) or ('query' in target_name):
        module_name = 'q'
    elif ('k' in target_name) or ('k_proj' in target_name) or ('key' in target_name):
        module_name = 'k'
    elif ('v' in target_name) or ('v_proj' in target_name) or ('value' in target_name):
        module_name = 'v'
    elif ('proj' in target_name) or ('o_proj' in target_name) or ('attention.output.dense' in target_name):
        module_name = 'proj'
    elif ('fc1' in target_name) or ('up_proj' in target_name) or ('intermediate.dense' in target_name):
        module_name = 'fc1'
    elif ('fc2' in target_name) or ('down_proj' in target_name) or ('output.dense' in target_name and 'attention' not in target_name):
        module_name = 'fc2'
    
    return module_name
