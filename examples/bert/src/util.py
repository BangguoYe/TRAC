# This file contains code derived from the SoRA project:
#   https://github.com/TsinghuaC3I/SoRA
#
# Copyright (c) Tsinghua C3I
# Copyright (c) 2026 Bangguo Ye, Yuanwei Zhang, Xiaoqun Zhang
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
from transformers.trainer_pt_utils import get_parameter_names
import torch.nn as nn
from transformers import AdamW, get_linear_schedule_with_warmup


def create_optimizer_and_scheduler(args, model, num_training_steps: int):
    """
    Setup the optimizer and the learning rate scheduler.

    We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
    Trainer's init through :obj:`optimizers`, or subclass and override this method (or :obj:`create_optimizer`
    and/or :obj:`create_scheduler`) in a subclass.
    """
    optimizer = create_optimizer(args, model)
    scheduler = create_scheduler(args, num_training_steps=num_training_steps, optimizer=optimizer)
    return optimizer, scheduler

def create_optimizer(args, model):
    """
    Setup the optimizer.

    We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
    Trainer's init through :obj:`optimizers`, or subclass and override this method in a subclass.
    """

    decay_parameters = get_parameter_names(model, [nn.LayerNorm])
    decay_parameters = [name for name in decay_parameters if "bias" not in name]

    optimizer_grouped_parameters = [
    # 1. Parameters containing 'tt_dim_scale_factors' or 'tt_rank_scale_factors', with weight decay
    {
        "params": [
            p for n, p in model.named_parameters()
            if (
                (("tt_dim_scale_factors" in n) or ("tt_rank_scale_factors" in n))
                and (n in decay_parameters)
                and p.requires_grad
            )
        ],
        "weight_decay": args.weight_decay,
        "lr": args.scale_factors_learning_rate,
    },
    # 2. Parameters containing 'tt_dim_scale_factors' or 'tt_rank_scale_factors', without weight decay
    {
        "params": [
            p for n, p in model.named_parameters()
            if (
                (("tt_dim_scale_factors" in n) or ("tt_rank_scale_factors" in n))
                and (n not in decay_parameters)
                and p.requires_grad
            )
        ],
        "weight_decay": 0.0,
        "lr": args.scale_factors_learning_rate,
    },
    # 3. Other parameters, with weight decay
    {
        "params": [
            p for n, p in model.named_parameters()
            if (
                (("tt_dim_scale_factors" not in n) and ("tt_rank_scale_factors" not in n))
                and (n in decay_parameters)
                and p.requires_grad
            )
        ],
        "weight_decay": args.weight_decay,
        "lr": args.learning_rate,
    },
    # 4. Other parameters, without weight decay
    {
        "params": [
            p for n, p in model.named_parameters()
            if (
                (("tt_dim_scale_factors" not in n) and ("tt_rank_scale_factors" not in n))
                and (n not in decay_parameters)
                and p.requires_grad
            )
        ],
        "weight_decay": 0.0,
        "lr": args.learning_rate,
    },
    ]

    optimizer_kwargs = {
        "betas": (args.adam_beta1, args.adam_beta2),
        "eps": args.adam_epsilon,
    }
    optimizer_kwargs["lr"] = args.learning_rate
    optimizer = AdamW(optimizer_grouped_parameters, **optimizer_kwargs)

    return optimizer

def create_scheduler(args, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
    """
    Setup the scheduler. The optimizer of the trainer must have been set up either before this method is called or
    passed as an argument.

    Args:
        num_training_steps (int): The number of training steps to do.
    """
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.get_warmup_steps(num_training_steps),
            num_training_steps=num_training_steps)
    return lr_scheduler