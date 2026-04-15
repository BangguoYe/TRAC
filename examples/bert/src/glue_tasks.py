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

from collections import OrderedDict
import datasets
import logging
from glue_processor import AbstractTask

logger = logging.getLogger(__name__)

main_dir = 'datasets/GLUE'

# GLUE
class COLA(AbstractTask):
    name = "cola"
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

    def load_dataset(self, split):
        return datasets.load_from_disk(f"{main_dir}/cola")[split]


class SST2(AbstractTask):
    name = "sst2"
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}




class MRPC(AbstractTask):
    name = "mrpc"
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}


class QQP(AbstractTask):
    name = "qqp"
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

class STSB(AbstractTask):
    name = "stsb"
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}



class MNLI(AbstractTask):
    name = "mnli"
    split_to_data_split = {"train": "train",
                           "validation": "validation_matched",
                           "test": "validation_matched"}

class MNLI_M(AbstractTask):
    name = "mnli"
    split_to_data_split = {"train": "train",
                           "validation": "validation_matched",
                           "test": "validation_matched"}

class MNLI_MM(AbstractTask):
    name = "mnli"
    split_to_data_split = {"train": "train",
                           "validation": "validation_mismatched",
                           "test": "validation_mismatched"}


class QNLI(AbstractTask):
    name = "qnli"
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}


#Tested
class RTE(AbstractTask):
    name = "rte"
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

class WNLI(AbstractTask):
    name = "wnli"
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}


TASK_MAPPING = OrderedDict(
    [
        ('mrpc', MRPC),
        ('cola', COLA),
        ('sst2', SST2),
        ('qnli', QNLI),
        ('rte', RTE),
        ('wnli', WNLI),
        ('mnli', MNLI),
        ('mnli-m', MNLI_M),
        ('mnli-mm', MNLI_MM),
        ('qqp', QQP),
        ('stsb', STSB),
    ]
)

class AutoTask:
    @classmethod
    def get(self, task, config, data_args, seed=42):
        print(f'--- ! The seed we use in the data loading process is : {seed}')
        if task in TASK_MAPPING:
            return TASK_MAPPING[task](config, data_args, seed)
        raise ValueError(
            "Unrecognized task {} for AutoTask Model: {}.\n"
            "Task name should be one of {}.".format(
                ", ".join(c for c in TASK_MAPPING.keys())
            )
        )

if __name__ == "__main__":
    for name in TASK_MAPPING:
        print(name)
        task = AutoTask().get(name, None, None)
        print(task.split_train_to_make_test)
        print(task.split_valid_to_make_test)
        train_set = task.get("train", split_validation_test=True)
        print(train_set[0])
