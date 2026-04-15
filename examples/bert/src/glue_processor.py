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

import abc
from typing import Mapping
import datasets
import logging
import torch
logger = logging.getLogger(__name__)

main_dir = 'datasets/GLUE'

class AbstractTask(abc.ABC):
    name = NotImplemented
    config = NotImplemented
    prefix = NotImplemented
    split_map = None
    split_to_data_split: Mapping[str, str] = \
        {"train": "train", "validation": "validation", "test": "test"}
    small_datasets_without_all_splits = ["cola", "wnli", "rte", "superglue-cb", "superglue-copa", "superglue-multirc",
                                         "superglue-wic", "superglue-wsc.fixed", "superglue-rte", "mrpc", "stsb",
                                         "superglue-boolq", "mnli"] 
    large_data_without_all_splits = ["qqp", "qnli", "superglue-record", "sst2"]

    split_valid_to_make_test = True
    split_train_to_make_test = False
    keep_fields_after_preprocess = ["label"]  # The fields that should be kept even after preprocessiing

    def __init__(self, config, data_args, seed=42, default_max_length=1):
        self.config = config
        self.seed = seed
        self.data_args = data_args

        self.default_max_length = default_max_length
        self.__post_init__()
    
    def __post_init__(self):
        self.split_valid_to_make_test = self.name in self.small_datasets_without_all_splits
        self.split_train_to_make_test = self.name in self.large_data_without_all_splits
    
    def load_dataset(self, split):
        tmp = datasets.load_from_disk(f"{main_dir}/{self.name}")

        return tmp[split]

    def check_n_obs(self, n_obs, total_size):
        if n_obs is not None and n_obs > total_size:
            n_obs = total_size
            logger.warning("n_obs is set to %s", n_obs)
        return n_obs

    def shuffled_indices(self, dataset):
        num_samples = len(dataset)
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        return torch.randperm(num_samples, generator=generator).tolist()

    def subsample(self, dataset, n_obs=None, indices=None):
        """
        Given a dataset returns the subsampled dataset.
        :param n_obs: the number of samples of the subsampled dataset.
        :param indices: indices to select the samples from, if not given, indices are computed
        from by shuffling the given dataset.
        :return: subsampled dataset.
        """
        num_samples = len(dataset)
        n_obs = self.check_n_obs(n_obs, num_samples)
        if indices is None:
           indices = self.shuffled_indices(dataset)
        indices = indices[:n_obs]
        return dataset.select(indices)


    def get_split_indices(self, split, dataset, validation_size):
        indices = self.shuffled_indices(dataset)
        if split == "validation":
            return indices[:validation_size]
        else:
            return indices[validation_size:]

    def preprocessor(self, example):
        return example

    def get(self, split, n_obs=None, split_validation_test=False):
        # For small datasets (n_samples < 10K) without test set, we divide validation set to
        # half, use one half as test set and one half as validation set.
        if split in ["eval", "dev", "valid"]:
            split = "validation"
        if split_validation_test and self.split_valid_to_make_test \
                and split != "train":
            mapped_split = self.split_to_data_split["validation"]
            dataset = self.load_dataset(split=mapped_split)
            indices = self.get_split_indices(split, dataset, validation_size=len(dataset)//2)
            dataset = self.subsample(dataset, n_obs, indices)
        # For larger datasets (n_samples > 10K), we divide training set into 1K as
        # validation and the rest as training set, keeping the original validation
        # set as the test set.
        elif split_validation_test and self.split_train_to_make_test \
                and split != "test":
            dataset = self.load_dataset(split="train")
            indices = self.get_split_indices(split, dataset, validation_size=1000)
            dataset = self.subsample(dataset, n_obs, indices)
        else:
            mapped_split = self.split_to_data_split[split]
            dataset = self.load_dataset(split=mapped_split)
            # shuffles the data and samples it.
            if n_obs is not None:
                dataset = self.subsample(dataset, n_obs)

        this_method = getattr(self.__class__, 'preprocessor')
        base_method = getattr(AbstractTask, 'preprocessor')
        if this_method is not base_method:
            return dataset.map(self.preprocessor)
        else:
            return dataset