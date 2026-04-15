# This file contains code derived from the NOLA project:
#   https://github.com/UCDvision/NOLA
#
# Copyright (c) 2023 UCDvision
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

import os
import random
from collections import defaultdict

from PIL import ImageFilter
import torch
from torchvision import datasets, transforms


# Extended version of ImageFolder to return index of image too.
class ImageFolderEx(datasets.ImageFolder):
    def __init__(self, root, *args, **kwargs):
        super(ImageFolderEx, self).__init__(root, *args, **kwargs)

    def __kshot__(self, k, seed):
        """Convert dataset to contain k-samples per class. Randomly\
        subsample images per class to obtain the k samples."""
        data_dict = defaultdict(list)
        k_samples = []
        targets = []
        for sample in self.samples:
            data_dict[sample[1]].append(sample[0])
        for i in range(len(self.classes)):
            # Select k-samples per class
            gen = torch.random.manual_seed(seed=seed)
            chosen = torch.randperm(len(data_dict[i]), generator=gen)[:k]
            for j in chosen:
                k_samples.append([data_dict[i][j], i])
                targets.append(i)
        self.samples = k_samples
        self.imgs = k_samples
        self.targets = targets

    def __getitem__(self, index):
        sample, target = super(ImageFolderEx, self).__getitem__(index)
        return sample, target


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def load_dataset(cfg):
    traindir = os.path.join(cfg.train_data_path, 'train')
    valdir = os.path.join(cfg.val_data_path, 'val')
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean=mean, std=std)

    augmentation_train = [
        transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]

    augmentation_val = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]

    train_dataset = ImageFolderEx(traindir, transforms.Compose(augmentation_train))
    if cfg.kshot > 0:
        train_dataset.__kshot__(cfg.kshot, cfg.kshot_seed)
    print(train_dataset)
    val_dataset = ImageFolderEx(valdir, transforms.Compose(augmentation_val))
    test_dataset = ImageFolderEx(valdir, transforms.Compose(augmentation_val))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=256, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
    )
    return train_loader, val_loader, val_loader


def load_dataset_no_val(cfg):
    if not cfg.val_data_path:
        traindir = cfg.train_data_path
        valdir = None
    else:
        traindir = os.path.join(cfg.train_data_path, 'train')
        valdir = os.path.join(cfg.val_data_path, 'val') if cfg.val_data_path else None

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean=mean, std=std)

    augmentation_train = [
        transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]

    augmentation_val = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]

    train_dataset = ImageFolderEx(traindir, transforms.Compose(augmentation_train))

    if cfg.kshot > 0:
        train_samples = defaultdict(list)
        for sample in train_dataset.samples:
            train_samples[sample[1]].append(sample[0])

        kshot_samples = []
        val_samples = []
        val_targets = []

        for class_idx, samples in train_samples.items():
            gen = torch.random.manual_seed(seed=cfg.kshot_seed)
            perm = torch.randperm(len(samples), generator=gen)
            
            train_idxs = perm[:cfg.kshot]
            val_idxs = perm[cfg.kshot:]

            for idx in train_idxs:
                kshot_samples.append([samples[idx], class_idx])
            for idx in val_idxs:
                val_samples.append([samples[idx], class_idx])
                val_targets.append(class_idx)

        train_dataset.samples = kshot_samples
        train_dataset.imgs = kshot_samples
        train_dataset.targets = [sample[1] for sample in kshot_samples]

        if not valdir:
            val_dataset = ImageFolderEx(traindir, transforms.Compose(augmentation_val))
            val_dataset.samples = val_samples
            val_dataset.imgs = val_samples
            val_dataset.targets = val_targets
        else:
            val_dataset = ImageFolderEx(valdir, transforms.Compose(augmentation_val))
    else:
        val_dataset = ImageFolderEx(valdir, transforms.Compose(augmentation_val)) if valdir else None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=256, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True) if val_dataset else None

    return train_loader, val_loader, val_loader
