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
import time
import logging

import numpy as np
import torch


def mkdirs(path: str):
    if not os.path.exists(path):
        os.makedirs(path)
    return


def acc_score(gt, pred):
    correct = (gt == pred).astype(float)
    acc = correct.sum() / len(gt)
    return acc


def get_one_hot(label, num_cls):
    label = label.reshape(-1)
    label = torch.eye(num_cls, device=label.device)[label]
    return label


class ResultCLS:
    def __init__(self, num_cls) -> None:
        self.epoch = 1
        self.best_epoch = 0
        self.best_val_result = 0.0
        self.test_acc = 0.0
        self.test_auc = 0.0
        self.test_f1 = 0.0
        self.test_sen = 0.0
        self.test_spe = 0.0
        self.test_pre = 0.0
        self.num_cls = num_cls

        return

    def eval(self, label, pred):
        self.pred.append(pred)
        self.true.append(label)
        return

    def init(self):
        self.st = time.time()
        self.pred = []
        self.true = []
        return

    @torch.no_grad()
    def stastic(self):
        num_cls = self.num_cls

        pred = torch.cat(self.pred, dim=0)
        true = torch.cat(self.true, dim=0)

        true = true.cpu().detach().numpy()
        pred = torch.argmax(pred, dim=1).cpu().detach().numpy()

        self.acc = acc_score(true, pred)
        self.time = np.round(time.time() - self.st, 1)

        self.pars = [self.acc]
        return

    def print(self, epoch: int, datatype='test'):
        self.stastic()
        titles = ["dataset", "ACC"]
        items = [datatype.upper()] + self.pars
        forma_1 = "\n|{:^8}" + "|{:^5}" * (len(titles) - 1) + "|"
        forma_2 = "\n|{:^8}" + "|{:^.3f}" * (len(titles) - 1) + "|"
        logging.info(f"ACC: {self.pars[0]:.3f}, TIME: {self.time:.1f}s")
        logging.info((forma_1 + forma_2).format(*titles, *items))
        self.epoch = epoch

        if datatype == 'val' and self.acc > self.best_val_result:
            self.best_epoch = epoch
            self.best_val_result = self.acc
        return