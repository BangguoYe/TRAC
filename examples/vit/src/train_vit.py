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

import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp.autocast_mode import autocast
from torch.cuda.amp.grad_scaler import GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
import csv
import sys
import random
import numpy as np
from transformers import AutoConfig, AutoModelForImageClassification


from dataloader import load_dataset, load_dataset_no_val
from utils.result import ResultCLS
from utils.utils import init

from trac import compute_trainable_param


model_to_path={'base': 'models/vit/vit-base-patch16-224-in21k-hf',
               'large': 'models/vit/vit-large-patch16-224-in21k-hf'
                }

dataset_to_class_n={
                    'CIFAR10': 10, 
                    'CIFAR100': 100,
                    'CUB_200_2011': 200,
                    'flowers102': 102,
                    }

dataset_to_path={
                    'CIFAR10': {'train_data_path': 'datasets/CIFAR10/ImageFolder', 'val_data_path': 'datasets/CIFAR10/ImageFolder'}, 
                    'CIFAR100': {'train_data_path': 'datasets/CIFAR100/ImageFolder', 'val_data_path': 'datasets/CIFAR100/ImageFolder'},
                    'CUB_200_2011': {'train_data_path': 'datasets/CUB_200_2011/CUB_200_2011/images', 'val_data_path': None},
                    'flowers102': {'train_data_path': 'datasets/Flowers102', 'val_data_path': 'datasets/Flowers102'},
                    }

def save2file(acc, loss, best_acc, best_ep, exp, fname):
    """Save(append) accuracy values to a csv file.

    """
    with open(fname, 'a') as f:
        csvwriter = csv.writer(f)
        csvwriter.writerow([exp, '{:.2f}'.format(acc), '{:.3f}'.format(loss), 'best: {:.1f}'.format(best_acc), best_ep])


def set_classifier_to_identity(model):
    """Change the number of classes in linear classifier equal to embed size and set the
    linear classifier to identity matrix.

    Alternative approach to obtain the pre_logits output (i.e., input to classifier head).
    Set linear layer weight to identity and bias to zero.
    """
    embed_dim = model.module.embed_dim
    model.module.reset_classifier(num_classes=embed_dim)
    model.module.head.weight.data = torch.eye(embed_dim).cuda()
    model.module.head.bias.data = torch.zeros(embed_dim).cuda()


def train(epoch, trainset):
    running_loss = 0.0
    this_lr = scheduler.get_last_lr()[0]
    net.train()
    # Save backbone features for k-NN evaluation
    save_feats = False
    if save_feats:
        feats_dict = {}
        set_classifier_to_identity(net)

    idx = 0
    for image, label in trainset:
        idx += 1
        image, label = image.to(device), label.to(device)
        optimizer.zero_grad()
        
        with autocast(enabled=True):
            pred = net.forward(image)
            
            if save_feats:
                feats_dict[idx] = [pred.logits, label]
                continue
            loss = loss_func(pred.logits, label)

        if save_feats:
            outfile = 'pretrained_feats_clstoken/feats.pth'
            torch.save(outfile, feats_dict)
            logging.info('Features saved in: ', outfile)
            sys.exit()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        
        scaler.update()
        running_loss = running_loss + loss.item()
    scheduler.step()
    
    loss = running_loss / len(trainset)
    logging.info(f"EPOCH: {epoch}, LOSS : {loss:.4f}, LR: {this_lr:.2e}")

    return loss

@torch.no_grad()
def eval(net, cfg, epoch, testset, datatype='val'):
    result = ResultCLS(cfg.num_classes)
    result.init()
    net.eval()
    for image, label in testset:
        image, label = image.to(device), label.to(device)
        with autocast(enabled=True):
            pred = net.forward(image)
            result.eval(label, pred.logits)
    result.print(epoch, datatype)
    return result

def set_random_seed(seed: int, rank: int = 0, deterministic: bool = True):
    adjusted_seed = seed + rank
    random.seed(adjusted_seed)
    np.random.seed(adjusted_seed)
    torch.manual_seed(adjusted_seed)
    torch.cuda.manual_seed(adjusted_seed)
    torch.cuda.manual_seed_all(adjusted_seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


if __name__ == "__main__":
    scaler = GradScaler()
    parser = argparse.ArgumentParser()
    parser.add_argument("-bs", "--batch_size", type=int, default=16)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--model_path", type=str, default='../model')
    parser.add_argument("--train_data_path", type=str, default='../data')
    parser.add_argument("--val_data_path", type=str, default=None)
    parser.add_argument("--outdir", type=str, default='./exp')
    parser.add_argument("--logdir", type=str, default='./exp.log')
    parser.add_argument("--data_info", type=str, default='data.json')
    parser.add_argument("--annotation", type=str, default='data.csv')
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_classes", "-nc", type=int, default=100)
    parser.add_argument("--train_type", "-tt", type=str, default="linear", help="nola, lora, full, linear, adapter")
    parser.add_argument("--rank", "-r", type=int, default=4)
    parser.add_argument("--vit", type=str, default="base")
    parser.add_argument("--kshot", type=int, default=0,
                        help="use only k-samples per category for training")
    parser.add_argument("--kshot_seed", type=int, default=0,
                        help='seed to use to select kshot samples')
    parser.add_argument("--seed", type=int, default=0,
                        help='seed for other all operations')
    parser.add_argument("--dset", type=str, default='CIFAR10')
    parser.add_argument('--eval', action='store_true',
                        help='evaluation of pretrained model only, no training')

    cfg = parser.parse_args()
    ckpt_path = init(cfg.outdir, cfg.logdir)
    for key, value in vars(cfg).items():
        print(f"--{key}: {value}")

    acc_file = '%s/acc_file.csv' % cfg.outdir
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_random_seed(cfg.seed)

    cfg.model_path = model_to_path[cfg.vit]
    cfg.num_classes = dataset_to_class_n[cfg.dset]
    config = AutoConfig.from_pretrained(cfg.model_path)
    config.num_labels = cfg.num_classes

    model = AutoModelForImageClassification.from_pretrained(cfg.model_path, config=config)

    if cfg.train_type == "lora":
        from peft import get_peft_model, LoraConfig

        lora_config = LoraConfig(
            r=cfg.rank,
            lora_alpha=32.,
            lora_dropout=0.05,
            target_modules=['query', 'value'],
        )

        net = get_peft_model(model, lora_config)

        for name, param in net.named_parameters():
            if 'classifier' in name:
                param.data = param.data.to(torch.float32)
                param.requires_grad = True

    elif cfg.train_type == "trac":
        from trac import TracConfig, get_peft_model

        hidden_size_dict = {'base': 768, 'large': 1024}  
        mlp_size_dict =  {'base': 3072, 'large': 4096}

        config = TracConfig(
            r=cfg.rank,
            lora_alpha=32.,
            target_modules=['query', 'value'],
            hidden_size=hidden_size_dict[cfg.vit],
            mlp_hidden_dim=mlp_size_dict[cfg.vit],
            lora_dropout=0.05,
        )
        
        net = get_peft_model(model, config)

        for name, param in net.named_parameters():
            if 'classifier' in name:
                param.data = param.data.to(torch.float32)
                param.requires_grad = True

    elif cfg.train_type == "full":
        net = model
    
    elif cfg.train_type == "head":
        net = model

        for name, param in net.named_parameters():
            if 'classifier' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    total_param, peft_trainable_param = compute_trainable_param(net)
    print(f'--- total_param: {total_param} \n--- peft_trainable_param: {peft_trainable_param}')

    net = torch.nn.DataParallel(net)

    cfg.train_data_path = dataset_to_path[cfg.dset]['train_data_path']
    cfg.val_data_path = dataset_to_path[cfg.dset]['val_data_path']
    if cfg.val_data_path == None:
        trainset, valset, testset = load_dataset_no_val(cfg)
    else:
        trainset, valset, testset = load_dataset(cfg)
    
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in net.named_parameters() if (p.requires_grad)], 
        }]
    
    optimizer = optim.AdamW(optimizer_grouped_parameters)
    scheduler = CosineAnnealingLR(optimizer, cfg.epochs, 1e-6)

    loss_func = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)

    if cfg.eval:
        result = eval(net, cfg, 0, valset, datatype='val')
        logging.info(f"BEST VAL: {result.best_val_result:.3f}, EPOCH: {result.best_epoch:3}")
        sys.exit()

    best_val = 0.
    best_val_ep = 0
    
    for epoch in range(1, cfg.epochs+1):
        if not cfg.eval:
            loss = train(epoch, trainset)
            
        if (epoch == cfg.epochs) :
            net.train()

            checkpoint_name = f"checkpoint-epoch-{epoch}"
            net.module.save_pretrained(save_directory=f"{cfg.outdir}/{checkpoint_name}", weight_name="adapter_weight.pt")

            classifier_state = {
                "classifier.weight": net.module.base_model.model.classifier.weight.detach().cpu(),
                "classifier.bias": net.module.base_model.model.classifier.bias.detach().cpu()
            }
            torch.save(classifier_state, f"{cfg.outdir}/{checkpoint_name}/classifier_weight.pt")

            net.eval()
            result = eval(net, cfg, epoch, valset, datatype='val')
            best_val = result.best_val_result
            best_val_ep = result.best_epoch

    print('='*50)
    print(f"BEST VAL: {best_val:.4f}, EPOCH: {best_val_ep:3}")

    if not cfg.eval:
        save2file(result.acc * 100., loss, best_val*100., best_val_ep, cfg.outdir, acc_file)


