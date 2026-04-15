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

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

timestamp=$(date "+%Y%m%d-%H%M%S")

arch='base' # 'base' 'large'
train_type='trac' # 'full' 'head' 'lora' 'trac'
kshot=10

for rank in 16 
do
for lr in 2e-3
do
for epoch in 50
do
for bsz in 16
do
for dset in 'CIFAR10' # 'CIFAR10' 'CIFAR100' 'CUB_200_2011' 'flowers102'
do
# Loop - 4 diff seeds for dataset kshot sampling (idx j) --> 3 runs (with diff seeds for network init, idx k)
# add eval argument to perform only evaluation
for seed in 0 # 0 42 100
do
for j in 0 # {0..3}  or  0 1 2 3
do

log_file="vit/results/results_${train_type}_${arch}/data_${dset}/${dset}_${timestamp}_vit_${arch}_${train_type}_rank_${rank}_lr_${lr}_epoch_${epoch}_bs_${bsz}_shot_${kshot}_sampling_${j}_seed_${seed}.log"
echo "$log_file"
echo "device=" $CUDA_VISIBLE_DEVICES

mkdir -p "$(dirname "$log_file")"
(exec > "$log_file" 2>&1

python vit/src/train_vit.py \
    --train_type "$train_type" \
    --rank $rank \
    --lr $lr \
    --vit $arch \
    --batch_size $bsz \
    --kshot $kshot \
    --kshot_seed "$j" \
    --seed $seed \
    --dset $dset \
    --epochs $epoch \
    --outdir vit/results/results_${train_type}_${arch}/data_${dset}/${dset}_${timestamp}_vit_${arch}_${train_type}_rank_${rank}_lr_${lr}_epoch_${epoch}_bs_${bsz}_shot_${kshot}_sampling_${j}_seed_${seed} \
    --logdir $log_file)
done
done
done
done
done
done
done