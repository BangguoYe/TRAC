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

export PYTHONUNBUFFERED=1
export WANDB_DISABLED=true

TIMESTAMP=$(date "+%Y%m%d-%H%M%S")

device=0

for model_name in 'roberta-base' # 'debertav3-base' 'roberta-base' 'roberta-large'
do
for seed in 42 # 0 21 42 81 100
do
for task in cola # cola mrpc rte stsb sst2 qnli
do
for peft_type in lora # lora trac
do
for rank_r in 16
do
for lr in 1e-3
do
for scale_factors_lr in 1e-3
do

eval_steps=50

# Use different epochs and batch sizes for different tasks
case "$task" in
    cola|mrpc|stsb|rte)
        epoch=20
        bsz=64
        ;;
    sst2|qnli)
        epoch=10
        bsz=64
        ;;
    *)
        echo "Unknown task: $task"
        exit 1
        ;;
esac

echo "task=" $task
echo "lr=" $lr
echo "epoch=" $epoch
echo "rank_r=" $rank_r
echo "seed=" $seed
echo "device=" $device

log_file="bert/results/results_${peft_type}_${model_name}/task_${task}/${task}_${TIMESTAMP}_model_${model_name}_${peft_type}_rank_${rank_r}_lr_${lr}_epoch_${epoch}_bs_${bsz}_seed_${seed}.log"
echo "$log_file"
mkdir -p "$(dirname "$log_file")"
(exec > "$log_file" 2>&1

CUDA_VISIBLE_DEVICES=$device \
python -u bert/src/train_glue.py \
    --do_eval \
    --do_predict \
    --do_train \
    --task_name $task \
    --eval_steps $eval_steps \
    --evaluation_strategy epoch \
    --greater_is_better true \
    --learning_rate $lr \
    --scale_factors_learning_rate $scale_factors_lr \
    --max_grad_norm 1000 \
    --load_best_model_at_end false \
    --logging_steps 10 \
    --max_steps -1 \
    --model_name_or_path models/bert/roberta-base \
    --model_name $model_name \
    --num_train_epochs $epoch \
    --output_dir bert/results/results_${peft_type}_${model_name}/task_${task}/${task}_${TIMESTAMP}_model_${model_name}_${peft_type}_rank_${rank_r}_lr_${lr}_epoch_${epoch}_bs_${bsz}_seed_${seed} \
    --overwrite_output_dir \
    --per_device_eval_batch_size $bsz \
    --per_device_train_batch_size $bsz \
    --save_steps $eval_steps \
    --save_strategy epoch \
    --save_total_limit 2 \
    --tokenizer_name models/bert/roberta-base \
    --warmup_ratio 0.06 \
    --warmup_steps 0 \
    --weight_decay 0.1 \
    --disable_tqdm true \
    --load_best_model_at_end false \
    --ddp_find_unused_parameters false \
    --seed $seed \
    --lora_r $rank_r \
    --lora_alpha 32.0 \
    --peft_type $peft_type \
    --do_test_during_training false \
    --max_seq_length 128) 
done
done
done
done
done
done
done