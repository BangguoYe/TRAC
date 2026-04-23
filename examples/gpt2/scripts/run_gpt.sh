# This file contains code derived from the LoRA project:
#   https://github.com/microsoft/LoRA
#
# Original work Copyright (c) Microsoft Corporation.
# Modifications Copyright (c) 2026 Bangguo Ye, Yuanwei Zhang, Xiaoqun Zhang
#
# Licensed under the MIT License (MIT).

# ============================================================
# Configuration=
# ============================================================
MODEL_SIZE="md"       # "md" (gpt2-medium) or "lg" (gpt2-large)
CUDA_DEVICE=4
MASTER_PORT=10601

# ============================================================
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE

export MASTER_PORT=$MASTER_PORT
export RANK=0
export WORLD_SIZE=1
export MASTER_ADDR=localhost
export TORCH_DISTRIBUTED_DEBUG=DETAIL   # Can be set to INFO or commented out

model_card="gpt2.${MODEL_SIZE}"

if [ "$MODEL_SIZE" = "md" ]; then
    init_checkpoint="models/gpt/pretrained_checkpoints/gpt2-medium-pytorch_model.bin"
elif [ "$MODEL_SIZE" = "lg" ]; then
    init_checkpoint="models/gpt/pretrained_checkpoints/gpt2-large-pytorch_model.bin"
else
    echo "Error: MODEL_SIZE must be 'md' or 'lg', got '${MODEL_SIZE}'"
    exit 1
fi

# ============================================================
# Training loop
# ============================================================
for lr in 0.005
do
scale_factors_lr=0.0005
TIMESTAMP=$(date "+%Y%m%d-%H%M%S")

# Method and dataset
TYPE='trac'
TASK='e2e'
SEED=110

# Training hyperparameters
lora_r=16
max_epoch=5
train_batch_size=8
target_modules="q,v"

# Output paths
RESULT_BASE="examples/gpt2/results"
EXP_NAME="${TASK}_${TIMESTAMP}_model_${model_card}_ft_${TYPE}_rank_${lora_r}_lr_${lr}_scale_factors_lr_${scale_factors_lr}_epoch_${max_epoch}_bs_${train_batch_size}_seed_${SEED}"
output_dir="${RESULT_BASE}/results_${TYPE}_${model_card}/${EXP_NAME}"

log_file="${output_dir}.log"
final_checkpoint_path="${output_dir}/checkpoint-26290.pt"   # The exact checkpoint index may need adjustment based on the actual number of training steps
predict_output_file_json="${output_dir}/e2e_predict.jsonl"
predict_output_file_txt="${output_dir}/e2e_predict.txt"
ref_output_file_txt="${output_dir}/e2e_ref.txt"

echo "$log_file"
mkdir -p "$output_dir"

(
exec > "$log_file" 2>&1

python -m torch.distributed.launch --nproc_per_node=1 --master_port=$MASTER_PORT \
    examples/gpt2/src/gpt2_ft.py \
    --train_data examples/gpt2/e2e/train.jsonl \
    --valid_data examples/gpt2/e2e/valid.jsonl \
    --train_batch_size $train_batch_size \
    --grad_acc 1 \
    --valid_batch_size 4 \
    --seq_len 512 \
    --model_card "$model_card" \
    --init_checkpoint $init_checkpoint \
    --platform local \
    --clip 0.0 \
    --lr $lr \
    --scale_factors_lr $scale_factors_lr \
    --weight_decay 0.01 \
    --correct_bias \
    --adam_beta2 0.999 \
    --scheduler linear \
    --warmup_step 500 \
    --max_epoch $max_epoch \
    --log_interval 100 \
    --save_interval 1000 \
    --eval_interval 2000 \
    --lora_dim $lora_r \
    --target_modules $target_modules \
    --lora_alpha 32 \
    --lora_dropout 0.0 \
    --label_smooth 0.1 \
    --work_dir $output_dir \
    --random_seed $SEED

python -m torch.distributed.launch --nproc_per_node=1 --master_port=$MASTER_PORT \
    examples/gpt2/src/gpt2_beam.py \
    --data examples/gpt2/e2e/test.jsonl \
    --batch_size 1 \
    --seq_len 512 \
    --eval_len 64 \
    --model_card "$model_card" \
    --init_checkpoint $init_checkpoint \
    --final_checkpoint $final_checkpoint_path \
    --platform local \
    --lora_dim $lora_r \
    --target_modules $target_modules \
    --lora_alpha 32 \
    --beam 10 \
    --length_penalty 0.8 \
    --no_repeat_ngram_size 4 \
    --repetition_penalty 1.0 \
    --eos_token_id 628 \
    --work_dir $output_dir \
    --output_file $predict_output_file_json

python examples/gpt2/src/gpt2_decode.py \
    --vocab examples/gpt2/vocab \
    --sample_file $predict_output_file_json \
    --input_file examples/gpt2/e2e/test_formatted.jsonl \
    --output_ref_file $ref_output_file_txt \
    --output_pred_file $predict_output_file_txt

python examples/gpt2/eval/e2e/measure_scores.py $ref_output_file_txt $predict_output_file_txt -p

)

done