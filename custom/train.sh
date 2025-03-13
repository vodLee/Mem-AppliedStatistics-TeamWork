#! /usr/bin/env bash

set -ex

LR=2e-4

DATESTR=`date +%Y%m%d-%H%M%S`
RUN_NAME=news_qwen2
OUTPUT_DIR=output/${RUN_NAME}-${DATESTR}
mkdir -p $OUTPUT_DIR

MODEL_PATH="/root/autodl-tmp/Qwen2.5-7B-Instruct"

CUDA_VISIBLE_DEVICES=0 python finetune.py \
    --do_train \
    --do_eval \
    --train_file ./train.jsonl \
    --validation_file ./dev.jsonl \
    --prompt_column context \
    --response_column response \
    --model_name_or_path "${MODEL_PATH}" \
    --output_dir $OUTPUT_DIR \
    --max_source_length 4096 \
    --max_target_length 4096 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy steps \
    --eval_steps 300 \
    --num_train_epochs 2 \
    --logging_steps 30 \
    --logging_dir $OUTPUT_DIR/logs \
    --save_steps 300 \
    --learning_rate $LR \
    --lora_rank 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 2>&1 | tee ${OUTPUT_DIR}/train.log
