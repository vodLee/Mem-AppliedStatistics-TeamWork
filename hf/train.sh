#! /usr/bin/env bash

set -ex

DATESTR=`date +%Y%m%d-%H%M%S`
RUN_NAME=news_qwen2
OUTPUT_DIR=output/${RUN_NAME}-${DATESTR}
mkdir -p $OUTPUT_DIR

MODEL_PATH="/root/autodl-tmp/Qwen2.5-7B-Instruct"

CUDA_VISIBLE_DEVICES=0 python reward_modeling.py \
    --model_name_or_path "${MODEL_PATH}" \
    --dataset_name train.json \
    --output_dir ${OUTPUT_DIR} \
    --per_device_train_batch_size 8 \
    --num_train_epochs 1 \
    --gradient_checkpointing True \
    --learning_rate 1.0e-4 \
    --logging_steps 25 \
    --eval_strategy steps \
    --eval_steps 50 \
    --max_length 2048 \
    --use_peft \
    --lora_r 32 \
    --lora_alpha 16
    # --do_train \
    # --do_eval \
    # --train_file ./train.json \
    # --validation_file ./val.json \
    # --prompt_column context \
    # --response_column response \
    # --model_name_or_path "${MODEL_PATH}" \
    # --output_dir $OUTPUT_DIR \
    # --max_source_length 4096 \
    # --max_target_length 4096 \
    # --per_device_train_batch_size 1 \
    # --per_device_eval_batch_size 1 \
    # --gradient_accumulation_steps 8 \
    # --evaluation_strategy steps \
    # --lr_scheduler_type cosine \
    # --eval_steps 300 \
    # --num_train_epochs 5 \
    # --logging_steps 30 \
    # --max_grad_norm 1.0 \
    # --logging_dir $OUTPUT_DIR/logs \
    # --save_steps 300 \
    # --learning_rate $LR \
    # --lora_rank 8 \
    # --lora_alpha 32 \
    # --lora_dropout 0.3 2>&1 | tee ${OUTPUT_DIR}/train.log
