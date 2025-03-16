#! /usr/bin/env bash
MODEL_DIR="/root/autodl-tmp/Qwen2.5-7B-Instruct"
CHECKPOINT_DIR="output/news_qwen2-20250316-091136/checkpoint-440"

DATESTR=`date +%Y%m%d-%H%M%S`
RUN_NAME=news_test_qwen2
OUTPUT_DIR=output/${RUN_NAME}-${DATESTR}
mkdir -p $OUTPUT_DIR

CUDA_VISIBLE_DEVICES=0 python evaluate.py \
  --model $MODEL_DIR \
  --ckpt $CHECKPOINT_DIR \
  --output_dir $OUTPUT_DIR \
  --data ./test.json
