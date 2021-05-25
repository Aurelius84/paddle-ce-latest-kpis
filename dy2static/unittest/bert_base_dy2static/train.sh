#!/bin/bash

export CUDA_VISIBLE_DEVICES="7"

bs=$1

# dygraph
python run_pretrain.py \
    --model_type bert \
    --to_static False \
    --model_name_or_path bert-base-uncased \
    --max_predictions_per_seq 20 \
    --batch_size $bs   \
    --learning_rate 1e-4 \
    --weight_decay 1e-2 \
    --adam_epsilon 1e-6 \
    --warmup_steps 10000 \
    --num_train_epochs 1 \
    --input_dir data/bert_data \
    --output_dir pretrained_models/ \
    --logging_steps 20 \
    --save_steps 20000 \
    --max_steps 100 \
    --device gpu \
    --use_amp False  > bert_base.log 2>&1

wait

# to_static
python run_pretrain.py \
    --model_type bert \
    --to_static True \
    --model_name_or_path bert-base-uncased \
    --max_predictions_per_seq 20 \
    --batch_size $bs   \
    --learning_rate 1e-4 \
    --weight_decay 1e-2 \
    --adam_epsilon 1e-6 \
    --warmup_steps 10000 \
    --num_train_epochs 1 \
    --input_dir data/bert_data \
    --output_dir pretrained_models/ \
    --logging_steps 20 \
    --save_steps 20000 \
    --max_steps 100 \
    --device gpu \
    --use_amp False >> bert_base.log 2>&1

wait

# # static baseline
python static_baseline.py \
    --model_type bert \
    --model_name_or_path bert-base-uncased \
    --max_predictions_per_seq 20 \
    --batch_size $bs   \
    --learning_rate 1e-4 \
    --weight_decay 1e-2 \
    --adam_epsilon 1e-6 \
    --warmup_steps 10000 \
    --input_dir data/bert_data \
    --output_dir pretrained_models/ \
    --logging_steps 20 \
    --save_steps 20000 \
    --max_steps 100 \
    --device gpu \
    --use_amp False > bert_base_static_baseline.log 2>&1