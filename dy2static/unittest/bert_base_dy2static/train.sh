#!/bin/bash

if [ ! -f "data/training_data.hdf5" ]; then
    python create_pretraining_data.py \
    --input_file=data/sample_text.txt \
    --output_file=data/training_data.hdf5 \
    --bert_model=bert-base-uncased \
    --max_seq_length=128 \
    --max_predictions_per_seq=20 \
    --masked_lm_prob=0.15 \
    --random_seed=12345 \
    --dupe_factor=5
fi

export CUDA_VISIBLE_DEVICES="7"

python run_pretrain.py \
    --model_type bert \
    --to_static False \
    --model_name_or_path bert-base-uncased \
    --max_predictions_per_seq 20 \
    --batch_size 16   \
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

python run_pretrain.py \
    --model_type bert \
    --to_static True \
    --model_name_or_path bert-base-uncased \
    --max_predictions_per_seq 20 \
    --batch_size 16   \
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