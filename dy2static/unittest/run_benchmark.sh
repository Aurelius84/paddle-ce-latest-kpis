#!/bin/bash

run_model_train()
{
    model_dir=$1
    script=$2
    bs=$3
    cd $model_dir
    echo "traing $model_dir with $script ..."
    bash $script $bs
    wait
    cd ..
}

export CUDA_VISIBLE_DEVICES="7"

batch_size=128

# mnist
# run_model_train mnist_dy2static train.sh $batch_size
# mobilenet v1/v2
run_model_train mobile_net_dy2static train.sh $batch_size
# ptb lm
run_model_train ptb_lm_dy2static run.sh $batch_size
# resnet
run_model_train resnet_dy2static train.sh $batch_size
# seresnet
run_model_train seresnet_dy2static train.sh $batch_size
# reinforcement learning
# run_model_train reinforcement_learning_dy2static train.sh

# reinforcement learning
run_model_train bert_base_dy2static train.sh 64

# parse log
python benchmark_parser.py --output $1
