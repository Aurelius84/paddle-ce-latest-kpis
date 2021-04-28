#!/bin/bash

run_model_train()
{
    model_dir=$1
    script=$2
    cd $model_dir
    echo "traing $model_dir with $script ..."
    bash $script
    wait
    cd ..
}

# mnist
run_model_train mnist_dy2static train.sh
# mobilenet v1/v2
run_model_train mobile_net_dy2static train.sh
# ptb lm
run_model_train ptb_lm_dy2static run.sh
# resnet
run_model_train resnet_dy2static train.sh
# seresnet
run_model_train seresnet_dy2static train.sh
# reinforcement learning
# run_model_train reinforcement_learning_dy2static train.sh

# parse log
python benchmark_parser.py --output $1
