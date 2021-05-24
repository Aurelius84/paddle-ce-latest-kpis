#!/bin/bash

model="ptb_lm"
bs=$1
print_step=`expr 1280 / $bs`

log_path="./${model}.log"
python model.py --device=GPU --batch_size=128 --pass_num=2 --log_internal=$print_step > $log_path 2>&1
wait 

log_path="./${model}_static_baseline.log"
python static_baseline.py --device=GPU --batch_size=128 --pass_num=2 --log_internal=$print_step > $log_path 2>&1

