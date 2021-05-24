#!/bin/bash

model="mnist"
bs=$1
print_step=`expr 6400 / $bs`

log_path="./${model}.log"

python model.py --device=GPU --batch_size=$bs --pass_num=1 --log_internal=$print_step > $log_path 2>&1
wait

log_path="./${model}_static_baseline.log"
python static_baseline.py --device=GPU --batch_size=$bs --pass_num=1 --log_internal=$print_step > $log_path 2>&1
wait
