#!/bin/bash

model="mobile_net"
bs=$1
print_step=`expr 640 / $bs`

type="v1"
log_path="./${model}_${type}.log"
python model.py --model_type=$type --device=GPU --batch_size=$bs --pass_num=1 --log_internal=$print_step > $log_path 2>&1
wait

log_path="./${model}_${type}_static_baseline.log"
python static_baseline.py --model_type=$type --device=GPU --batch_size=$bs --pass_num=1 --log_internal=$print_step > $log_path 2>&1
wait


type="v2"
log_path="./${model}_${type}.log"
python model.py --model_type=$type --device=GPU --batch_size=$bs --pass_num=1 --log_internal=$print_step > $log_path 2>&1
wait

log_path="./${model}_${type}_static_baseline.log"
python static_baseline.py --model_type=$type --device=GPU --batch_size=$bs --pass_num=1 --log_internal=$print_step > $log_path 2>&1
