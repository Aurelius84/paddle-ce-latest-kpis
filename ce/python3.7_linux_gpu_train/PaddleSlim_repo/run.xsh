#!/bin/bash
export slim_dir=$PWD/../../slim_repo
#copy PaddleSlim files
cp -r ${slim_dir}/. ./
python setup.py install
if [ -d "demo/data" ];then rm -rf demo/data
fi
ln -s ${dataset_path}/slim/data demo/data
./.run_ce.sh
