#!/bin/bash
#PBS-N UM2KM
#PBS-l select=1:ngpus=1:mem=440G
#PBS-l walltime=6:00:00
#PBS-j oe
#PBS-o out-run.txt
#PBS-q normal
module load miniforge3
conda activate mqbench
module load cuda/12.2.1

cd ~/mydir
python3 /home/users/ntu/sooq0001/mydir/MQBench/imagenet_example/PTQ/ptq/verify_ptq.py --config /home/users/ntu/sooq0001/mydir/MQBench/imagenet_example/PTQ/configs/adaround/FYP_r18_8_8_SQ05.yaml
