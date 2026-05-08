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
export TRT_DIR=$HOME/TensorRT-10.14.1.48
export LD_LIBRARY_PATH=$TRT_DIR/lib:$LD_LIBRARY_PATH
export PATH=$TRT_DIR/bin:$PATH
export PYTHONPATH=$TRT_DIR/python:$PYTHONPATH

cd ~/mydir
#nsys profile --gpu-metrics-device=all --trace cuda,nvtx,osrt -o nsys_run 
#ncu -o ncureport --target-processes all 
python3 /home/users/ntu/sooq0001/mydir/MQBench/imagenet_example/onnx2trt.py --onnx-path /home/users/ntu/sooq0001/res18_8_8_SQ05_deploy_model.onnx --trt /home/users/ntu/sooq0001/res18_8_8_SQ05_deploy_model.trt --data /home/users/ntu/sooq0001/scratch/ImageNet-ILSVRC2012/val --clip-range-file /home/users/ntu/sooq0001/res18_8_8_SQ05_clip_ranges.json --evaluate --explicit --batch-size 1

