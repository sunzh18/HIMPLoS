
in_dataset=$1
model=$2
GPU=0

model_path="checkpoints/network"
logdir='result'
name='baseline'

CUDA_VISIBLE_DEVICES=${GPU} python precompute.py \
--model ${model} \
--name ${name} \
--batch 16 \
--in_dataset ${in_dataset} \
--logdir ${logdir} \
--model_path ${model_path} 
