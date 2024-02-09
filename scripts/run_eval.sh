
in_dataset=$1
model=$2
SCORE=$3
p=$4
threshold=$5
GPU=0

model_path="checkpoints/network"
logdir='result'
name='baseline'

CUDA_VISIBLE_DEVICES=${GPU} python ood_eval.py \
--model ${model} \
--name ${name} \
--batch 64 \
--in_dataset ${in_dataset} \
--logdir ${logdir} \
--model_path ${model_path} \
--score ${SCORE} \
--p ${p}    \
--threshold ${threshold}
