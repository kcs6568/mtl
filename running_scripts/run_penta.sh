#!/bin/bash

GPUS=$1
PORT=$2
METHOD=$3
TRAIN_ROOT=/root/mtl_cl/
CFG_PATH=/root/mtl_cl/cfgs/five_task/$METHOD
KILL_PROC="kill $(ps aux | grep train.py | grep -v grep | awk '{print $2}')"

# make visible devices order automatically
DEVICES=""
for ((n=7; n>=((8-$GPUS)); n--))
do
    DEVICES+=$n

    if [ $n -gt $((8-$GPUS)) ]
    then
        DEVICES+=,
    fi
done

$KILL_PROC
$KILL_PROC
$KILL_PROC
sleep 2

: <<'END'
<CIFAR10-STL10-minicoco-voc-cityscape>
    - AdamW / lr 1e-4
        - baseline_nGPU8_nowarmup_multi_gamma0.25_lossbal[1,1,6,1,1]: 성능 x

    - AdamW / lr 1e-3
        - baseline_nGPU8_nowarmup_multi_gamma0.1_lossbal[1,1,6,1,1]: 성능 x
        - baseline_nGPU8_nowarmup_multi_gamma0.25_lossbal[1,1,6,1,1]: 

    - SGD / lr 1e-4

    - SGD / lr 1e-3
        - baseline_nGPU8_nowarmup_multi_gamma0.1_lossbal[1,1,6,1,1]
        
### ONGOING ###



### To Do ###

END

CUDA_VISIBLE_DEVICES=$DEVICES torchrun --nproc_per_node=$1 --master_port=$2 \
    $TRAIN_ROOT/train.py --lossbal --loss-ratio 1 1 6 1 1 \
    --cfg $CFG_PATH/cifar10_stl10_minicoco_voc_city/resnet50_clf_det_seg.yaml \
    --exp-case baseline_nGPU"$GPUS"_adamw_lr1e-3_nowarmup_gamma0.1_lossbal[1,1,6,1,1] \
    --lr 1e-3 --use-minids --task-bs 4 4 2 2 2 \
    --opt adamw --warmup-ratio -1 --lr-gamma 0.1 --workers 8
 
sleep 5
$KILL_PROC
$KILL_PROC
$KILL_PROC
sleep 2

CUDA_VISIBLE_DEVICES=$DEVICES torchrun --nproc_per_node=$1 --master_port=$2 \
    $TRAIN_ROOT/train.py --lossbal --loss-ratio 1 1 6 1 1 \
    --cfg $CFG_PATH/cifar10_stl10_minicoco_voc_city/resnet50_clf_det_seg.yaml \
    --exp-case baseline_nGPU"$GPUS"_adamw_lr1e-3_nowarmup_gamma0.25_lossbal[1,1,6,1,1] \
    --lr 1e-3 --use-minids --task-bs 4 4 2 2 2 \
    --opt adamw --warmup-ratio -1 --lr-gamma 0.25 --workers 8
 
sleep 5
$KILL_PROC
$KILL_PROC
$KILL_PROC
sleep 2


CUDA_VISIBLE_DEVICES=$DEVICES torchrun --nproc_per_node=$1 --master_port=$2 \
    $TRAIN_ROOT/train.py --lossbal --loss-ratio 1 1 6 1 1 \
    --cfg $CFG_PATH/cifar10_stl10_minicoco_voc_city/resnet50_clf_det_seg.yaml \
    --exp-case baseline_nGPU"$GPUS"_sgd_lr1e-3_nowarmup_gamma0.1_lossbal[1,1,6,1,1] \
    --lr 1e-3 --use-minids --task-bs 4 4 2 2 2 \
    --opt sgd --warmup-ratio -1 --lr-gamma 0.1 --workers 8
 
sleep 5
$KILL_PROC
$KILL_PROC
$KILL_PROC
sleep 2


CUDA_VISIBLE_DEVICES=$DEVICES torchrun --nproc_per_node=$1 --master_port=$2 \
    $TRAIN_ROOT/train.py --lossbal --loss-ratio 1 1 6 1 1 \
    --cfg $CFG_PATH/cifar10_stl10_minicoco_voc_city/resnet50_clf_det_seg.yaml \
    --exp-case baseline_nGPU"$GPUS"_sgd_lr1e-4_nowarmup_gamma0.1_lossbal[1,1,6,1,1] \
    --lr 1e-4 --use-minids --task-bs 4 4 2 2 2 \
    --opt sgd --warmup-ratio -1 --lr-gamma 0.1 --workers 8
 
sleep 5
$KILL_PROC
$KILL_PROC
$KILL_PROC
sleep 2


