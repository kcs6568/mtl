#!/bin/bash

GPUS=$1
PORT=$2
METHOD=$3
TRAIN_ROOT=/root/mtl_cl/
CFG_PATH=/root/mtl_cl/cfgs/four_task/$METHOD
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
<CIFAR10-STL10-minicoco-voc>
 - SGD
    
        
<CIFAR100-minicoco-voc>
 - SGD

### ONGOING ###



### To Do ###

END

CUDA_VISIBLE_DEVICES=$DEVICES torchrun --nproc_per_node=$1 --master_port=$2 \
    $TRAIN_ROOT/train.py --lossbal --loss-ratio 1 1 6 1 1 \
    --cfg $CFG_PATH/cifar10_stl10_minicoco_voc_city/resnet50_clf_det_seg.yaml \
    --exp-case baseline_nGPU"$GPUS"_adamw_nowarmup_gamma0.25_lossbal[1,1,6,1,1] \
    --lr 1e-4 --use-minids --task-bs 1 1 1 1 1 \
    --opt adamw --warmup-ratio -1 --lr-gamma 0.25 --workers 8
 
sleep 5
$KILL_PROC
$KILL_PROC
$KILL_PROC
sleep 2

# CUDA_VISIBLE_DEVICES=$DEVICES torchrun --nproc_per_node=$1 --master_port=$2 \
#     $TRAIN_ROOT/train.py --lossbal --loss-ratio 1 1 6 2 \
#     --cfg $CFG_PATH/cifar10_stl10_minicoco_voc/resnet50_clf_det_seg.yaml \
#     --exp-case lossbal_alpha0.9_beta0.1_nowarmup_sgd_stitchlr1e-2_defaultlr1e-4_gamma0.1_lossbal[1,1,6,2] \
#     --lr 1e-4 --use-minids --task-bs 2 2 2 2 \
#     --warmup-ratio -1 --workers 8
 
# sleep 5
# $KILL_PROC
# $KILL_PROC
# $KILL_PROC
# sleep 2


# # AdamW / no Warmup / lossbal ratio 1 1 6 2 / multi 2e-4 / lr gamma 0.25 / fft 
# CUDA_VISIBLE_DEVICES=$DEVICES torchrun --nproc_per_node=$1 --master_port=$2 \
#     $TRAIN_ROOT/train.py --lossbal --loss-ratio 1 1 6 2 \
#     --cfg $CFG_PATH/cifar10_stl10_minicoco_voc/resnet50_clf_det_seg.yaml \
#     --exp-case aaaaa \
#     --lr 2e-4 --use-minids --task-bs 4 4 2 2 --allbackbone-train --freeze-bn --dilation-type fft \
#     --opt adamw --warmup-ratio -1 --lr-gamma 0.25
 
# sleep 5
# $KILL_PROC
# $KILL_PROC
# $KILL_PROC
# sleep 2

# CUDA_VISIBLE_DEVICES=$DEVICES torchrun --nproc_per_node=$1 --master_port=$2 \
#     $TRAIN_ROOT/lossbal_train_up_two.py --lossbal --loss-ratio 1 1 6 2 \
#     --cfg $CFG_PATH/cifar10_stl10_minicoco_voc/resnet50_clf_det_seg.yaml \
#     --exp-case adamw_nowarmup_gamma0.25_nGPU"$GPUS"_multi_lr2e-4_FreezeBN_alltrain_lossbal[1,1,6,2]_dil[fft] \
#     --lr 2e-4 --use-minids --task-bs 4 4 2 2 --allbackbone-train --freeze-bn --dilation-type fft \
#     --opt adamw --warmup-ratio -1 --lr-gamma 0.1
 
# sleep 5
# $KILL_PROC
# $KILL_PROC
# $KILL_PROC
# sleep 2

