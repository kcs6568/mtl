#!/bin/bash

GPUS=$1
PORT=$2
TRAIN_ROOT=/root/mtl_cl/
CFG_PATH=/root/mtl_cl/cfgs/single_task
KILL_PROC="kill $(ps aux | grep train.py | grep -v grep | awk '{print $2}')"
DEVICES=""
for ((n=7; n>=((8-$GPUS)); n--))
do
    DEVICES+=$n

    if [ $n -gt $((8-$GPUS)) ]
    then
        DEVICES+=,
    fi
done

: <<'END'

[CLF]
- CIFAR10
    - default / multi / 0.01 / fff: 95.19

    - Same Setting for multi task learning & ftt dilation setting
        - multi / 0.01: 94.91
        - cosine / 0.01: 91.71


- CIFAR100
    - default / multi / 0.01: 78.3
    - default / multi / 0.02: x

[DET]
- Minicoco
    - Same Setting for multi task learning & different dilation setting & 12e
        -  multi / 0.01: 25.02
        -  cosine / 0.01: 24.6

        -  multi / 0.02:
        -  cosine / 0.02:
        


[SEG]
- voc 0712
    - multi / 0.01 / aux: 75.7
    - multi / 0.02 / aux: 76.87

    - cosine / 0.01 / aux: 76.82
    - cosine / 0.02 / aux: 78.2

    - Same Setting for multi task learning & different dilation setting
        - multi / 0.01 / aux: 66.29
        - multi / 0.02 / aux: 62.66

        - cosine / 0.01 / aux: 59.73
        - cosine / 0.02 / aux: 55.43

    - Same Setting for multi task learning & different dilation setting & 145 epoch training
        - multi / 0.01 / aux: 90.45
        - multi / 0.02 / aux: 

        - cosine / 0.01 / aux: 90.53
        - cosine / 0.02 / aux: 

    - Same Setting for multi task learning & different dilation setting & 97 epoch training
        - multi / 0.01 / aux: 88.42
        - multi / 0.02 / aux: 86.32

        - cosine / 0.01 / aux: 89.1
        - cosine / 0.02 / aux: 88.89

END

LR_LISTS="0.01 0.02 0.015 0.1"

# for lr in $LR_LISTS
# do
#     CUDA_VISIBLE_DEVICES=$DEVICES torchrun --nproc_per_node=$1 --master_port=$2 \
#     $TRAIN_ROOT/single_train.py --general \
#     --cfg $CFG_PATH/resnet50_clf_stl10.yaml \
#     --exp-case baseline_nGPU"$GPUS"_multi_lr"$lr"_alltrain_fff \
#     --lr $lr --task-bs 4

#     sleep 5
#     $KILL_PROC
#     $KILL_PROC
#     $KILL_PROC
#     sleep 2
# done


for lr in $LR_LISTS
do
# Cityscape-baseline / bs 4 / lr 0.01
CUDA_VISIBLE_DEVICES=$DEVICES torchrun --nproc_per_node=$1 --master_port=$2 \
    $TRAIN_ROOT/train.py --general \
    --cfg $CFG_PATH/resnet50_seg_cityscape.yaml \
    --exp-case baseline_nGPU"$GPUS"_multi_lr"$lr"_alltrain_ftt \
    --lr $lr --dilation-type fff --allbackbone-train \
    --task-bs 4

sleep 5
$KILL_PROC
$KILL_PROC
$KILL_PROC
sleep 2

done

# # STL10-baseline / bs 4 / lr 0.01
# CUDA_VISIBLE_DEVICES=$DEVICES torchrun --nproc_per_node=$1 --master_port=$2 \
#     $TRAIN_ROOT/single_train.py --general \
#     --cfg $CFG_PATH/resnet50_clf_stl10.yaml \
#     --exp-case aaaaa \
#     --lr 0.01 --dilation-type fff --allbackbone-train --flop-size 96 \
#     --epochs 133 --lr-steps 88 121 --task-bs 4

# sleep 5
# $KILL_PROC
# $KILL_PROC
# $KILL_PROC
# sleep 2

# # STL10-baseline / bs 4 / lr 0.02
# CUDA_VISIBLE_DEVICES=$DEVICES torchrun --nproc_per_node=$1 --master_port=$2 \
#     $TRAIN_ROOT/single_train.py --general \
#     --cfg $CFG_PATH/resnet50_clf_stl10.yaml \
#     --exp-case baseline_nGPU"$GPUS"_multi_lr0.01_alltrain_fff \
#     --lr 0.01 --dilation-type fff --allbackbone-train --flop-size 96 \
#     --epochs 133 --lr-steps 88 121 --task-bs 4

# sleep 5
# $KILL_PROC
# $KILL_PROC
# $KILL_PROC
# sleep 2

# # STL10-baseline / bs 4 / 0.6warmup / alltrain / fff 
# CUDA_VISIBLE_DEVICES=$DEVICES torchrun --nproc_per_node=$1 --master_port=$2 \
#     $TRAIN_ROOT/single_train.py --general \
#     --cfg $CFG_PATH/resnet50_clf_stl10.yaml \
#     --exp-case baseline_nGPU"$GPUS"_multi_lr0.01_alltrain_fff \
#     --lr 0.01 --dilation-type fff --allbackbone-train --flop-size 96 \
#     --epochs 133 --lr-steps 88 121 --task-bs 4

# sleep 5
# $KILL_PROC
# $KILL_PROC
# $KILL_PROC
# sleep 2




