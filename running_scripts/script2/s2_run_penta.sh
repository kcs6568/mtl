#!/bin/bash

GPUS=$1
PORT=$2
METHOD=$3
TRAIN_ROOT=/root/src/mtl_cl/
./run_penta.sh 2 29500 $METHOD
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

# echo $DEVICES
# exit 1

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

SCH="cosine"
OPT="adamw"
LR="1e-4 1e-5"
GAMMA="0.1 0.25"

# for sch in $SCH
# do
#     # echo $sch
#     for opt in $OPT
#     do
#         # echo $opt
#         for lr in $LR
#         do
#             # echo $lr
#             for gamma in $GAMMA
#             do
#                 # echo $gamma
#                 exp_case=baseline_nGPU"$GPUS"_"$sch"_"$opt"_lr"$lr"_nowarmup_gamma"$gamma"_lossbal[1,1,6,1,1]_hyp-test
#                 # echo $exp_case
#                 # exit 1

#                 CUDA_VISIBLE_DEVICES=$DEVICES torchrun --nproc_per_node=$1 --master_port=$2 \
#                 $TRAIN_ROOT/train.py --lossbal --loss-ratio 1 1 6 1 1 \
#                 --cfg $CFG_PATH/cifar10_stl10_minicoco_voc_city/resnet50_clf_det_seg.yaml \
#                 --exp-case $exp_case \
#                 --lr-scheduler $sch --opt $opt --lr $lr --lr-gamma $gamma \
#                 --task-bs 4 4 2 2 2 --warmup-ratio -1 --workers 8 --find-epoch 3
            
#                 sleep 5
#                 $KILL_PROC
#                 $KILL_PROC
#                 $KILL_PROC
#                 sleep 2
#             done
#         done
#     done
# done


CUDA_VISIBLE_DEVICES=$DEVICES torchrun --nproc_per_node=$1 --master_port=$2 \
    $TRAIN_ROOT/train.py --lossbal --loss-ratio 1 1 6 1 1 \
    --cfg $CFG_PATH/cifar10_stl10_minicoco_voc_city/resnet50_se_clf_det_seg.yaml \
    --exp-case baseline_nGPU"$GPUS"_multi_adamw_lr1e-4_nowarmup_gamma0.25_lossbal[1,1,6,1,1]_addSE \
    --lr-scheduler multi --opt adamw --lr 1e-4 --lr-gamma 0.25 \
    --task-bs 4 4 2 2 2 --warmup-ratio 1 --workers 8
 
sleep 5
$KILL_PROC
$KILL_PROC
$KILL_PROC
sleep 2

# CUDA_VISIBLE_DEVICES=$DEVICES torchrun --nproc_per_node=$1 --master_port=$2 \
#     $TRAIN_ROOT/train.py --lossbal --loss-ratio 1 1 6 1 1 \
#     --cfg $CFG_PATH/cifar10_stl10_minicoco_voc_city/resnet50_clf_det_seg.yaml \
#     --exp-case baseline_nGPU"$GPUS"_multi_adamw_lr1e-3_nowarmup_gamma0.1_lossbal[1,1,6,1,1] \
#     --lr-scheduler multi --opt adamw --lr 1e-3 --lr-gamma 0.1 \
#     --task-bs 4 4 2 2 2 --warmup-ratio 1 --workers 8
 
# sleep 5
# $KILL_PROC
# $KILL_PROC
# $KILL_PROC
# sleep 2


# CUDA_VISIBLE_DEVICES=$DEVICES torchrun --nproc_per_node=$1 --master_port=$2 \
#     $TRAIN_ROOT/train.py --lossbal --loss-ratio 1 1 6 1 1 \
#     --cfg $CFG_PATH/cifar10_stl10_minicoco_voc_city/resnet50_clf_det_seg.yaml \
#     --exp-case baseline_nGPU"$GPUS"_multi_adamw_lr1e-4_nowarmup_gamma0.1_lossbal[1,1,6,1,1]_tasklr \
#     --lr-scheduler multi --opt adamw --lr 1e-4 --lr-gamma 0.1 \
#     --task-bs 4 4 2 2 2 --warmup-ratio -1 --workers 8
 
# sleep 5
# $KILL_PROC
# $KILL_PROC
# $KILL_PROC
# sleep 2







