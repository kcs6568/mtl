#!/bin/bash

GPUS=$1
PORT=$2
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
    - default / multi / 0.01: 95.19

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
        - multi / 0.02 / aux: 

        - cosine / 0.01 / aux: 89.1
        - cosine / 0.02 / aux: 

END



# CUDA_VISIBLE_DEVICES=$DEVICES torchrun --nproc_per_node=$1 --master_port=$2 \
#     ../single_train.py --general \
#     --cfg $CFG_PATH/resnet50_det_minicoco.yaml \
#     --exp-case nGPU"$GPUS"_cosine_lr0.01_FreezeBN_alltrain_dil[ftt] \
#     --lr 0.01 --use-minids --allbackbone-train --freeze-bn --dilation-type ftt \
#     --lr-scheduler cosine --resume

# sleep 5
# $KILL_PROC
# $KILL_PROC
# $KILL_PROC
# sleep 2

# CUDA_VISIBLE_DEVICES=$DEVICES torchrun --nproc_per_node=$1 --master_port=$2 \
#     ../single_train.py --general \
#     --cfg $CFG_PATH/resnet50_det_minicoco.yaml \
#     --exp-case nGPU"$GPUS"_multi_lr0.01_FreezeBN_alltrain_dil[ftt] \
#     --lr 0.01 --use-minids --allbackbone-train --freeze-bn --dilation-type ftt 

# sleep 5
# $KILL_PROC
# $KILL_PROC
# $KILL_PROC
# sleep 2

# # 97 epoch multi
# CUDA_VISIBLE_DEVICES=$DEVICES torchrun --nproc_per_node=$1 --master_port=$2 \
#     ../single_train.py --general \
#     --cfg $CFG_PATH/resnet50_seg_97e_voc0712.yaml \
#     --exp-case nGPU"$GPUS"_multi_lr0.01_FreezeBN_alltrain_97e_lrdecay \
#     --lr 0.01 --warmup-ratio 1 --allbackbone-train --freeze-bn \
#     --dilation-type ftt 
    
# $KILL_PROC
# $KILL_PROC
# $KILL_PROC
# sleep 2


# # 97 epoch cosine
# CUDA_VISIBLE_DEVICES=$DEVICES torchrun --nproc_per_node=$1 --master_port=$2 \
#     ../single_train.py --general \
#     --cfg $CFG_PATH/resnet50_seg_97e_voc0712.yaml \
#     --exp-case nGPU"$GPUS"_cosine_lr0.01_FreezeBN_alltrain_97e_lrdecay \
#     --lr 0.01 --warmup-ratio 1 --allbackbone-train --freeze-bn --lr-scheduler cosine \
#     --dilation-type ftt 
    
# $KILL_PROC
# $KILL_PROC
# $KILL_PROC
# sleep 2


CUDA_VISIBLE_DEVICES=$DEVICES torchrun --nproc_per_node=$1 --master_port=$2 \
    ../single_train.py --general \
    --cfg $CFG_PATH/resnet50_clf_cifar10.yaml \
    --exp-case nGPU"$GPUS"_cosine_lr0.01_FreezeBN_alltrain_dil[ftt] \
    --lr 0.01 --use-minids --allbackbone-train --freeze-bn --lr-scheduler cosine \
    --dilation-type ftt 

sleep 5
$KILL_PROC
$KILL_PROC
$KILL_PROC
sleep 2


# 12 epoch
# CUDA_VISIBLE_DEVICES=$DEVICES torchrun --nproc_per_node=$1 --master_port=$2 \
#     ../single_train.py \
#     --cfg $CFG_PATH/resnet50_seg_voc0712.yaml \
#     --exp-case default_"nGPU$GPUS"_multi_lr0.01_FreezeBN_alltrain \
#     --lr 0.01 --warmup-ratio 1 --allbackbone-train --freeze-bn

# $KILL_PROC
# $KILL_PROC
# $KILL_PROC
# sleep 2

# CUDA_VISIBLE_DEVICES=$DEVICES torchrun --nproc_per_node=$1 --master_port=$2 \
#     ../single_train.py \
#     --cfg $CFG_PATH/resnet50_seg_voc0712.yaml \
#     --exp-case default_"nGPU$GPUS"_cosine_lr0.01_FreezeBN_alltrain \
#     --lr 0.01 --warmup-ratio 1 --allbackbone-train --freeze-bn --lr-scheduler cosine

# $KILL_PROC
# $KILL_PROC
# $KILL_PROC
# sleep 2

# CUDA_VISIBLE_DEVICES=$DEVICES torchrun --nproc_per_node=$1 --master_port=$2 \
#     ../single_train.py \
#     --cfg $CFG_PATH/resnet50_seg_voc0712.yaml \
#     --exp-case default_"nGPU$GPUS"_multi_lr0.02_FreezeBN_alltrain \
#     --lr 0.02 --warmup-ratio 1 --allbackbone-train --freeze-bn

# $KILL_PROC
# $KILL_PROC
# $KILL_PROC
# sleep 2

# CUDA_VISIBLE_DEVICES=$DEVICES torchrun --nproc_per_node=$1 --master_port=$2 \
#     ../single_train.py \
#     --cfg $CFG_PATH/resnet50_seg_voc0712.yaml \
#     --exp-case default_"nGPU$GPUS"_cosine_lr0.02_FreezeBN_alltrain \
#     --lr 0.02 --warmup-ratio 1 --allbackbone-train --freeze-bn --lr-scheduler cosine

# $KILL_PROC
# $KILL_PROC
# $KILL_PROC
# sleep 2

# ########################################################################################

# # 145 epoch
# CUDA_VISIBLE_DEVICES=$DEVICES torchrun --nproc_per_node=$1 --master_port=$2 \
#     ../single_train.py \
#     --cfg $CFG_PATH/resnet50_seg_145e_voc0712.yaml \
#     --exp-case default_"nGPU$GPUS"_multi_lr0.01_FreezeBN_alltrain_145e_lrdecay[96,144] \
#     --lr 0.01 --warmup-ratio 1 --allbackbone-train --freeze-bn 

# $KILL_PROC
# $KILL_PROC
# $KILL_PROC
# sleep 2

# CUDA_VISIBLE_DEVICES=$DEVICES torchrun --nproc_per_node=$1 --master_port=$2 \
#     ../single_train.py \
#     --cfg $CFG_PATH/resnet50_seg_145e_voc0712.yaml \
#     --exp-case default_"nGPU$GPUS"_cosine_lr0.01_FreezeBN_alltrain_145e_lrdecay[12...] \
#     --lr 0.01 --warmup-ratio 1 --allbackbone-train --freeze-bn --lr-scheduler cosine

# $KILL_PROC
# $KILL_PROC
# $KILL_PROC
# sleep 2

# CUDA_VISIBLE_DEVICES=$DEVICES torchrun --nproc_per_node=$1 --master_port=$2 \
#     ../single_train.py \
#     --cfg $CFG_PATH/resnet50_clf_cifar10.yaml \
#     --exp-case default_"nGPU$GPUS"_multi_lr0.01_FreezeBN_alltrain \
#     --lr 0.01 --warmup-ratio 1 --allbackbone-train --freeze-bn

# $KILL_PROC
# $KILL_PROC
# $KILL_PROC
# sleep 2

# CUDA_VISIBLE_DEVICES=$DEVICES torchrun --nproc_per_node=$1 --master_port=$2 \
#     ../single_train.py \
#     --cfg $CFG_PATH/resnet50_det_minicoco.yaml \
#     --exp-case default_"nGPU$GPUS"_multi_lr0.01_FreezeBN_alltrain \
#     --lr 0.01 --warmup-ratio 1 --allbackbone-train --freeze-bn

# $KILL_PROC
# $KILL_PROC
# $KILL_PROC
# sleep 2

# CUDA_VISIBLE_DEVICES=$DEVICES torchrun --nproc_per_node=$1 --master_port=$2 \
#     ../single_train.py \
#     --cfg $CFG_PATH/resnet50_seg_145e_voc0712.yaml \
#     --exp-case default_"nGPU$GPUS"_multi_lr0.02_FreezeBN_alltrain_145e_lrdecay[96,144] \
#     --lr 0.02 --warmup-ratio 1 --allbackbone-train --freeze-bn 

# $KILL_PROC
# $KILL_PROC
# $KILL_PROC
# sleep 2

# CUDA_VISIBLE_DEVICES=$DEVICES torchrun --nproc_per_node=$1 --master_port=$2 \
#     ../single_train.py \
#     --cfg $CFG_PATH/resnet50_seg_145e_voc0712.yaml \
#     --exp-case default_"nGPU$GPUS"_cosine_lr0.02_FreezeBN_alltrain_145e_lrdecay[12...] \
#     --lr 0.02 --warmup-ratio 1 --allbackbone-train --freeze-bn --lr-scheduler cosine

# $KILL_PROC
# $KILL_PROC
# $KILL_PROC
# sleep 2






