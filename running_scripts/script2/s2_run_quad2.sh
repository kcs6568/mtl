#!/bin/bash

PORT=$1
METHOD=$2
START_GPU=$3
NUM_GPU=$4
BACKBONE=$5
TRAIN_ROOT=/root/src/mtl_cl/
CFG_PATH=/root/src/mtl_cl/cfgs/four_task/$2
KILL_PROC="kill $(ps aux | grep train2.py | grep -v grep | awk '{print $2}')"
TRAIN_SCRIPT=$TRAIN_ROOT/train2.py

# $KILL_PROC
# exit 1

# make visible devices order automatically
DEVICES=""
d=$(($3-$4))
for ((n=$3; n>$d; n--))
do
    # $n > 0
    if [ $n -lt 0 ]; then 
        echo The gpu number $n is not valid. START_GPU: $3 / NUM_GPU: $4
        exit 1
    else
        DEVICES+=$n
        # $n < ~
        if [ $n -gt $(($d + 1)) ]
        then
            DEVICES+=,
        fi
    fi
done

: <<'END'
<CIFAR10-STL10-minicoco-voc>
 - SGD
    
        
<CIFAR100-minicoco-voc>
 - SGD

### ONGOING ###



### To Do ###

END

if [ $5 = resnet50 ]
then
    YAML_CFG=resnet50_clf_det_seg.yaml    

elif [ $5 = mobilenetv3 ]
then
    YAML_CFG=mobile_v3_large_clf_det_seg.yaml
else
    echo Not supported backbone
fi

SCH="multi "
OPT="adamw"
LR="1e-4"
GAMMA="0.1 0.25"
ADD_DISC="clip1_ratio[1,1,6,2]_sep-train"

for sch in $SCH
do
    # 1echo $sch
    for opt in $OPT
    do
        for lr in $LR
        do
            for gamma in $GAMMA
            do
                $KILL_PROC
                $KILL_PROC
                $KILL_PROC
                $KILL_PROC
                $KILL_PROC

                exp_case=nGPU"$4"_"$sch"_"$opt"_lr"$lr"
                
                if [ $sch != "cosine" ]
                then
                    exp_case="$exp_case"_gamma"$gamma"_$ADD_DISC
                else
                    exp_case="$exp_case"_$ADD_DISC
                fi

                CUDA_VISIBLE_DEVICES=$DEVICES torchrun --nproc_per_node=$4 --master_port=$1 \
                    $TRAIN_SCRIPT --lossbal \
                    --cfg $CFG_PATH/cifar10_stl10_minicoco_voc/$YAML_CFG \
                    --warmup-ratio -1 --workers 8 \
                    --exp-case $exp_case \
                    --lr-scheduler $sch --opt $opt --lr $lr --gamma $gamma --resume


                $KILL_PROC
                $KILL_PROC
                $KILL_PROC
                $KILL_PROC
                $KILL_PROC

                sleep 5

                if [ $sch == "cosine" ]
                then
                    break
                fi

            done
        done
    done
done


