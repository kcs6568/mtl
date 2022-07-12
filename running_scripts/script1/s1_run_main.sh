#!/bin/bash

PORT=$1
METHOD=$2
START_GPU=$3
NUM_GPU=$4
BACKBONE=$5

# ./s1_run_single.sh $1 single $3 $4 resnet50
# ./s1_run_single.sh $1 single $3 $4 mobilenetv3
./s1_run_quad2.sh $1 $2 $3 $4 $5



# ./running_scripts/run_triple.sh \
#     8 29500 \
#     minicoco_voc/resnet50_clf_det_seg.yaml \
#     threetask_defalut_multi[8,11]_lr0.01

# ./running_scripts/run_triple.sh \
#     8 29500 \
#     minicoco_voc/resnet50_clf_det_seg.yaml \
#     threetask_defalut_cosine_lr0.01