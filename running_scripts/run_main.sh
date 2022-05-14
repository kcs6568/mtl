#!/bin/bash

PORT=29500
METHOD=$1

# ./run_single.sh 8 29500
# ./run_quad.sh 8 29500 $METHOD
./run_penta.sh 8 29500 $METHOD


# ./running_scripts/run_triple.sh \
#     8 29500 \
#     minicoco_voc/resnet50_clf_det_seg.yaml \
#     threetask_defalut_multi[8,11]_lr0.01

# ./running_scripts/run_triple.sh \
#     8 29500 \
#     minicoco_voc/resnet50_clf_det_seg.yaml \
#     threetask_defalut_cosine_lr0.01