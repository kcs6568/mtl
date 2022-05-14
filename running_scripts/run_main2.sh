#!/bin/bash

PORT=6000

# ./run_multi.sh 8 6000 
./run_single2.sh 8 6000 
# ./run_triple.sh 8 6000


# ./running_scripts/run_triple.sh \
#     8 6000 \
#     minicoco_voc/resnet50_clf_det_seg.yaml \
#     threetask_defalut_multi[8,11]_lr0.01

# ./running_scripts/run_triple.sh \
#     8 6000 \
#     minicoco_voc/resnet50_clf_det_seg.yaml \
#     threetask_defalut_cosine_lr0.01