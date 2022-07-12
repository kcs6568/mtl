#!/bin/bash

GPU=$1
SAVE_NAME=$2
# METHOD=$2
# BACKBONE=$3


# if [ $3 = resnet50 ]
# then
#     YAML_CFG=resnet50_clf_det_seg.yaml    

# elif [ $3 = resnext50 ]
# then
#     YAML_CFG=resnext50_32x4d_clf_det_seg.yaml

# elif [ $3 = mobilenetv3 ]
# then
#     YAML_CFG=mobile_v3_large_clf_det_seg_2.yaml
# else
#     echo Not supported backbone
# fi

python3 ../inference.py \
    --yaml_cfg /root/src/mtl_cl/cfgs/inference.yaml \
    --gpu $1 --save_name $2

    