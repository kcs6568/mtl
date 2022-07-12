#!/bin/bash

PORT=$1
METHOD=$2
START_GPU=$3
NUM_GPU=$4
BACKBONE=$5

# ./s2_run_single2.sh $1 $2 $3 $4 $5
./s2_run_quad.sh $1 $2 $3 $4 $5
# ./s2_run_quad2.sh $1 $2 $3 $4 resnet50
