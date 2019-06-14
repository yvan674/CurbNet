#!/bin/bash
# Used to train the network without having to input all the arguments every
# time.

DATE=$(date +"%d-%m")
WEIGHTS="/home/satyaway/Documents/Thesis/Weights/mce-14-jun.pth"
DATA="/tmp/mapillary-vistas-dataset_public_v1.1/training"
LR=0.0005
OPTIMIZER="adam"
BATCH=32
EPOCH=10
PLOT="/home/satyaway/Documents/Thesis/Logs"
NETWORK=d
PRETRAINED=""
PX_COORDINATES='-x'

python3 main.py ${WEIGHTS} -t ${DATA} -r ${LR} -o ${OPTIMIZER} -b ${BATCH} -e ${EPOCH} -a -p ${PLOT} -c -n ${NETWORK} ${PRETRAINED} ${PX_COORDINATES}
