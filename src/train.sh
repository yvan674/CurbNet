#!/bin/bash
# Used to train the network without having to input all the arguments every
# time.

DATE=$(date +"%d-%m")
WEIGHTS="/home/satyaway/Documents/Thesis/Weights/deeplab.pth"
DATA="/tmp/mapillary/"
LR=0.005
OPTIMIZER="adam"
BATCH=16
EPOCH=10
PLOT="/home/satyaway/Documents/Thesis/Logs"
NETWORK=d

python3 main.py ${WEIGHTS} -t ${DATA} -r ${LR} -o ${OPTIMIZER} -b ${BATCH} -e ${EPOCH} -a -p ${PLOT} -c -n ${NETWORK}
