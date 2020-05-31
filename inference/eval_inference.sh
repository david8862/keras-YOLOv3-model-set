#!/bin/bash

if [ "$#" -ne 6 ]; then
    echo "Usage: $0 <model_file> <image_path> <anchor_file> <class_file> <conf_thrd> <result_file>"
    exit 1
fi

MODEL_FILE=$1
IMAGE_PATH=$2
ANCHOR_FILE=$3
CLASS_FILE=$4
CONF_THRD=$5
RESULT_FILE=$6

IMAGE_LIST=$(ls $IMAGE_PATH)
IMAGE_NUM=$(ls $IMAGE_PATH | wc -l)

#prepare process bar
i=0
ICON_ARRAY=("\\" "|" "/" "-")


for IMAGE in $IMAGE_LIST
do
    ./yoloDetection -m $MODEL_FILE -i $IMAGE_PATH"/"$IMAGE -a $ANCHOR_FILE -l $CLASS_FILE -n $CONF_THRD -r $RESULT_FILE -t 4 -c 1 -w 1 2>&1 >> /dev/null
    #update process bar
    let index=i%4
    printf "inference process: %d/%d [%c]\r" "$i" "$IMAGE_NUM" "${ICON_ARRAY[$index]}"
    let i=i+1
done

