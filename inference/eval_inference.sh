#!/bin/bash

if [[ "$#" -ne 5 ]] && [[ "$#" -ne 6 ]]; then
    echo "Usage: $0 <model_file> <image_path> <anchor_file> <class_file> <result_file> [conf_thrd=0.1]"
    exit 1
fi

MODEL_FILE=$1
IMAGE_PATH=$2
ANCHOR_FILE=$3
CLASS_FILE=$4
RESULT_FILE=$5

if [ "$#" -eq 6 ]; then
    CONF_THRD=$6
else
    CONF_THRD=0.1
fi

IMAGE_LIST=$(ls $IMAGE_PATH)
IMAGE_NUM=$(ls $IMAGE_PATH | wc -l)

# prepare process bar
i=0
ICON_ARRAY=("\\" "|" "/" "-")

# clean result file first
rm -rf $RESULT_FILE

for IMAGE in $IMAGE_LIST
do
    ./yoloDetection -m $MODEL_FILE -i $IMAGE_PATH"/"$IMAGE -a $ANCHOR_FILE -l $CLASS_FILE -n $CONF_THRD -r $RESULT_FILE -t 4 -c 1 -w 1 2>&1 >> /dev/null
    # update process bar
    let index=i%4
    let percent=i*100/IMAGE_NUM
    let num=percent/2
    bar=$(seq -s "#" $num | tr -d "[:digit:]")
    #printf "inference process: %d/%d [%c]\r" "$i" "$IMAGE_NUM" "${ICON_ARRAY[$index]}"
    printf "inference process: %d/%d [%-50s] %d%% \r" "$i" "$IMAGE_NUM" "$bar" "$percent"
    let i=i+1
done
printf "\nDone\n"
