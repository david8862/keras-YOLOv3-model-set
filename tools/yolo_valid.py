"""
A MSCOCO eval script for YOLOv3 keras implementation, got from
https://github.com/muyiguangda/tensorflow-keras-yolov3/blob/master/yolo_valid.py

Created on April, 2019
@authors: Hulking
"""

import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import argparse
from yolo import YOLO, detect_video
from PIL import Image
from pycocoEval import cal_coco_map

import glob
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input as preprocess
from keras import backend as K

np.set_printoptions(suppress=True)  # to supress scientific notation in print

from functools import wraps

'''
计算全部图片检测结果
'''
def valid_detector(yolo):
    classes_path="model_data/coco_classes.txt"
    with open(classes_path) as f:
        obj_list = f.readlines()
        ## remove whitespace characters like `\n` at the end of each line
        obj_list = [x.strip() for x in obj_list]

    coco_ids= [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32,
               33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
               59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88,
               89, 90]

    imglist_path="model_data/5k.txt"
    dt_result_path = "results/cocoapi_results.json"

    if os.path.exists(dt_result_path):
        os.remove(dt_result_path)
    with open(dt_result_path, "a") as new_p:
        new_p.write("[")
        with open(imglist_path) as f:
            total_img_list = f.readlines()
            total_img_list = [x.strip() for x in total_img_list]
            total_img_num = len(total_img_list)
            i=0
            for image_path in total_img_list:

                if (os.path.exists(image_path)):
                    print(i,image_path)

                orig_index=int(image_path[50:56])
                img = Image.open(image_path)

                boxes, scores, classes =yolo.valid_image(img)

                for j in range(len(classes)):
                    coco_id=coco_ids[int(classes[j])]
                    top, left, bottom, right=boxes[j]

                    width=round(right-left,4)
                    height=round(bottom-top,4)

                    # print("\ni, j ",i,j)
                    # print("\nleft, top,  width, height ",left, top, width, height)

                    if i==(total_img_num-1) and j== (len(classes)-1):
                        new_p.write(
                            "{\"image_id\":" + str(orig_index) + ", \"category_id\":" + str(coco_id) + ", \"bbox\":[" + \
                            str(left) + ", " + str(top) + ", " + str(width) + ", " + str(height) + "], \"score\":" + str(scores[j]) + "}")
                    else:
                    #print("corrected left, top, width, height", left, top, width, height)
                        new_p.write(
                            "{\"image_id\":"+str(orig_index)+", \"category_id\":" +str(coco_id)+ ", \"bbox\":[" + \
                                str(left)+ ", " + str(top) + ", " + str(width) + ", " + str(height) + "], \"score\":"+str(scores[j]) +"},\n")
                i += 1
            new_p.write("]")
        print("\n\n\n")

if __name__=='__main__':
    valid_detector(YOLO())
    cal_coco_map()
