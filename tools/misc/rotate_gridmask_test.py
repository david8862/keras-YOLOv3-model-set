#!/usr/bin/python3
# -*- coding=utf-8 -*-
"""
test random rotate or gridmask augment for image and boxes
"""
import os, sys
import numpy as np
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))
from common.data_utils import random_rotate, random_gridmask
from common.utils import get_classes, get_colors, draw_boxes


def main():
    # load some VOC2007 images for test
    #image = Image.open("000001.jpg").convert('RGB')
    #boxes = np.array([[48,240,195,371,11],[8,12,352,498,14]])

    image = Image.open("000004.jpg").convert('RGB')
    boxes = np.array([[13,311,84,362,6],[362,330,500,389,6],[235,328,334,375,6],[175,327,252,364,6],[139,320,189,359,6],[108,325,150,353,6],[84,323,121,350,6]])

    #image = Image.open("000010.jpg").convert('RGB')
    #boxes = np.array([[87,97,258,427,12],[133,72,245,284,14]])


    classes = boxes[:, -1]
    scores = [1.0]*len(classes)

    class_names = get_classes('../../configs/voc_classes.txt')
    colors = get_colors(len(class_names))


    image_origin = draw_boxes(np.array(image, dtype='uint8'), boxes[:, :4], classes, scores, class_names, colors)

    # choose rotate or gridmask augment
    #image, boxes = random_rotate(image, boxes, prob=1.0)
    image, boxes = random_gridmask(image, boxes, prob=1.0)

    if len(boxes) > 0:
        classes = boxes[:, -1]
        scores = [1.0]*len(classes)
        boxes = boxes[:, :4]

    image = draw_boxes(np.array(image, dtype='uint8'), boxes, classes, scores, class_names, colors)

    Image.fromarray(image_origin).show()
    Image.fromarray(image).show()


if __name__ == "__main__":
    main()
