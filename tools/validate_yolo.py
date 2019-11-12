#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
from PIL import Image
import os, sys, argparse
import numpy as np
from tensorflow.keras.models import load_model

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
from yolo3.postprocess_np import yolo3_postprocess_np
from yolo2.postprocess_np import yolo2_postprocess_np
from common.data_utils import preprocess_image
from common.utils import get_classes, get_anchors, get_colors, draw_boxes


def validate_yolo_model(model, image_file, anchors, class_names, model_image_size, loop_count):
    image = Image.open(image_file)
    image_array = np.array(image, dtype='uint8')
    image_data = preprocess_image(image, model_image_size)
    image_shape = image.size

    # predict once first to bypass the model building time
    model.predict([image_data])

    start = time.time()
    for i in range(loop_count):
        if len(anchors) == 5:
            # YOLOv2 use 5 anchors
            boxes, classes, scores = yolo2_postprocess_np(model.predict([image_data]), image_shape, anchors, len(class_names), model_image_size)
        else:
            boxes, classes, scores = yolo3_postprocess_np(model.predict([image_data]), image_shape, anchors, len(class_names), model_image_size)
    end = time.time()

    print('Found {} boxes for {}'.format(len(boxes), image_file))

    for box, cls, score in zip(boxes, classes, scores):
        print("Class: {}, Score: {}".format(class_names[cls], score))

    colors = get_colors(class_names)
    image_array = draw_boxes(image_array, boxes, classes, scores, class_names, colors)
    print("Average Inference time: {:.8f}s".format((end - start)/loop_count))

    Image.fromarray(image_array).show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', help='model file to predict', type=str, required=True)
    parser.add_argument('--image_file', help='image file to predict', type=str, required=True)
    parser.add_argument('--anchors_path',help='path to anchor definitions', type=str, required=True)
    parser.add_argument('--classes_path', help='path to class definitions, default ../configs/voc_classes.txt', type=str, default='../configs/voc_classes.txt')
    parser.add_argument('--model_image_size', help='model image input size as <num>x<num>, default 416x416', type=str, default='416x416')
    parser.add_argument('--loop_count', help='loop inference for certain times', type=int, default=1)

    args = parser.parse_args()

    # param parse
    model = load_model(args.model_path, compile=False)
    anchors = get_anchors(args.anchors_path)
    class_names = get_classes(args.classes_path)
    height, width = args.model_image_size.split('x')
    model_image_size = (int(height), int(width))

    validate_yolo_model(model, args.image_file, anchors, class_names, model_image_size, args.loop_count)


if __name__ == '__main__':
    main()
