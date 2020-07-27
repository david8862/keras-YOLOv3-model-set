#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test enhace data argument functions (mosaic/cutmix)
"""
import os, sys, argparse
import numpy as np
from PIL import Image
import cv2

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))
from yolo3.data import get_ground_truth_data
from common.utils import get_dataset, get_classes, draw_label
from common.data_utils import random_mosaic_augment, random_cutmix_augment


def draw_boxes(images, boxes, class_names, output_path):
    for i in range(images.shape[0]):
        img = images[i]
        for j in range(len(boxes[i])):
            # bypass all 0 box
            if (boxes[i][j][:4] == 0).all():
                continue

            x_min = int(boxes[i][j][0])
            y_min = int(boxes[i][j][1])
            x_max = int(boxes[i][j][2])
            y_max = int(boxes[i][j][3])
            cls = int(boxes[i][j][4])

            class_name = class_names[cls]
            color = (255,0,0)  #Red box

            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)
            img = draw_label(img, class_name, color, (x_min, y_min))
        image = Image.fromarray(img)
        image.save(os.path.join(output_path, str(i)+".jpg"))


def main():
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS, description='Test tool for enhance mosaic data augment function')
    parser.add_argument('--annotation_file', type=str, required=True, help='data annotation txt file')
    parser.add_argument('--classes_path', type=str, required=True, help='path to class definitions')
    parser.add_argument('--output_path', type=str, required=False,  help='output path for augmented images, default=%(default)s', default='./test')
    parser.add_argument('--batch_size', type=int, required=False, help = "batch size for test data, default=%(default)s", default=16)
    parser.add_argument('--model_image_size', type=str, required=False, help='model image input size as <height>x<width>, default=%(default)s', default='416x416')
    parser.add_argument('--augment_type', type=str, required=False, help = "enhance data augmentation type (mosaic/cutmix), default=%(default)s", default='mosaic', choices=['mosaic', 'cutmix'])

    args = parser.parse_args()
    class_names = get_classes(args.classes_path)
    height, width = args.model_image_size.split('x')
    model_image_size = (int(height), int(width))
    assert (model_image_size[0]%32 == 0 and model_image_size[1]%32 == 0), 'model_image_size should be multiples of 32'

    annotation_lines = get_dataset(args.annotation_file)
    os.makedirs(args.output_path, exist_ok=True)

    image_data = []
    boxes_data = []
    for i in range(args.batch_size):
        annotation_line = annotation_lines[i]
        image, boxes = get_ground_truth_data(annotation_line, input_shape=model_image_size, augment=True)
        #un-normalize image
        image = image*255.0
        image = image.astype(np.uint8)

        image_data.append(image)
        boxes_data.append(boxes)
    image_data = np.array(image_data)
    boxes_data = np.array(boxes_data)

    if args.augment_type == 'mosaic':
        image_data, boxes_data = random_mosaic_augment(image_data, boxes_data, prob=1)
    elif args.augment_type == 'cutmix':
        image_data, boxes_data = random_cutmix_augment(image_data, boxes_data, prob=1)
    else:
        raise ValueError('Unsupported augment type')

    draw_boxes(image_data, boxes_data, class_names, args.output_path)


if __name__ == "__main__":
    main()

