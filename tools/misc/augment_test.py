#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test enhace data argument functions (mosaic/mosaic_v5/cutmix)
"""
import os, sys, argparse
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))
from yolo3.data import get_ground_truth_data
from common.utils import get_dataset, get_classes, draw_label, labelType
from common.data_utils import random_mosaic_augment, random_mosaic_augment_v5, random_cutmix_augment, denormalize_image


def draw_boxes(images, boxes, class_names, output_path):
    for i in range(images.shape[0]):
        img = images[i]
        for j in range(len(boxes[i])):
            # bypass all 0 box
            if (boxes[i][j][:4] == 0).all():
                continue

            xmin = int(boxes[i][j][0])
            ymin = int(boxes[i][j][1])
            xmax = int(boxes[i][j][2])
            ymax = int(boxes[i][j][3])
            cls = int(boxes[i][j][4])

            class_name = class_names[cls]
            color = (255,0,0)  #Red box

            # choose label type according to box size
            if ymin > 20:
                label_coords = (xmin, ymin)
                label_type = label_type=labelType.LABEL_TOP_OUTSIDE
            elif ymin <= 20 and ymax <= img.shape[0] - 20:
                label_coords = (xmin, ymax)
                label_type = label_type=labelType.LABEL_BOTTOM_OUTSIDE
            elif ymax > img.shape[0] - 20:
                label_coords = (xmin, ymin)
                label_type = label_type=labelType.LABEL_TOP_INSIDE

            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 1, cv2.LINE_AA)
            img = draw_label(img, class_name, color, label_coords, label_type)

        image = Image.fromarray(img)
        image.save(os.path.join(output_path, str(i)+".jpg"))


def main():
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS, description='Test tool for enhance mosaic data augment function')
    parser.add_argument('--annotation_file', type=str, required=True, help='data annotation txt file')
    parser.add_argument('--classes_path', type=str, required=True, help='path to class definitions')
    parser.add_argument('--output_path', type=str, required=False,  help='output path for augmented images, default=%(default)s', default='./test')
    parser.add_argument('--batch_size', type=int, required=False, help = "batch size for test data, default=%(default)s", default=16)
    parser.add_argument('--model_input_shape', type=str, required=False, help='model image input shape as <height>x<width>, default=%(default)s', default='416x416')
    parser.add_argument('--enhance_augment', type=str, required=False, help = "enhance data augmentation type, default=%(default)s", default=None, choices=['mosaic', 'mosaic_v5', 'cutmix', None])

    args = parser.parse_args()
    class_names = get_classes(args.classes_path)
    height, width = args.model_input_shape.split('x')
    model_input_shape = (int(height), int(width))
    assert (model_input_shape[0]%32 == 0 and model_input_shape[1]%32 == 0), 'model_input_shape should be multiples of 32'

    annotation_lines = get_dataset(args.annotation_file)
    os.makedirs(args.output_path, exist_ok=True)

    image_data = []
    boxes_data = []

    pbar = tqdm(total=args.batch_size, desc='Generate augment image')
    for i in range(args.batch_size):
        pbar.update(1)
        annotation_line = annotation_lines[i]
        image, boxes = get_ground_truth_data(annotation_line, input_shape=model_input_shape, augment=True)
        #denormalize image
        image = denormalize_image(image)

        image_data.append(image)
        boxes_data.append(boxes)
    pbar.close()
    image_data = np.array(image_data)
    boxes_data = np.array(boxes_data)

    if args.enhance_augment == 'mosaic':
        image_data, boxes_data = random_mosaic_augment(image_data, boxes_data, prob=1)
    elif args.enhance_augment == 'mosaic_v5':
        image_data, boxes_data = random_mosaic_augment_v5(image_data, boxes_data, prob=1)
    elif args.enhance_augment == 'cutmix':
        image_data, boxes_data = random_cutmix_augment(image_data, boxes_data, prob=1)
    elif args.enhance_augment == None:
        print('No enhance augment type. Will only apply base augment')
    else:
        raise ValueError('Unsupported augment type')

    draw_boxes(image_data, boxes_data, class_names, args.output_path)
    print('Done. augment images have been saved in', args.output_path)


if __name__ == "__main__":
    main()

