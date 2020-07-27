#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A simple script to pick out PascalVOC object from COCO annotation dataset
"""
import os, sys, argparse
import numpy as np
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))
from common.utils import get_classes, get_dataset


def main():
    parser = argparse.ArgumentParser(description='Pick out VOC object from COCO annotation dataset')
    parser.add_argument('--coco_annotation_file', type=str, required=True,
        help='coco annotation txt file')
    parser.add_argument('--coco_classes_path', type=str, default='../../configs/coco_classes.txt',
        help='path to coco class definitions, default=%(default)s')
    parser.add_argument('--voc_classes_path', type=str, default='../../configs/voc_classes.txt',
        help='path to voc class definitions, default=%(default)s')
    parser.add_argument('--output_voc_annotation_file', type=str, required=True,
        help='output voc classes annotation file')

    args = parser.parse_args()

    # param parse
    coco_class_names = get_classes(args.coco_classes_path)
    voc_class_names = get_classes(args.voc_classes_path)
    coco_annotation_lines = get_dataset(args.coco_annotation_file)

    output_file = open(args.output_voc_annotation_file, 'w')
    for coco_annotation_line in coco_annotation_lines:
        # parse annotation line
        coco_line = coco_annotation_line.split()
        image_name = coco_line[0]
        boxes = np.array([np.array(list(map(int,box.split(',')))) for box in coco_line[1:]])

        has_voc_object = False
        for box in boxes:
            coco_class_id = box[-1]
            # check if coco object in voc class list
            # if true, keep the image & box info
            if coco_class_names[coco_class_id] in voc_class_names:
                if has_voc_object == False:
                    has_voc_object = True
                    output_file.write(image_name)
                # get VOC class ID of the COCO object
                voc_class_id = voc_class_names.index(coco_class_names[coco_class_id])
                output_file.write(" " + ",".join([str(b) for b in box[:-2]]) + ',' + str(voc_class_id))

        if has_voc_object == True:
            output_file.write('\n')
    output_file.close()


if __name__ == '__main__':
    main()
