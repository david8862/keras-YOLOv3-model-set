#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, argparse
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--image_path', type=str, required=True, help='path to image files')
parser.add_argument('--bbox_info_file', type=str, required=True, help='csv bounding box info file')
parser.add_argument('--output_file', type=str,  help='generated annotation txt files, default=%(default)s', default='OpenImage.txt')

args = parser.parse_args()

# parse the bouding box info file
bbox_info_file = open(args.bbox_info_file)
lines = bbox_info_file.readlines()
# Bounding box info file columns, should be like:
# ['ImageID', 'Source', 'LabelName', 'Confidence', 'XMin', 'XMax', 'YMin', 'YMax', 'IsOccluded', 'IsTruncated', 'IsGroupOf', 'IsDepiction', 'IsInside']
bbox_items = lines[0].strip().split(',')

bbox_lines = lines[1:]
bbox_lines = [bbox_line.strip().split(',') for bbox_line in bbox_lines]

# get the box we need, now it's "FootWare" = "/m/09j5n"
footware_lines = [bbox_line for bbox_line in bbox_lines if bbox_line[bbox_items.index('LabelName')] == '/m/09j5n']


# convert box coordinate
gt_dict = {}
for footware_line in footware_lines:
    image_file = os.path.join(args.image_path, footware_line[0] + '.jpg')
    image = Image.open(image_file)
    width, height = image.size

    xmin = int(float(footware_line[bbox_items.index('XMin')]) * width)
    xmax = int(float(footware_line[bbox_items.index('XMax')]) * width)
    ymin = int(float(footware_line[bbox_items.index('YMin')]) * height)
    ymax = int(float(footware_line[bbox_items.index('YMax')]) * height)
    box = (xmin, ymin, xmax, ymax)

    if image_file in gt_dict:
        gt_dict[image_file].append(box)
    else:
        gt_dict[image_file] = list([box])


# save to annotation file, use 0 as footware class id
cls_id = 0
annotation_file = open(args.output_file, 'w')
for image_file, boxes in gt_dict.items():
    annotation_file.write(image_file)
    for box in boxes:
        annotation_file.write(" " + ",".join([str(value) for value in box]) + ',' + str(cls_id))
    annotation_file.write('\n')

annotation_file.close()

