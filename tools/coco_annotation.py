#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json, argparse
from collections import defaultdict
from os import getcwd

sets=[('instances_train2017', 'train2017'), ('instances_val2017', 'val2017')]

parser = argparse.ArgumentParser(description='convert COCO dataset annotation to txt annotation file')
parser.add_argument('--dataset_path', type=str, help='path to MSCOCO dataset, default is ../mscoco2017', default=getcwd()+'/../mscoco2017')
parser.add_argument('--output_path', type=str,  help='output path for generated annotation txt files, default is ./', default='./')
parser.add_argument('--include_no_obj', action="store_true", help='to include no object image', default=False)
args = parser.parse_args()

for dataset, datatype in sets:
    image_annotation_dict = defaultdict(list)
    coco_annotation_file = open("%s/annotations/%s.json"%(args.dataset_path, dataset),
                           encoding='utf-8')
    # annotation_data format:
    # {
    #  "info": info,
    #  "licenses": [license],
    #  "images": [image],
    #  "type": "instances",
    #  "annotations": [annotation],
    #  "categories": [category]
    # }
    annotation_data = json.load(coco_annotation_file)
    annotations = annotation_data['annotations']

    # to include no object image, we need to involve
    # all images to image_annotation_dict
    if args.include_no_obj:
        images = annotation_data['images']
        for image in images:
            # image format:
            # {
            #  "license": int,
            #  "url": "url_string",
            #  "file_name": "name_string",
            #  "height": int,
            #  "width": int,
            #  "date_captured": "date_string",
            #  "id": int
            # }
            image_id = image['id']
            image_file = '%s/%s/%012d.jpg' % (args.dataset_path, datatype, image_id)
            image_annotation_dict[image_file] = []

    for annotation in annotations:
        # annotation format:
        # {
        #  "id": int,
        #  "image_id": int,
        #  "category_id": int,
        #  "segmentation": RLE or [polygon],
        #  "area": float,
        #  "bbox": [x,y,width,height],
        #  "iscrowd": 0 or 1
        # }
        image_id = annotation['image_id']
        image_file = '%s/%s/%012d.jpg' % (args.dataset_path, datatype, image_id)
        category_id = annotation['category_id']

        # since original 80 COCO category_ids is not continuous,
        # we need to align them to continuous id (0~79) for further process
        if category_id >= 1 and category_id <= 11:
            category_id = category_id - 1
        elif category_id >= 13 and category_id <= 25:
            category_id = category_id - 2
        elif category_id >= 27 and category_id <= 28:
            category_id = category_id - 3
        elif category_id >= 31 and category_id <= 44:
            category_id = category_id - 5
        elif category_id >= 46 and category_id <= 65:
            category_id = category_id - 6
        elif category_id == 67:
            category_id = category_id - 7
        elif category_id == 70:
            category_id = category_id - 9
        elif category_id >= 72 and category_id <= 82:
            category_id = category_id - 10
        elif category_id >= 84 and category_id <= 90:
            category_id = category_id - 11

        # merge to image bbox annotations
        image_annotation_dict[image_file].append([annotation['bbox'], category_id])

    # save converting result to our annotation file
    annotation_file = open('%s/%s.txt'%(args.output_path, datatype), 'w')
    for image_file in image_annotation_dict.keys():
        annotation_file.write(image_file)
        box_infos = image_annotation_dict[image_file]
        for box_info in box_infos:
            # bbox format: [xmin, ymin, w, h]
            bbox = box_info[0]
            category_id = box_info[1]
            x_min = int(bbox[0])
            y_min = int(bbox[1])
            x_max = x_min + int(bbox[2])
            y_max = y_min + int(bbox[3])

            box_annotation = " %d,%d,%d,%d,%d" % (
                x_min, y_min, x_max, y_max, int(category_id))
            annotation_file.write(box_annotation)
        annotation_file.write('\n')
    annotation_file.close()
