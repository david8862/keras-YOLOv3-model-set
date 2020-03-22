#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import xml.etree.ElementTree as ET
import numpy as np
import os, argparse
from os import getcwd

sets=[('2007', 'train'), ('2007', 'val'), ('2007', 'test'), ('2012', 'train'), ('2012', 'val')]
classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

class_count = {}

def convert_annotation(dataset_path, year, image_id, list_file, include_difficult):
    in_file = open('%s/VOC%s/Annotations/%s.xml'%(dataset_path, year, image_id))
    tree=ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = obj.find('difficult')
        if difficult is None:
            difficult = '0'
        else:
            difficult = difficult.text
        cls = obj.find('name').text
        if cls not in classes:
            continue
        if not include_difficult and int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))
        class_count[cls] = class_count[cls] + 1


def has_object(dataset_path, year, image_id, include_difficult):
    try:
        in_file = open('%s/VOC%s/Annotations/%s.xml'%(dataset_path, year, image_id))
    except:
        # bypass image if no annotation
        return False
    tree=ET.parse(in_file)
    root = tree.getroot()
    count = 0

    for obj in root.iter('object'):
        difficult = obj.find('difficult')
        if difficult is None:
            difficult = '0'
        else:
            difficult = difficult.text
        cls = obj.find('name').text
        if cls not in classes:
            continue
        if not include_difficult and int(difficult)==1:
            continue
        count = count +1
    return count != 0


def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        classes = f.readlines()
    classes = [c.strip() for c in classes]
    return classes


parser = argparse.ArgumentParser(description='convert PascalVOC dataset annotation to txt annotation file')
parser.add_argument('--dataset_path', type=str, help='path to PascalVOC dataset, default is ../VOCdevkit', default=getcwd()+'/../VOCdevkit')
parser.add_argument('--output_path', type=str,  help='output path for generated annotation txt files, default is ./', default='./')
parser.add_argument('--classes_path', type=str, required=False, help='path to class definitions')
parser.add_argument('--include_difficult', action="store_true", help='to include difficult object', default=False)
parser.add_argument('--include_no_obj', action="store_true", help='to include no object image', default=False)
args = parser.parse_args()

# update class names
if args.classes_path:
    classes = get_classes(args.classes_path)

# get real path for dataset
dataset_realpath = os.path.realpath(args.dataset_path)

for year, image_set in sets:
    # count class item number in each set
    class_count = {itm: 0 for itm in classes}

    image_ids = open('%s/VOC%s/ImageSets/Main/%s.txt'%(dataset_realpath, year, image_set)).read().strip().split()
    list_file = open('%s/%s_%s.txt'%(args.output_path, year, image_set), 'w')
    for image_id in image_ids:
        if has_object(dataset_realpath, year, image_id, args.include_difficult):
            list_file.write('%s/VOC%s/JPEGImages/%s.jpg'%(dataset_realpath, year, image_id))
            convert_annotation(dataset_realpath, year, image_id, list_file, args.include_difficult)
            list_file.write('\n')
        elif args.include_no_obj:
            # include no object image. just write file path
            list_file.write('%s/VOC%s/JPEGImages/%s.jpg'%(dataset_realpath, year, image_id))
            list_file.write('\n')
    list_file.close()
    # print out item number statistic
    print('\nDone for %s_%s.txt. classes number statistic'%(year, image_set))
    print('Image number: %d'%(len(image_ids)))
    print('Object class number:')
    for (class_name, number) in class_count.items():
        print('%s: %d' % (class_name, number))
    print('total object number:', np.sum(list(class_count.values())))

