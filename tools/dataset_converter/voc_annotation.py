#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, argparse
import numpy as np
import xml.etree.ElementTree as ET
from tqdm import tqdm
from collections import OrderedDict

sets=[('2007', 'train'), ('2007', 'val'), ('2007', 'test'), ('2012', 'train'), ('2012', 'val')]
classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

class_count = {}

def convert_annotation(dataset_path, year, image_id, list_file, include_difficult):
    xml_file = open('%s/VOC%s/Annotations/%s.xml'%(dataset_path, year, image_id), encoding='utf-8')
    tree=ET.parse(xml_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = obj.find('difficult')
        if difficult is None:
            difficult = '0'
        else:
            difficult = difficult.text
        class_name = obj.find('name').text
        if class_name not in classes:
            continue
        if not include_difficult and int(difficult)==1:
            continue
        class_id = classes.index(class_name)

        # parse box coordinate to (xmin,ymin,xmax,ymax) format
        xml_box = obj.find('bndbox')
        box = (int(float(xml_box.find('xmin').text)), int(float(xml_box.find('ymin').text)), int(float(xml_box.find('xmax').text)), int(float(xml_box.find('ymax').text)))
        # write box info to txt
        list_file.write(" " + ",".join([str(item) for item in box]) + ',' + str(class_id))
        class_count[class_name] = class_count[class_name] + 1


def has_object(dataset_path, year, image_id, include_difficult):
    '''
    check if an image annotation has valid object bbox info,
    return a boolean result
    '''
    try:
        xml_file = open('%s/VOC%s/Annotations/%s.xml'%(dataset_path, year, image_id), encoding='utf-8')
    except:
        # bypass image if no annotation
        return False
    tree=ET.parse(xml_file)
    root = tree.getroot()
    count = 0

    for obj in root.iter('object'):
        difficult = obj.find('difficult')
        if difficult is None:
            difficult = '0'
        else:
            difficult = difficult.text
        class_name = obj.find('name').text
        if class_name not in classes:
            continue
        if not include_difficult and int(difficult)==1:
            continue
        count = count + 1
    return count != 0


def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        classes = f.readlines()
    classes = [c.strip() for c in classes]
    return classes


parser = argparse.ArgumentParser(description='convert PascalVOC dataset annotation to txt annotation file')
parser.add_argument('--dataset_path', type=str, help='path to PascalVOC dataset, default=%(default)s', default=os.getcwd()+'/../../VOCdevkit')
parser.add_argument('--year', type=str, help='subset path of year (2007/2012), default will cover both', default=None)
parser.add_argument('--set', type=str, help='convert data set, default will cover train, val and test', default=None)
parser.add_argument('--output_path', type=str,  help='output path for generated annotation txt files, default=%(default)s', default='./')
parser.add_argument('--classes_path', type=str, required=False, help='path to class definitions')
parser.add_argument('--include_difficult', action="store_true", help='to include difficult object', default=False)
parser.add_argument('--include_no_obj', action="store_true", help='to include no object image', default=False)
args = parser.parse_args()

# update class names
if args.classes_path:
    classes = get_classes(args.classes_path)

# get real path for dataset
dataset_realpath = os.path.realpath(args.dataset_path)

# create output path
os.makedirs(args.output_path, exist_ok=True)


# get specific sets to convert
if args.year is not None:
    sets = [item for item in sets if item[0] == args.year]
if args.set is not None:
    sets = [item for item in sets if item[1] == args.set]

for year, image_set in sets:
    # count class item number in each set
    class_count = OrderedDict([(item, 0) for item in classes])

    image_ids = open('%s/VOC%s/ImageSets/Main/%s.txt'%(dataset_realpath, year, image_set)).read().strip().split()
    list_file = open('%s/%s_%s.txt'%(args.output_path, year, image_set), 'w')
    pbar = tqdm(total=len(image_ids), desc='Converting VOC%s_%s'%(year, image_set))
    for image_id in image_ids:
        file_string = '%s/VOC%s/JPEGImages/%s.jpg'%(dataset_realpath, year, image_id)
        # check if the image file exists
        if not os.path.exists(file_string):
            file_string = '%s/VOC%s/JPEGImages/%s.jpeg'%(dataset_realpath, year, image_id)
        if not os.path.exists(file_string):
            raise ValueError('image file for id: {} not exists'.format(image_id))

        if has_object(dataset_realpath, year, image_id, args.include_difficult):
            list_file.write(file_string)
            convert_annotation(dataset_realpath, year, image_id, list_file, args.include_difficult)
            list_file.write('\n')
        elif args.include_no_obj:
            # include no object image. just write file path
            list_file.write(file_string)
            list_file.write('\n')
        pbar.update(1)
    pbar.close()
    list_file.close()
    # print out item number statistic
    print('\nDone for VOC%s %s. classes number statistic'%(year, image_set))
    print('Image number: %d'%(len(image_ids)))
    print('Object class number:')
    for (class_name, number) in class_count.items():
        print('%s: %d' % (class_name, number))
    print('total object number:', np.sum(list(class_count.values())))

