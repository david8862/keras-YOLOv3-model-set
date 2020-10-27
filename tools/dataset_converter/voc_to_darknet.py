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

def convert_to_darknet(size, box):
    '''
    convert box format from (xmin,xmax,ymin,ymax) to darknet
    style (x,y,w,h) format with relative coordinate
    '''
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)


def convert_annotation(dataset_path, year, image_id, output_path, include_difficult):
    in_file = open('%s/VOC%s/Annotations/%s.xml'%(dataset_path, year, image_id), encoding='utf-8')
    out_file = open('%s/%s.txt'%(output_path, image_id), 'w')
    tree=ET.parse(in_file)
    root = tree.getroot()
    # get image size (width, height)
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)

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

        # parse box coordinate to (xmin,xmax,ymin,ymax) format
        xml_box = obj.find('bndbox')
        box = (float(xml_box.find('xmin').text), float(xml_box.find('xmax').text), float(xml_box.find('ymin').text), float(xml_box.find('ymax').text))

        # convert box to darknet format and save to txt
        darknet_box = convert_to_darknet((width, height), box)
        out_file.write(str(class_id) + " " + " ".join([str(item) for item in darknet_box]) + '\n')
        class_count[class_name] = class_count[class_name] + 1
    out_file.close()


def has_object(dataset_path, year, image_id, include_difficult):
    '''
    check if an image annotation has valid object bbox info,
    return a boolean result
    '''
    try:
        in_file = open('%s/VOC%s/Annotations/%s.xml'%(dataset_path, year, image_id), encoding='utf-8')
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
        count = count + 1
    return count != 0


def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        classes = f.readlines()
    classes = [c.strip() for c in classes]
    return classes


parser = argparse.ArgumentParser(description='convert PascalVOC dataset annotation to darknet/YOLOv5 txt annotation files')
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
    pbar = tqdm(total=len(image_ids), desc='Converting VOC%s_%s'%(year, image_set))
    for image_id in image_ids:
        file_string = '%s/VOC%s/JPEGImages/%s.jpg'%(dataset_realpath, year, image_id)
        # check if the image file exists
        if not os.path.exists(file_string):
            file_string = '%s/VOC%s/JPEGImages/%s.jpeg'%(dataset_realpath, year, image_id)
        if not os.path.exists(file_string):
            raise ValueError('image file for id: {} not exists'.format(image_id))

        if has_object(dataset_realpath, year, image_id, args.include_difficult):
            convert_annotation(dataset_realpath, year, image_id, args.output_path, args.include_difficult)
        elif args.include_no_obj:
            # include no object image. just create an empty file
            os.system('touch %s/%s.txt'%(args.output_path, image_id))
        pbar.update(1)
    pbar.close()
    # print out item number statistic
    print('\nDone for VOC%s %s. classes number statistic'%(year, image_set))
    print('Image number: %d'%(len(image_ids)))
    print('Object class number:')
    for (class_name, number) in class_count.items():
        print('%s: %d' % (class_name, number))
    print('total object number:', np.sum(list(class_count.values())))

