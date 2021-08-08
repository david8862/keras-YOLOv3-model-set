#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from collections import OrderedDict


def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        classes = f.readlines()
    classes = [c.strip() for c in classes]
    return classes


def yolo_annotation(image_path, label_path, classes_path, output_file):
    # get real path for images
    image_realpath = os.path.realpath(image_path)

    # get class names and count class item number
    classes = get_classes(classes_path)
    class_count = OrderedDict([(item, 0) for item in classes])

    list_file = open(output_file, 'w')

    # filter invalid files
    image_files = os.listdir(image_path)
    image_files = list(filter(lambda x: x.endswith('.jpg'), image_files))

    pbar = tqdm(total=len(image_files), desc='Converting YOLO annotation')
    for image_file in image_files:
        pbar.update(1)

        # record real image path in txt
        file_string = os.path.join(image_realpath, image_file)
        list_file.write(file_string)

        label_file = image_file.replace('.jpg','.txt')
        label_file = os.path.join(label_path, label_file)

        # only record image path if no label file
        if not os.path.exists(label_file):
            list_file.write('\n')
            continue

        with open(label_file, encoding='utf-8') as f:
            labels = f.readlines()
            labels = [label.strip() for label in labels]

        # get image width & height for bbox convert
        image = Image.open(file_string)
        width, height = image.size

        for label in labels:
            label = label.split()
            class_id = int(label[0])
            box_x = float(label[1]) * width
            box_y = float(label[2]) * height
            box_width = float(label[3]) * width
            box_height = float(label[4]) * height

            # count class item number
            class_name = classes[class_id]
            class_count[class_name] = class_count[class_name] + 1

            # convert bbox to (xmin,ymin,xmax,ymax) format
            box = map(round, [box_x-(box_width/2), box_y-(box_height/2), box_x+(box_width/2), box_y+(box_height/2)])
            list_file.write(" " + ",".join([str(item) for item in box]) + ',' + str(class_id))

        list_file.write('\n')
    pbar.close()
    list_file.close()
    # print out item number statistic
    print('\nDone for %s. classes number statistic'%(output_file))
    print('Image number: %d'%(len(image_files)))
    print('Object class number:')
    for (class_name, number) in class_count.items():
        print('%s: %d' % (class_name, number))
    print('total object number:', np.sum(list(class_count.values())))


def main():
    parser = argparse.ArgumentParser(description='convert YOLO dataset annotation to txt annotation file')

    parser.add_argument('--image_path', required=True, type=str, help='image file path')
    parser.add_argument('--label_path', required=True, type=str, help='YOLO label file path')
    parser.add_argument('--classes_path', type=str, required=True, help='path to class definitions')
    parser.add_argument('--output_file', type=str,  help='output path for generated annotation txt files, default=%(default)s', default='./')

    args = parser.parse_args()

    yolo_annotation(args.image_path, args.label_path, args.classes_path, args.output_file)


if __name__ == '__main__':
    main()
