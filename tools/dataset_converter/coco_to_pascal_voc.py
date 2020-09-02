#!/usr/bin/env python3
# -*- coding=utf-8 -*-
import os, sys, argparse
import shutil
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageDraw
from tqdm import tqdm
from pycocotools.coco import COCO

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))
from common.utils import get_classes

sets=['train2017', 'val2017']

xml_head_template = """\
<annotation>
    <folder>JPEGImages</folder>
    <filename>%s</filename>
    <relpath>../JPEGImages/%s</relpath>
    <source>
        <database>Unknown</database>
        <annotation>PASCAL VOC2007</annotation>
        <image>flickr</image>
        <flickrid>NULL</flickrid>
    </source>
    <owner>
        <flickrid>NULL</flickrid>
        <name>company</name>
    </owner>
    <size>
        <width>%d</width>
        <height>%d</height>
        <depth>%d</depth>
    </size>
    <segmented>0</segmented>
"""

xml_obj_template = """\
    <object>
        <name>%s</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>%d</xmin>
            <ymin>%d</ymin>
            <xmax>%d</xmax>
            <ymax>%d</ymax>
        </bndbox>
    </object>
"""

xml_tail_template = '''\
</annotation>
'''


def write_xml(annotation_path, head, objs, tail):
    f = open(annotation_path, "w")
    f.write(head)
    for obj in objs:
        f.write(xml_obj_template%(obj[0],obj[1],obj[2],obj[3],obj[4]))
    f.write(tail)


def save_annotations_and_imgs(coco, dataset, filename, objs, coco_root_path, output_path):
    xml_dir = os.path.join(output_path, 'Annotations')
    img_dir = os.path.join(output_path, 'JPEGImages')
    os.makedirs(xml_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)

    # e.g. COCO_train2014_000000196610.jpg --> COCO_train2014_000000196610.xml
    annotation_path = os.path.join(xml_dir, filename.split('.')[0]+'.xml')
    img_path = os.path.join(coco_root_path, dataset, filename)

    dst_imgpath= os.path.join(img_dir, filename)

    img=cv2.imread(img_path)
    if (img.shape[2] == 1):
        print(filename + " not a RGB image")
        return
    shutil.copy(img_path, dst_imgpath)

    head = xml_head_template % (filename, filename, img.shape[1], img.shape[0], img.shape[2])
    tail = xml_tail_template
    write_xml(annotation_path, head, objs, tail)


def get_coco_classes(coco):
    classes=dict()
    for cls in coco.dataset['categories']:
        classes[cls['id']]=cls['name']
    return classes


def parse_obj_info(coco, dataset, img, class_dict, cls_id, class_names):
    #get annotation info with image id
    annIds = coco.getAnnIds(imgIds=img['id'], catIds=cls_id, iscrowd=None)
    annotations = coco.loadAnns(annIds)
    # coco.showAnns(anns)
    objs = []
    for annotation in annotations:
        class_name=class_dict[annotation['category_id']]
        if class_name in class_names:
            if 'bbox' in annotation:
                bbox=annotation['bbox']
                xmin = int(bbox[0])
                ymin = int(bbox[1])
                xmax = int(bbox[2] + bbox[0])
                ymax = int(bbox[3] + bbox[1])
                obj = [class_name, xmin, ymin, xmax, ymax]
                objs.append(obj)
    return objs


def coco_to_pascalvoc(coco_root_path, output_path, class_names):
    for dataset in sets:
        # e.g. COCO/annotations/instances_train2017.json
        annotation_file='{}/annotations/instances_{}.json'.format(coco_root_path, dataset)
        # COCO API for initializing annotated data
        coco = COCO(annotation_file)

        # dict for all classes in coco
        class_dict = get_coco_classes(coco)
        # if no class specified, then use all COCO classes
        if not class_names:
            class_names = list(class_dict.values())

        # mapping class name to COCO class id
        classes_ids = coco.getCatIds(catNms=class_names)

        # use set to avoid dumpliate image id
        dataset_list = set()
        for cls in class_names:
            #Get ID number of this class
            cls_id=coco.getCatIds(catNms=[cls])
            img_ids=coco.getImgIds(catIds=cls_id)
            for imgId in tqdm(img_ids, desc='Extracting %s'%(cls)):
                img = coco.loadImgs(imgId)[0]
                filename = img['file_name']
                objs = parse_obj_info(coco, dataset, img, class_dict, classes_ids, class_names)
                save_annotations_and_imgs(coco, dataset, filename, objs, coco_root_path, output_path)
                # record image id
                dataset_list.add(os.path.splitext(filename)[0])

        # get sorted dataset list
        dataset_list = list(dataset_list)
        dataset_list.sort()

        # save image id list to dataset txt file
        os.makedirs(os.path.join(output_path, 'ImageSets', 'Main'), exist_ok=True)
        dataset_file = open(os.path.join(output_path, 'ImageSets', 'Main', dataset+'.txt'), 'w')

        for image_id in dataset_list:
                dataset_file.write(image_id)
                dataset_file.write('\n')
        dataset_file.close()


def main():
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS, description='convert MSCOCO dataset to PascalVOC dataset')

    parser.add_argument('--coco_root_path', type=str, required=True, help='path to MSCOCO dataset')
    parser.add_argument('--output_path', type=str, required=True,  help='output path for generated PascalVOC dataset')
    parser.add_argument('--classes_path', type=str, required=False, help='path to a selected sub-classes definition, optional', default=None)

    args = parser.parse_args()
    if args.classes_path:
        class_names = get_classes(args.classes_path)
    else:
        class_names = None

    coco_to_pascalvoc(args.coco_root_path, args.output_path, class_names)


if __name__ == '__main__':
    main()
