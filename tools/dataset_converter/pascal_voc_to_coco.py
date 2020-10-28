#!/usr/bin/env python3
# -*- coding=utf-8 -*-

import sys
import os, argparse
import json
from tqdm import tqdm
import xml.etree.ElementTree as ET

# start id of bbox in coco annotation
START_BOUNDING_BOX_ID = 1

# If necessary, pre-define category and its id
#PRE_DEFINE_CATEGORIES = {"aeroplane": 1, "bicycle": 2, "bird": 3, "boat": 4,
                          #"bottle":5, "bus": 6, "car": 7, "cat": 8, "chair": 9,
                          #"cow": 10, "diningtable": 11, "dog": 12, "horse": 13,
                          #"motorbike": 14, "person": 15, "pottedplant": 16,
                          #"sheep": 17, "sofa": 18, "train": 19, "tvmonitor": 20}


def get_and_check(root, name, length):
    vars = root.findall(name)
    if len(vars) == 0:
        raise NotImplementedError('Can not find %s in %s.'%(name, root.tag))
    if length > 0 and len(vars) != length:
        raise NotImplementedError('The size of %s is supposed to be %d, but is %d.'%(name, length, len(vars)))
    if length == 1:
        vars = vars[0]
    return vars


# get image_id from image file name
def parse_image_id(filename):
    filename = os.path.splitext(filename)[0]
    try:
        # get rid of the underline in VOC2012 filename
        #filename = filename.replace('_', '')
        #return int(filename)
        image_id = int(filename)
    except:
        # if image name is not a number, try to use
        # the name string as image_id
        image_id = filename
        #raise NotImplementedError('Filename %s is supposed to be an integer.'%(filename))

    return image_id


def get_category(category_file):
    '''loads the classes'''
    category_dict = {}
    with open(category_file) as f:
        categories_names = f.readlines()
    for index, category_name in enumerate(categories_names):
        category_dict[category_name.strip()] = index + 1
    return category_dict


def convert(xml_list, xml_dir, categories, json_file, merge_category=False):
    '''
    :param xml_list: list of PascalVOC annotation XMLs
    :param xml_dir: path of PascalVOC annotation XMLs
    :param categories: dict of object category name to id
    :param json_file: output coco json file
    :return: None
    '''
    # COCO instances data struct
    json_dict = {"images":[], "type": "instances", "annotations": [], "categories": []}
    #categories = PRE_DEFINE_CATEGORIES
    bnd_id = START_BOUNDING_BOX_ID

    pbar = tqdm(total=len(xml_list), desc='convert PascalVOC XML')
    for line in xml_list:
        line = line.strip()
        #print("Processing {}".format(line))
        pbar.update(1)

        # parse XML
        xml_f = os.path.join(xml_dir, line)
        tree = ET.parse(xml_f)
        root = tree.getroot()

        # get image filename
        path = root.findall('path')
        if len(path) == 1:
            filename = os.path.basename(path[0].text)
        elif len(path) == 0:
            filename = get_and_check(root, 'filename', 1).text
        else:
            raise NotImplementedError('%d paths found in %s'%(len(path), line))

        ## get image ID & size
        image_id = parse_image_id(filename)  # image ID
        size = get_and_check(root, 'size', 1)

        # form up image info
        width = int(get_and_check(size, 'width', 1).text)
        height = int(get_and_check(size, 'height', 1).text)
        image = {'file_name': filename, 'height': height, 'width': width, 'id':image_id}
        json_dict['images'].append(image)

        ## Currently we do not support segmentation
        #  segmented = get_and_check(root, 'segmented', 1).text
        #  assert segmented == '0'

        # handle object bounding box
        for obj in root.findall('object'):
            # parse object name
            category = get_and_check(obj, 'name', 1).text

            # update category ID dict if allow merge
            if category not in categories:
                if merge_category:
                    new_id = len(categories)
                    categories[category] = new_id
                else:
                    continue
            category_id = categories[category]

            # get bounding box
            bndbox = get_and_check(obj, 'bndbox', 1)
            xmin = int(get_and_check(bndbox, 'xmin', 1).text)
            ymin = int(get_and_check(bndbox, 'ymin', 1).text)
            xmax = int(get_and_check(bndbox, 'xmax', 1).text)
            ymax = int(get_and_check(bndbox, 'ymax', 1).text)
            assert(xmax > xmin)
            assert(ymax > ymin)
            o_width = abs(xmax - xmin)
            o_height = abs(ymax - ymin)

            annotation = dict()
            # set segmentation info. points in anti-clockwise direction
            annotation['segmentation'] = [[xmin,ymin,xmin,ymax,xmax,ymax,xmax,ymin]]
            annotation['area'] = o_width*o_height
            annotation['iscrowd'] = 0
            annotation['image_id'] = image_id
            annotation['bbox'] = [xmin, ymin, o_width, o_height]
            annotation['category_id'] = category_id
            annotation['id'] = bnd_id
            annotation['ignore'] = 0

            json_dict['annotations'].append(annotation)
            bnd_id = bnd_id + 1
    pbar.close()

    # create category ID dict
    for cate, cid in categories.items():
        cat = {'supercategory': 'none', 'id': cid, 'name': cate}
        json_dict['categories'].append(cat)

    # save to COCO json file
    json_fp = open(json_file, 'w')
    json_str = json.dumps(json_dict)
    json_fp.write(json_str)
    json_fp.close()


def pascalvoc_to_coco(voc_root_path, dataset_file, category_file, json_file, merge_category):
    xml_dir = os.path.join(voc_root_path, 'Annotations')

    # get dataset info
    with open(dataset_file) as f:
        lines = f.readlines()
    xml_list = [line.split()[0]+'.xml' for line in lines]

    # get category info
    categories = get_category(category_file)

    # convert the xml list to coco json file
    convert(xml_list, xml_dir, categories, json_file, merge_category)


def main():
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS, description='convert PascalVOC XML annotation to MSCOCO JSON annotation')

    parser.add_argument('--voc_root_path', required=True, type=str, help='VOCdevkit root path, e.g VOCdevkit/VOC2007')
    parser.add_argument('--dataset_file', required=True, type=str, help='PascalVOC dataset file path')
    parser.add_argument('--category_file', required=True, type=str, help='category definitions file')
    parser.add_argument('--json_file', required=True, type=str, help='Output COCO format JSON file path')
    parser.add_argument('--merge_category', default=False, action="store_true", help='Merge the object class in dataset to category list if not exist')

    args = parser.parse_args()

    pascalvoc_to_coco(args.voc_root_path, args.dataset_file, args.category_file, args.json_file, args.merge_category)


if __name__ == '__main__':
    main()
