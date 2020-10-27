#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import xml.etree.ElementTree as ET
import numpy as np
import os, argparse, cv2, glob
from os import getcwd
from functools import reduce
from PIL import Image


#sets=[('2007', 'train'), ('2007', 'val'), ('2007', 'test'), ('2012', 'train'), ('2012', 'val')]
sets=[('2007', 'train')]
classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

class_count = {}


def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        classes = f.readlines()
    classes = [c.strip() for c in classes]
    return classes


def crop_bbox_image(image_name, bbox, count, class_name, output_path):
    image = cv2.imread(image_name)
    xmin, ymin, xmax, ymax = bbox

    cropImg = image[ymin:ymax, xmin:xmax]

    output_class_path = os.path.join(output_path, class_name)
    os.makedirs(output_class_path, exist_ok=True)
    output_image_name = os.path.join(output_class_path, os.path.basename(image_name).split('.')[0] + '_' + class_name + '_' + str(count) + '.jpg')
    cv2.imwrite(output_image_name, cropImg)


def parse_and_crop(dataset_path, year, image_id, include_difficult, output_path):
    in_file = open('%s/VOC%s/Annotations/%s.xml'%(dataset_path, year, image_id), encoding='utf-8')
    tree=ET.parse(in_file)
    root = tree.getroot()
    count = 0

    for obj in root.iter('object'):
        difficult = obj.find('difficult')
        if not difficult:
            difficult = '0'
        else:
            difficult = difficult.text
        cls = obj.find('name').text
        if cls not in classes:
            continue
        if not include_difficult and int(difficult)==1:
            continue
        xmlbox = obj.find('bndbox')
        cls_id = classes.index(cls)
        bbox = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        image_name = '%s/VOC%s/JPEGImages/%s.jpg'%(dataset_path, year, image_id)
        crop_bbox_image(image_name, bbox, count, cls, output_path)
        count = count + 1
        class_count[cls] = class_count[cls] + 1


def has_object(dataset_path, year, image_id, include_difficult):
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
        if not difficult:
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


def ORB_img_similarity(img1_path,img2_path):
    """
    :param img1_path: 图片1路径
    :param img2_path: 图片2路径
    :return: 图片相似度
    """
    try:
        # 读取图片
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)

        # 初始化ORB检测器
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)

        # 提取并计算特征点
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        # knn筛选结果
        matches = bf.knnMatch(des1, trainDescriptors=des2, k=2)

        # 查看最大匹配点数目
        good = [m for (m, n) in matches if m.distance < 0.75 * n.distance]
        similary = len(good) / len(matches)
        return similary

    except:
        return 0


# 计算图片的局部哈希值--pHash
def phash(img):
    """
    :param img: 图片
    :return: 返回图片的局部hash值
    """
    img = img.resize((8, 8), Image.ANTIALIAS).convert('L')
    avg = reduce(lambda x, y: x + y, img.getdata()) / 64.
    hash_value=reduce(lambda x, y: x | (y[1] << y[0]), enumerate(map(lambda i: 0 if i < avg else 1, img.getdata())), 0)
    return hash_value


#计算两个图片相似度函数局部敏感哈希算法
def phash_img_similarity(img1_path,img2_path):
    """
    :param img1_path: 图片1路径
    :param img2_path: 图片2路径
    :return: 图片相似度
    """
    # 读取图片
    img1 = Image.open(img1_path)
    img2 = Image.open(img2_path)

    # 计算汉明距离
    distance = bin(phash(img1) ^ phash(img2)).count('1')
    similary = 1 - distance / max(len(bin(phash(img1))), len(bin(phash(img1))))
    return similary


# 直方图计算图片相似度算法
def make_regalur_image(img, size=(128, 128)):
    """我们有必要把所有的图片都统一到特别的规格，这里选择128x128的分辨率。"""
    return img.resize(size).convert('RGB')

def hist_similar(lh, rh):
    assert len(lh) == len(rh)
    return sum(1 - (0 if l == r else float(abs(l - r))/max(l, r)) for l, r in zip(lh, rh))/len(lh)

def calc_similar(li, ri):
    return sum(hist_similar(l.histogram(), r.histogram()) for l, r in zip(split_image(li), split_image(ri))) / 16.0

def calc_similar_by_path(lf, rf):
    li, ri = make_regalur_image(Image.open(lf)), make_regalur_image(Image.open(rf))
    return calc_similar(li, ri)

def split_image(img, part_size = (32, 32)):
    w, h = img.size
    pw, ph = part_size
    assert w % pw == h % ph == 0
    return [img.crop((i, j, i+pw, j+ph)).copy() for i in range(0, w, pw) \
            for j in range(0, h, ph)]



# 融合函数计算图片相似度
def calc_image_similarity(img1_path,img2_path):
    """
    :param img1_path: filepath+filename
    :param img2_path: filepath+filename
    :return: 图片最终相似度
    """
    # 融合相似度阈值
    threshold1=0.85

    similary_ORB=float(ORB_img_similarity(img1_path,img2_path))
    similary_phash=float(phash_img_similarity(img1_path,img2_path))
    similary_hist=float(calc_similar_by_path(img1_path, img2_path))

    # 如果三种算法的相似度最大的那个大于0.85，则相似度取最大，否则，取最小。
    max_three_similarity=max(similary_ORB,similary_phash,similary_hist)
    min_three_similarity=min(similary_ORB,similary_phash,similary_hist)
    if max_three_similarity>threshold1:
        result=max_three_similarity
    else:
        result=min_three_similarity

    return round(result,3)


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', type=str, help='path to PascalVOC dataset, default is ../../VOCdevkit', default=getcwd()+'/../../VOCdevkit')
parser.add_argument('--output_path', type=str,  help='output path for croped image and similarity merge txt files, default is ./', default='./')
parser.add_argument('--classes_path', type=str, required=False, help='path to class definitions')
parser.add_argument('--similarity_threshold', type=int, required=False, help='threshold for similarity score', default=0.6)
parser.add_argument('--include_difficult', action="store_true", help='to include difficult object', default=False)
args = parser.parse_args()


# update class names
if args.classes_path:
    classes = get_classes(args.classes_path)


for year, image_set in sets:
    # count class item number in each set
    class_count = {itm: 0 for itm in classes}

    print('\nHandling object crop in %s.txt'%(image_set))
    image_ids = open('%s/VOC%s/ImageSets/Main/%s.txt'%(args.dataset_path, year, image_set)).read().strip().split()
    for image_id in image_ids:
        if has_object(args.dataset_path, year, image_id, args.include_difficult):
            parse_and_crop(args.dataset_path, year, image_id, args.include_difficult, args.output_path)

    # print out item number statistic
    print('\nDone for %s_%s.txt. object number statistic'%(year, image_set))
    for (class_name, number) in class_count.items():
        print('%s: %d' % (class_name, number))
    print('total number:', np.sum(list(class_count.values())))


# calculate croped object image similarity for every class
for class_name in classes:
    print('calculating object similarity for', class_name)
    # get object image list
    class_image_path = os.path.join(args.output_path, class_name)
    image_list = glob.glob(os.path.join(class_image_path, '*.jpg'))
    image_list.sort()

    # prepare similarity merge result file
    similarity_file_name = os.path.join(args.output_path, class_name + '_similarity.txt')
    print('Will save in', similarity_file_name)
    similarity_file = open(similarity_file_name, 'w')

    for i, image_name in enumerate(image_list):
        # walk through the images to compare 1 by 1
        similarity_file.write(os.path.basename(image_name))
        sub_list = image_list[i+1:]
        for compare_image_name in sub_list:
            # check the base image_id of the obj image,
            # if they're from same image_id, then don't compare
            if os.path.basename(image_name).split('_')[0]  == os.path.basename(compare_image_name).split('_')[0]:
                continue
            similarity = calc_image_similarity(image_name, compare_image_name)
            if similarity > args.similarity_threshold:
                # if matched, save in result file and remove it from checking list
                image_list.remove(compare_image_name)
                similarity_file.write(', ' + os.path.basename(compare_image_name) + ': %.2f'%(similarity))
        similarity_file.write('\n')
    similarity_file.close()


