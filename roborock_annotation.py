import xml.etree.ElementTree as ET
from os import getcwd

sets=[('2007', 'train'), ('2007', 'val'), ('2007', 'test')]
classes = ["bar stool a", "shoe", "pet feces", "wire"]
dataset_root = 'det_trainset'


def convert_annotation(year, image_id, list_file):
    in_file = open('%s/VOC%s/Annotations/%s.xml'%(dataset_root, year, image_id))
    tree=ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))


def has_object(year, image_id):
    in_file = open('%s/VOC%s/Annotations/%s.xml'%(dataset_root, year, image_id))
    tree=ET.parse(in_file)
    root = tree.getroot()
    count = 0

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        count = count +1
    return count != 0

wd = getcwd()

for year, image_set in sets:
    image_ids = open('%s/VOC%s/ImageSets/Main/%s.txt'%(dataset_root, year, image_set)).read().strip().split()
    list_file = open('roborock_%s_%s.txt'%(year, image_set), 'w')
    for image_id in image_ids:
        if has_object(year, image_id):
            list_file.write('%s/%s/VOC%s/JPEGImages/%s.jpg'%(wd, dataset_root, year, image_id))
            convert_annotation(year, image_id, list_file)
            list_file.write('\n')
    list_file.close()

