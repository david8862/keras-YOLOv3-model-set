import json, argparse
from collections import defaultdict
from os import getcwd


#sets=[('instances_train2017', 'train2017'), ('instances_val2017', 'val2017'), ('image_info_test-dev2017', 'test2017')]
sets=[('instances_train2017', 'train2017'), ('instances_val2017', 'val2017')]


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', type=str, help='path to MSCOCO dataset, default is ../mscoco2017', default=getcwd()+'/../mscoco2017')
parser.add_argument('--output_path', type=str,  help='output path for generated annotation txt files, default is ./', default='./')
args = parser.parse_args()

for dataset, datatype in sets:
    name_box_id = defaultdict(list)
    id_name = dict()
    f = open(
        "%s/annotations/%s.json"%(args.dataset_path, dataset),
        encoding='utf-8')
    data = json.load(f)

    annotations = data['annotations']
    for ant in annotations:
        id = ant['image_id']
        name = '%s/%s/%012d.jpg' % (args.dataset_path, datatype, id)
        cat = ant['category_id']

        if cat >= 1 and cat <= 11:
            cat = cat - 1
        elif cat >= 13 and cat <= 25:
            cat = cat - 2
        elif cat >= 27 and cat <= 28:
            cat = cat - 3
        elif cat >= 31 and cat <= 44:
            cat = cat - 5
        elif cat >= 46 and cat <= 65:
            cat = cat - 6
        elif cat == 67:
            cat = cat - 7
        elif cat == 70:
            cat = cat - 9
        elif cat >= 72 and cat <= 82:
            cat = cat - 10
        elif cat >= 84 and cat <= 90:
            cat = cat - 11

        name_box_id[name].append([ant['bbox'], cat])

    f = open('%s/%s.txt'%(args.output_path, datatype), 'w')
    for key in name_box_id.keys():
        f.write(key)
        box_infos = name_box_id[key]
        for info in box_infos:
            x_min = int(info[0][0])
            y_min = int(info[0][1])
            x_max = x_min + int(info[0][2])
            y_max = y_min + int(info[0][3])

            box_info = " %d,%d,%d,%d,%d" % (
                x_min, y_min, x_max, y_max, int(info[1]))
            f.write(box_info)
        f.write('\n')
    f.close()
