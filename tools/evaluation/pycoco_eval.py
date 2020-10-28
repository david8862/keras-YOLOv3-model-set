#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
run this scipt to evaluate COCO AP with pycocotools
'''
import os, argparse, json
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def convert_coco_coordinate(box):
    '''
    convert box coordinate from
    [xmin, ymin, xmax, ymax] to
    [xmin, ymin, width, height]
    '''
    xmin, ymin, xmax, ymax = box
    #assert(xmax > xmin)
    #assert(ymax > ymin)

    x_min = float(xmin)
    y_min = float(ymin)
    width = float(abs(xmax - xmin))
    height = float(abs(ymax - ymin))

    return [x_min, y_min, width, height]


def convert_coco_category(category_id):
    '''
    convert continuous coco class id (0~79) to discontinuous coco category id
    '''
    if category_id >= 0 and category_id <= 10:
        category_id = category_id + 1
    elif category_id >= 11 and category_id <= 23:
        category_id = category_id + 2
    elif category_id >= 24 and category_id <= 25:
        category_id = category_id + 3
    elif category_id >= 26 and category_id <= 39:
        category_id = category_id + 5
    elif category_id >= 40 and category_id <= 59:
        category_id = category_id + 6
    elif category_id == 60:
        category_id = category_id + 7
    elif category_id == 61:
        category_id = category_id + 9
    elif category_id >= 62 and category_id <= 72:
        category_id = category_id + 10
    elif category_id >= 73 and category_id <= 79:
        category_id = category_id + 11
    else:
        raise ValueError('Invalid category id')
    return category_id


def coco_result_generate(result_txt, coco_result_json, customize_coco):
    with open(result_txt) as f:
        result_lines = f.readlines()

    output_list = []
    pbar = tqdm(total=len(result_lines), desc='COCO result generate')
    for result_line in result_lines:
        # result line format is same as annotation txt but adding score, like:
        #
        # path/to/img1.jpg 50,100,150,200,0,0.86 30,50,200,120,3,0.95
        #
        line = result_line.split()

        # parse image_id from full image_name
        image_name = line[0]
        try:
            image_id = int(os.path.basename(image_name).split('.')[0])
        except:
            # if image_name is not a number, try to use
            # the name string as image_id
            image_id = os.path.basename(image_name).split('.')[0]
        pbar.update(1)

        # parse boxes info
        boxes = [box.split(',') for box in line[1:]]

        for box in boxes:
            # check if box info is valid
            assert len(box) == 6, 'Invalid box format.'

            # get box coordinate, class and score info,
            # then convert to coco result format
            box_coordinate = [int(x) for x in box[:4]]
            box_class = int(box[4])
            box_score = float(box[5])
            box_coordinate = convert_coco_coordinate(box_coordinate)
            box_category = box_class+1 if customize_coco else convert_coco_category(box_class)

            # fullfil coco result dict item
            # coco detection result is a list of following format dict:
            # {
            #  "image_id": int,
            #  "category_id": int,
            #  "bbox": [x,y,width,height],
            #  "score": float
            # }
            result_dict = {}
            result_dict['image_id'] = image_id
            result_dict['category_id'] = box_category
            result_dict['bbox'] = box_coordinate
            result_dict['score'] = box_score

            # add result dict to output list
            output_list.append(result_dict)
    pbar.close()

    # save to coco result json
    json_fp = open(coco_result_json, 'w')
    json_str = json.dumps(output_list)
    json_fp.write(json_str)
    json_fp.close()


def pycoco_eval(annotation_file, result_file):
    cocoGt=COCO(annotation_file)
    cocoDt=cocoGt.loadRes(result_file)

    imgIds=sorted(cocoGt.getImgIds())
    #imgIds=imgIds[0:100]
    #imgIds = imgIds[np.random.randint(100)]

    # running evaluation
    cocoEval = COCOeval(cocoGt, cocoDt, iouType='bbox')
    cocoEval.params.imgIds = imgIds
    # cocoEval.params.catIds = [1] # we can specify some category ids to eval, e.g person=1
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()


def main():
    parser = argparse.ArgumentParser(description='generate coco result json and evaluate COCO AP with pycocotools')
    parser.add_argument('--result_txt', required=True, type=str, help='txt detection result file')
    parser.add_argument('--coco_annotation_json', required=True, type=str, help='coco json annotation file')
    parser.add_argument('--coco_result_json', required=False, type=str, help='output coco json result file, default=%(default)s', default='coco_result.json')
    parser.add_argument('--customize_coco', default=False, action="store_true", help='It is a user customize coco dataset. Will not follow standard coco class label')
    args = parser.parse_args()

    coco_result_generate(args.result_txt, args.coco_result_json, args.customize_coco)
    pycoco_eval(args.coco_annotation_json, args.coco_result_json)


if __name__ == "__main__":
    main()
