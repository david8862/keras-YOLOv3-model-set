#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculate mAP for YOLO model on some annotation dataset
"""
import numpy as np
import random
import os, argparse
from predict import predict, get_classes, get_anchors
from PIL import Image

import tensorflow as tf
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as KTF

config = tf.ConfigProto()
config.gpu_options.allow_growth=True   #dynamic alloc GPU resource
config.gpu_options.per_process_gpu_memory_fraction = 0.3  #GPU memory threshold 0.3
session = tf.Session(config=config)

# set session
KTF.set_session(session)

def annotation_parse(annotation_file, class_names):
    '''
    parse annotation file to get image dict and ground truth class dict

    image dict would be like:
    annotation_records = {
        '000001.jpg': {'100,120,200,235':'dog', '85,63,156,128':'car', ...},
        ...
    }

    ground truth class dict would be like:
    classes_records = {
        'car': {'00001.jpg':'100,120,200,235', '00002.jpg':'85,63,156,128', ...},
        ...
    }
    '''
    with open(annotation_file) as f:
        annotation_lines = f.readlines()
    # shuffle dataset
    np.random.seed(10101)
    np.random.shuffle(annotation_lines)
    np.random.seed(None)

    annotation_records = {}
    classes_records = {}

    for line in annotation_lines:
        box_records = {}
        image_name = line.split(' ')[0]
        boxes = line.split(' ')[1:]
        for box in boxes:
            #strip box coordinate and class
            class_name = class_names[int(box.split(',')[-1])]
            coordinate = ','.join(box.split(',')[:-1])
            box_records[coordinate] = class_name
            #append or add ground truth class item
            if class_name in classes_records:
                classes_records[class_name].append([image_name, coordinate])
            else:
                classes_records[class_name] = list([[image_name, coordinate]])
        annotation_records[image_name] = box_records

    return annotation_records, classes_records


def get_prediction_class_records(model, annotation_records, anchors, class_names, model_image_size):
    '''
    Do the predict with YOLO model on annotation images to get predict class dict

    predict class dict would be similar with ground truth class dict:
    pred_classes_records = {
        'car': {'00001.jpg':'94,115,203,232', '00002.jpg':'82,64,154,128', ...},
        ...
    }
    '''
    pred_classes_records = {}
    for (image_name, box_records) in annotation_records.items():
        image = Image.open(image_name)
        image_data = np.array(image, dtype='uint8')
        pred_boxes, pred_classes, pred_scores = predict(model, image_data, anchors, len(class_names), model_image_size)
        print('Found {} boxes for {}'.format(len(pred_boxes), image_name))

        # Try to show out the result for every image
        # enable it when debugging

        #from predict import get_colors, draw_boxes
        #colors = get_colors(class_names)
        #image = draw_boxes(image_data, pred_boxes, pred_classes, pred_scores, class_names, colors)
        #Image.fromarray(image).show()
        #a = input('Next: ')

        # Nothing detected
        if pred_boxes is None or len(pred_boxes) == 0:
            continue

        for box, cls, score in zip(pred_boxes, pred_classes, pred_scores):
            pred_class_name = class_names[cls]
            xmin, ymin, xmax, ymax = box
            coordinate = "{},{},{},{}".format(xmin, ymin, xmax, ymax)

            #append or add predict class item
            if pred_class_name in pred_classes_records:
                pred_classes_records[pred_class_name].append([image_name, coordinate])
            else:
                pred_classes_records[pred_class_name] = list([[image_name, coordinate]])

    return pred_classes_records


def box_iou(pred_box, gt_box):
    '''
    Calculate iou for predict box and ground truth box
    Param
         pred_box: predict box coordinate
                   (xmin,ymin,xmax,ymax) format
         gt_box: ground truth box coordinate
                 (xmin,ymin,xmax,ymax) format
    Return
         iou value
    '''
    # get intersection box
    inter_box = [max(pred_box[0], gt_box[0]), max(pred_box[1], gt_box[1]), min(pred_box[2], gt_box[2]), min(pred_box[3], gt_box[3])]
    # compute overlap (IoU) = area of intersection / area of union
    pred_area = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
    gt_area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
    inter_area = (inter_box[2] - inter_box[0]) * (inter_box[3] - inter_box[1])
    union_area = pred_area + gt_area - inter_area
    return 0 if union_area == 0 else float(inter_area) / float(union_area)


def match_gt_box(pred_record, gt_records, iou_threshold=0.5):
    '''
    Search gt_records list and try to find a matching box for the predict box

    Param
         pred_record: with format ['image_file', 'xmin,ymin,xmax,ymax']
         gt_records: record list with format
                     [
                      ['image_file', 'xmin,ymin,xmax,ymax', 'usage'],
                      ['image_file', 'xmin,ymin,xmax,ymax', 'usage'],
                      ...
                     ]
         iou_threshold:

         pred_record and gt_records should be from same annotation image file

    Return
         matching gt_record index. -1 when there's no matching gt
    '''
    max_iou = 0.0
    max_index = -1
    #get predict box coordinate
    pred_box = [float(x) for x in pred_record[1].split(',')]

    for i, gt_record in enumerate(gt_records):
        #get ground truth box coordinate
        gt_box = [float(x) for x in gt_record[1].split(',')]
        iou = box_iou(pred_box, gt_box)

        # if the ground truth has been assigned to other
        # prediction, we couldn't reuse it
        if iou > max_iou and gt_record[2] == 'unused' and pred_record[0] == gt_record[0]:
            max_iou = iou
            max_index = i

    # drop the prediction if couldn't match iou threshold
    if max_iou < iou_threshold:
        max_index = -1

    return max_index

'''
def voc_ap(rec, prec):
    """
    --- Official matlab code VOC2012---
    mrec=[0 ; rec ; 1];
    mpre=[0 ; prec ; 0];
    for i=numel(mpre)-1:-1:1
        mpre(i)=max(mpre(i),mpre(i+1));
    end
    i=find(mrec(2:end)~=mrec(1:end-1))+1;
    ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    rec.insert(0, 0.0)  # insert 0.0 at begining of list
    rec.append(1.0)  # insert 1.0 at end of list
    mrec = rec[:]
    prec.insert(0, 0.0)  # insert 0.0 at begining of list
    prec.append(0.0)  # insert 0.0 at end of list
    mpre = prec[:]
    """
     This part makes the precision monotonically decreasing
      (goes from the end to the beginning)
    """
    # matlab indexes start in 1 but python in 0, so I have to do:
    #   range(start=(len(mpre) - 2), end=0, step=-1)
    # also the python function range excludes the end, resulting in:
    #   range(start=(len(mpre) - 2), end=-1, step=-1)
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    """
     This part creates a list of indexes where the recall changes
    """
    # matlab: i=find(mrec(2:end)~=mrec(1:end-1))+1;
    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            i_list.append(i)  # if it was matlab would be i + 1
    """
     The Average Precision (AP) is the area under the curve
      (numerical integration)
    """
    # matlab: ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    ap = 0.0
    for i in i_list:
        ap += ((mrec[i] - mrec[i - 1]) * mpre[i])
    return ap#, mrec, mpre
'''

def voc_ap(rec, prec, use_07_metric=False):
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
        i = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def get_rec_prec(true_positive, false_positive, gt_records):
    '''
    Calculate precision/recall based on true_positive, false_positive
    result.
    '''
    cumsum = 0
    for idx, val in enumerate(false_positive):
        false_positive[idx] += cumsum
        cumsum += val

    cumsum = 0
    for idx, val in enumerate(true_positive):
        true_positive[idx] += cumsum
        cumsum += val

    rec = true_positive[:]
    for idx, val in enumerate(true_positive):
        rec[idx] = float(true_positive[idx]) / len(gt_records)

    prec = true_positive[:]
    for idx, val in enumerate(true_positive):
        prec[idx] = float(true_positive[idx]) / (false_positive[idx] + true_positive[idx])

    return rec, prec


def calc_AP(gt_records, pred_records):
    '''
    Calculate AP value for one class records

    Param
         gt_records: ground truth records list for one class, with format:
                     [
                      ['image_file', 'xmin,ymin,xmax,ymax'],
                      ['image_file', 'xmin,ymin,xmax,ymax'],
                      ...
                     ]
         pred_record: predict records for one class, same format

    Return
         AP value for the class
    '''
    # append usage flag in gt_records for matching gt search
    gt_records = [gt_record + ['unused'] for gt_record in gt_records]

    # init true_positive and false_positive list
    nd = len(pred_records)  # number of predict data
    true_positive = [0] * nd
    false_positive = [0] * nd
    # assign predictions to ground truth objects
    for idx, pred_record in enumerate(pred_records):
        # filter out gt record from same image
        image_gt_records = [ gt_record for gt_record in gt_records if gt_record[0] == pred_record[0]]

        i = match_gt_box(pred_record, image_gt_records, iou_threshold=0.5)
        if i != -1:
            # find a valid gt obj to assign, set
            # true_positive list and mark image_gt_records.
            #
            # trick: gt_records will also be marked
            # as 'used', since image_gt_records is a
            # reference list
            image_gt_records[i][2] = 'used'
            true_positive[idx] = 1
        else:
            false_positive[idx] = 1

    # compute precision/recall
    rec, prec = get_rec_prec(true_positive, false_positive, gt_records)
    ap = voc_ap(rec, prec)

    return ap


def compute_mAP(model, annotation_file, anchors, class_names, model_image_size):
    '''
    Compute mAP for YOLO model on annotation dataset
    '''
    annotation_records, gt_classes_records = annotation_parse(annotation_file, class_names)
    pred_classes_records = get_prediction_class_records(model, annotation_records, anchors, class_names, model_image_size)

    APs = []
    #get AP value for each of the ground truth classes
    for _, class_name in enumerate(sorted(gt_classes_records.keys())):
        gt_records = gt_classes_records[class_name]
        #if we didn't detect any obj for a class, bypass it
        if class_name not in pred_classes_records:
            continue
        pred_records = pred_classes_records[class_name]
        ap = calc_AP(gt_records, pred_records)
        APs.append(ap)

    #get mAP from APs
    print((sorted(gt_classes_records.keys())))
    print(APs)
    print(np.mean(APs)*100)

    #return mAP percentage value
    return np.mean(APs)*100



def main():
    # class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument(
        '--model_path', type=str,
        help='path to model weight file')

    parser.add_argument(
        '--anchors_path', type=str,
        help='path to anchor definitions')

    parser.add_argument(
        '--classes_path', type=str,
        help='path to class definitions, default model_data/voc_classes.txt', default='model_data/voc_classes.txt')

    parser.add_argument(
        '--model_image_size', type=str,
        help='model image input size as <num>x<num>, default 416x416', default='416x416')

    parser.add_argument(
        '--annotation_file', type=str,
        help='annotation txt file to varify')

    args = parser.parse_args()

    if not args.model_path:
        raise ValueError('model file is not specified')
    if not args.annotation_file:
        raise ValueError('annotation file is not specified')

    # param parse
    model = load_model(args.model_path, compile=False)
    anchors = get_anchors(args.anchors_path)
    class_names = get_classes(args.classes_path)
    height, width = args.model_image_size.split('x')
    model_image_size = (int(height), int(width))

    mAP = compute_mAP(model, args.annotation_file, anchors, class_names, model_image_size)
    print('mAP result: {}'.format(mAP))


if __name__ == '__main__':
    main()
