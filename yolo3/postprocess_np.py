#!/usr/bin/python3
# -*- coding=utf-8 -*-

import numpy as np
from common.yolo_postprocess_np import yolo_head, yolo_handle_predictions, yolo_correct_boxes, yolo_adjust_boxes


def yolo3_head(predictions, anchors, num_classes, input_dims):
    """
    YOLOv3 Head to process predictions from YOLOv3 models

    :param num_classes: Total number of classes
    :param anchors: YOLO style anchor list for bounding box assignment
    :param input_dims: Input dimensions of the image
    :param predictions: A list of three tensors with shape (N, 19, 19, 255), (N, 38, 38, 255) and (N, 76, 76, 255)
    :return: A tensor with the shape (N, num_boxes, 85)
    """
    assert len(predictions) == len(anchors)//3, 'anchor numbers does not match prediction.'

    if len(predictions) == 3: # assume 3 set of predictions is YOLOv3
        anchor_mask = [[6,7,8], [3,4,5], [0,1,2]]
    elif len(predictions) == 2: # 2 set of predictions is YOLOv3-tiny
        anchor_mask = [[3,4,5], [0,1,2]]
    else:
        raise ValueError('Unsupported prediction length: {}'.format(len(predictions)))

    results = []
    for i, prediction in enumerate(predictions):
        results.append(yolo_head(prediction, anchors[anchor_mask[i]], num_classes, input_dims, use_softmax=False))

    return np.concatenate(results, axis=1)


def yolo3_postprocess_np(yolo_outputs, image_shape, anchors, num_classes, model_image_size, max_boxes=100, confidence=0.1, iou_threshold=0.4):
    predictions = yolo3_head(yolo_outputs, anchors, num_classes, input_dims=model_image_size)
    predictions = yolo_correct_boxes(predictions, image_shape, model_image_size)

    boxes, classes, scores = yolo_handle_predictions(predictions,
                                                     image_shape,
                                                     max_boxes=max_boxes,
                                                     confidence=confidence,
                                                     iou_threshold=iou_threshold)

    boxes = yolo_adjust_boxes(boxes, image_shape)

    return boxes, classes, scores

