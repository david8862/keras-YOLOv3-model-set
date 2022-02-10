#!/usr/bin/python3
# -*- coding=utf-8 -*-

import numpy as np
from scipy.special import expit
from common.yolo_postprocess_np import yolo_handle_predictions, yolo_correct_boxes, yolo_adjust_boxes


def yolo5_decode_single_head(prediction, anchors, num_classes, input_shape, scale_x_y):
    '''Decode final layer features to bounding box parameters.'''
    batch_size = np.shape(prediction)[0]
    num_anchors = len(anchors)

    grid_shape = np.shape(prediction)[1:3]
    #check if stride on height & width are same
    assert input_shape[0]//grid_shape[0] == input_shape[1]//grid_shape[1], 'model stride mismatch.'
    stride = input_shape[0] // grid_shape[0]

    prediction = np.reshape(prediction,
                            (batch_size, grid_shape[0] * grid_shape[1] * num_anchors, num_classes + 5))

    ################################
    # generate x_y_offset grid map
    grid_y = np.arange(grid_shape[0])
    grid_x = np.arange(grid_shape[1])
    x_offset, y_offset = np.meshgrid(grid_x, grid_y)

    x_offset = np.reshape(x_offset, (-1, 1))
    y_offset = np.reshape(y_offset, (-1, 1))

    x_y_offset = np.concatenate((x_offset, y_offset), axis=1)
    x_y_offset = np.tile(x_y_offset, (1, num_anchors))
    x_y_offset = np.reshape(x_y_offset, (-1, 2))
    x_y_offset = np.expand_dims(x_y_offset, 0)

    ################################

    # Log space transform of the height and width
    anchors = np.tile(anchors, (grid_shape[0] * grid_shape[1], 1))
    anchors = np.expand_dims(anchors, 0)

    assert scale_x_y, 'YOLOv5 decode should have scale_x_y.'
    # YOLOv5 box decode
    #
    # Now all the prediction part (x,y,w,h,obj,cls) use
    # sigmoid(expit) for decode, so we do it together
    #
    # Reference:
    #     https://github.com/ultralytics/yolov5/blob/master/models/yolo.py#L56
    #     https://alexeyab84.medium.com/scaled-yolo-v4-is-the-best-neural-network-for-object-detection-on-ms-coco-dataset-39dfa22fa982
    prediction = expit(prediction)

    box_xy_tmp = prediction[..., :2] * scale_x_y - (scale_x_y - 1) / 2
    box_xy = (box_xy_tmp + x_y_offset) / np.array(grid_shape)[::-1]
    box_wh = ((prediction[..., 2:4]*2)**2 * anchors) / np.array(input_shape)[::-1]

    # Sigmoid objectness scores
    objectness = prediction[..., 4]  # p_o (objectness score)
    objectness = np.expand_dims(objectness, -1)  # To make the same number of values for axis 0 and 1

    # Sigmoid class scores
    class_scores = prediction[..., 5:]

    return np.concatenate([box_xy, box_wh, objectness, class_scores], axis=2)


def yolo5_decode(predictions, anchors, num_classes, input_shape, elim_grid_sense=True):
    """
    YOLOv5 Head to process predictions from YOLOv5 models

    :param num_classes: Total number of classes
    :param anchors: YOLO style anchor list for bounding box assignment
    :param input_shape: Input shape of the image
    :param predictions: A list of three tensors with shape (N, 19, 19, 255), (N, 38, 38, 255) and (N, 76, 76, 255)
    :return: A tensor with the shape (N, num_boxes, 85)
    """
    assert len(predictions) == len(anchors)//3, 'anchor numbers does not match prediction.'

    if len(predictions) == 3: # assume 3 set of predictions is YOLOv5
        anchor_mask = [[6,7,8], [3,4,5], [0,1,2]]
        # YOLOv5 enable "elim_grid_sense" by default
        scale_x_y = [2.0, 2.0, 2.0] #if elim_grid_sense else [None, None, None]
    elif len(predictions) == 2: # 2 set of predictions is YOLOv3-tiny
        anchor_mask = [[3,4,5], [0,1,2]]
        scale_x_y = [1.05, 1.05] #if elim_grid_sense else [None, None]
    else:
        raise ValueError('Unsupported prediction length: {}'.format(len(predictions)))

    results = []
    for i, prediction in enumerate(predictions):
        results.append(yolo5_decode_single_head(prediction, anchors[anchor_mask[i]], num_classes, input_shape, scale_x_y=scale_x_y[i]))

    return np.concatenate(results, axis=1)


def yolo5_postprocess_np(yolo_outputs, image_shape, anchors, num_classes, model_input_shape, max_boxes=100, confidence=0.1, iou_threshold=0.4, elim_grid_sense=True):
    # here we sort the prediction tensor list with grid size (e.g. 19/38/76)
    # to make sure it matches with anchors order
    yolo_outputs.sort(key=lambda x: x.shape[1])

    predictions = yolo5_decode(yolo_outputs, anchors, num_classes, input_shape=model_input_shape, elim_grid_sense=elim_grid_sense)
    predictions = yolo_correct_boxes(predictions, image_shape, model_input_shape)

    boxes, classes, scores = yolo_handle_predictions(predictions,
                                                     image_shape,
                                                     num_classes,
                                                     max_boxes=max_boxes,
                                                     confidence=confidence,
                                                     iou_threshold=iou_threshold)

    boxes = yolo_adjust_boxes(boxes, image_shape)

    return boxes, classes, scores

