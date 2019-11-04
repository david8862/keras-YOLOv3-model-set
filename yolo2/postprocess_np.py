#!/usr/bin/python3
# -*- coding=utf-8 -*-
import numpy as np
import copy
from scipy.special import expit, softmax
from yolo3.postprocess_np import yolo3_handle_predictions, yolo3_adjust_boxes


def yolo2_head(prediction, anchors, num_classes, input_dims):
    batch_size = np.shape(prediction)[0]
    stride = input_dims[0] // np.shape(prediction)[1]
    grid_size = input_dims[0] // stride
    num_anchors = len(anchors)

    prediction = np.reshape(prediction,
                            (batch_size, num_anchors * grid_size * grid_size, num_classes + 5))

    box_xy = expit(prediction[:, :, :2])  # t_x (box x and y coordinates)
    objectness = expit(prediction[:, :, 4])  # p_o (objectness score)
    objectness = np.expand_dims(objectness, 2)  # To make the same number of values for axis 0 and 1

    grid = np.arange(grid_size)
    a, b = np.meshgrid(grid, grid)

    x_offset = np.reshape(a, (-1, 1))
    y_offset = np.reshape(b, (-1, 1))

    x_y_offset = np.concatenate((x_offset, y_offset), axis=1)
    x_y_offset = np.tile(x_y_offset, (1, num_anchors))
    x_y_offset = np.reshape(x_y_offset, (-1, 2))
    x_y_offset = np.expand_dims(x_y_offset, 0)

    box_xy += x_y_offset

    # Log space transform of the height and width
    #anchors = [(a[0] / stride, a[1] / stride) for a in anchors]
    anchors = np.tile(anchors, (grid_size * grid_size, 1))
    anchors = np.expand_dims(anchors, 0)

    box_wh = np.exp(prediction[:, :, 2:4]) * anchors

    # Softmax class scores
    class_scores = softmax(prediction[:, :, 5:], axis=-1)
    #class_scores = expit(prediction[:, :, 5:])

    # Resize detection map back to the input image size
    box_xy *= stride
    box_wh *= stride

    # Convert centoids to top left coordinates
    box_xy -= box_wh / 2

    return np.concatenate([box_xy, box_wh, objectness, class_scores], axis=2)


def yolo2_postprocess_np(yolo_outputs, image_shape, anchors, num_classes, model_image_size, max_boxes=100, confidence=0.1, iou_threshold=0.4):
    predictions = yolo2_head(yolo_outputs, anchors, num_classes, input_dims=model_image_size)

    boxes, classes, scores = yolo3_handle_predictions(predictions,
                                                max_boxes=max_boxes,
                                                confidence=confidence,
                                                iou_threshold=iou_threshold)
    boxes = yolo3_adjust_boxes(boxes, image_shape, model_image_size)

    return boxes, classes, scores

