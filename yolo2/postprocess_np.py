#!/usr/bin/python3
# -*- coding=utf-8 -*-

from common.yolo_postprocess_np import yolo_decode, yolo_handle_predictions, yolo_correct_boxes, yolo_adjust_boxes


def yolo2_postprocess_np(yolo_outputs, image_shape, anchors, num_classes, model_input_shape, max_boxes=100, confidence=0.1, iou_threshold=0.4, elim_grid_sense=False):

    scale_x_y = 1.05 if elim_grid_sense else None
    predictions = yolo_decode(yolo_outputs, anchors, num_classes, input_shape=model_input_shape, scale_x_y=scale_x_y, use_softmax=True)
    predictions = yolo_correct_boxes(predictions, image_shape, model_input_shape)

    boxes, classes, scores = yolo_handle_predictions(predictions,
                                                     image_shape,
                                                     num_classes,
                                                     max_boxes=max_boxes,
                                                     confidence=confidence,
                                                     iou_threshold=iou_threshold)

    boxes = yolo_adjust_boxes(boxes, image_shape)

    return boxes, classes, scores

