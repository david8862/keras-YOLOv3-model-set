#!/usr/bin/python3
# -*- coding=utf-8 -*-

from common.yolo_postprocess_np import yolo_head, yolo_handle_predictions, yolo_correct_boxes, yolo_adjust_boxes


def yolo2_postprocess_np(yolo_outputs, image_shape, anchors, num_classes, model_image_size, max_boxes=100, confidence=0.1, iou_threshold=0.4):
    predictions = yolo_head(yolo_outputs, anchors, num_classes, input_dims=model_image_size, use_softmax=True)
    predictions = yolo_correct_boxes(predictions, image_shape, model_image_size)

    boxes, classes, scores = yolo_handle_predictions(predictions,
                                                max_boxes=max_boxes,
                                                confidence=confidence,
                                                iou_threshold=iou_threshold)
    boxes = yolo_adjust_boxes(boxes, image_shape)

    return boxes, classes, scores

