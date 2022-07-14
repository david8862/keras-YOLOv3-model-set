#!/usr/bin/python3
# -*- coding=utf-8 -*-
import math
import tensorflow as tf
from tensorflow.keras import backend as K

from common.loss_utils import box_iou, box_iou_loss, smooth_labels
from yolo2.postprocess import yolo2_decode



def yolo2_loss(args, anchors, num_classes, label_smoothing=0, elim_grid_sense=False, use_crossentropy_loss=False, use_crossentropy_obj_loss=False, rescore_confidence=False, iou_loss_type=None):
    """
    YOLOv2 loss function.

    Parameters
    ----------
    yolo_output : tensor
        Final convolutional layer features.

    y_true : array
        output of preprocess_true_boxes, with shape [conv_height, conv_width, num_anchors, 6]

    anchors : tensor
        Anchor boxes for model.

    num_classes : int
        Number of object classes.

    rescore_confidence : bool, default=False
        If true then set confidence target to IOU of best predicted box with
        the closest matching ground truth box.


    Returns
    -------
    total_loss : float
        total mean YOLOv2 loss across minibatch
    """
    (yolo_output, y_true) = args
    num_anchors = len(anchors)
    scale_x_y = 1.05 if elim_grid_sense else None
    yolo_output_shape = K.shape(yolo_output)
    input_shape = K.cast(yolo_output_shape[1:3] * 32, K.dtype(y_true))
    grid_shape = K.cast(yolo_output_shape[1:3], K.dtype(y_true)) # height, width
    batch_size_f = K.cast(yolo_output_shape[0], K.dtype(yolo_output)) # batch size, float tensor
    object_scale = 5
    no_object_scale = 1
    class_scale = 1
    location_scale = 1

    grid, raw_pred, pred_xy, pred_wh = yolo2_decode(
        yolo_output, anchors, num_classes, input_shape, scale_x_y=scale_x_y, calc_loss=True)
    pred_confidence = K.sigmoid(raw_pred[..., 4:5])
    pred_class_prob = K.softmax(raw_pred[..., 5:])

    object_mask = y_true[..., 4:5]

    # Expand pred x,y,w,h to allow comparison with ground truth.
    # batch, conv_height, conv_width, num_anchors, num_true_boxes, box_params
    pred_box = K.concatenate([pred_xy, pred_wh])
    pred_box = K.expand_dims(pred_box, 4)

    raw_true_box = y_true[...,0:4]
    raw_true_box = K.expand_dims(raw_true_box, 4)

    iou_scores = box_iou(pred_box, raw_true_box, expand_dims=False)
    iou_scores = K.squeeze(iou_scores, axis=0)

    # Best IOUs for each location.
    best_ious = K.max(iou_scores, axis=4)  # Best IOU scores.
    best_ious = K.expand_dims(best_ious)

    # A detector has found an object if IOU > thresh for some true box.
    object_detections = K.cast(best_ious > 0.6, K.dtype(best_ious))

    # Determine confidence weights from object and no_object weights.
    # NOTE: YOLOv2 does not use binary cross-entropy. Here we try it.
    no_object_weights = (no_object_scale * (1 - object_detections) *
                         (1 - object_mask))
    if use_crossentropy_obj_loss:
        no_objects_loss = no_object_weights * K.binary_crossentropy(K.zeros(K.shape(pred_confidence)), pred_confidence, from_logits=False)

        if rescore_confidence:
            objects_loss = (object_scale * object_mask *
                            K.binary_crossentropy(best_ious, pred_confidence, from_logits=False))
        else:
            objects_loss = (object_scale * object_mask *
                            K.binary_crossentropy(K.ones(K.shape(pred_confidence)), pred_confidence, from_logits=False))
    else:
        no_objects_loss = no_object_weights * K.square(-pred_confidence)

        if rescore_confidence:
            objects_loss = (object_scale * object_mask *
                            K.square(best_ious - pred_confidence))
        else:
            objects_loss = (object_scale * object_mask *
                            K.square(1 - pred_confidence))
    confidence_loss = objects_loss + no_objects_loss

    # Classification loss for matching detections.
    # NOTE: YOLOv2 does not use categorical cross-entropy loss.
    #       Here we try it.
    matching_classes = K.cast(y_true[..., 5], 'int32')
    matching_classes = K.one_hot(matching_classes, num_classes)

    if label_smoothing:
        matching_classes = smooth_labels(matching_classes, label_smoothing)

    if use_crossentropy_loss:
        classification_loss = (class_scale * object_mask *
                           K.expand_dims(K.categorical_crossentropy(matching_classes, pred_class_prob, from_logits=False), axis=-1))
    else:
        classification_loss = (class_scale * object_mask *
                           K.square(matching_classes - pred_class_prob))

    if iou_loss_type:
        # Calculate IoU style loss as location loss
        iou = box_iou_loss(raw_true_box, pred_box, iou_type=iou_loss_type)
        iou = K.squeeze(iou, axis=-1)
        iou_loss = location_scale * object_mask * (1 - iou)
        location_loss = iou_loss
    else:
        # YOLOv2 location loss for matching detection boxes.
        # Darknet trans box to calculate loss.
        trans_true_xy = y_true[..., :2]*grid_shape[::-1] - grid
        trans_true_wh = K.log(y_true[..., 2:4] / anchors * input_shape[::-1])
        trans_true_wh = K.switch(object_mask, trans_true_wh, K.zeros_like(trans_true_wh)) # avoid log(0)=-inf
        trans_true_boxes = K.concatenate([trans_true_xy, trans_true_wh])

        # Unadjusted box predictions for loss.
        trans_pred_boxes = K.concatenate(
            (K.sigmoid(raw_pred[..., 0:2]), raw_pred[..., 2:4]), axis=-1)

        location_loss = (location_scale * object_mask *
                            K.square(trans_true_boxes - trans_pred_boxes))


    confidence_loss_sum = K.sum(confidence_loss) / batch_size_f
    location_loss_sum = K.sum(location_loss) / batch_size_f
    # only involve class loss for multiple classes
    if num_classes == 1:
        classification_loss_sum = K.constant(0)
    else:
        classification_loss_sum = K.sum(classification_loss) / batch_size_f

    total_loss = 0.5 * (
        confidence_loss_sum + classification_loss_sum + location_loss_sum)

    # Fit for tf 2.0.0 loss shape
    total_loss = K.expand_dims(total_loss, axis=-1)

    return total_loss, location_loss_sum, confidence_loss_sum, classification_loss_sum

