# -*- coding=utf-8 -*-
#!/usr/bin/python3

import math
import tensorflow as tf
from tensorflow.keras import backend as K
from yolo2.postprocess import yolo2_decode


def box_iou(b1, b2):
    """
    Return iou tensor

    Parameters
    ----------
    b1: tensor, shape=(i1,...,iN, 4), xywh
    b2: tensor, shape=(j, 4), xywh

    Returns
    -------
    iou: tensor, shape=(i1,...,iN, j)
    """
    # Expand dim to apply broadcasting.
    #b1 = K.expand_dims(b1, -2)
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh/2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half

    # Expand dim to apply broadcasting.
    b2 = K.expand_dims(b2, 0)
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh/2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    intersect_mins = K.maximum(b1_mins, b2_mins)
    intersect_maxes = K.minimum(b1_maxes, b2_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    iou = intersect_area / (b1_area + b2_area - intersect_area)

    return iou


def box_giou(b_true, b_pred):
    """
    Calculate GIoU loss on anchor boxes
    Reference Paper:
        "Generalized Intersection over Union: A Metric and A Loss for Bounding Box Regression"
        https://arxiv.org/abs/1902.09630

    Parameters
    ----------
    b_true: GT boxes tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh
    b_pred: predict boxes tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh

    Returns
    -------
    giou: tensor, shape=(batch, feat_w, feat_h, anchor_num, 1)
    """
    b_true_xy = b_true[..., :2]
    b_true_wh = b_true[..., 2:4]
    b_true_wh_half = b_true_wh/2.
    b_true_mins = b_true_xy - b_true_wh_half
    b_true_maxes = b_true_xy + b_true_wh_half

    b_pred_xy = b_pred[..., :2]
    b_pred_wh = b_pred[..., 2:4]
    b_pred_wh_half = b_pred_wh/2.
    b_pred_mins = b_pred_xy - b_pred_wh_half
    b_pred_maxes = b_pred_xy + b_pred_wh_half

    intersect_mins = K.maximum(b_true_mins, b_pred_mins)
    intersect_maxes = K.minimum(b_true_maxes, b_pred_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b_true_area = b_true_wh[..., 0] * b_true_wh[..., 1]
    b_pred_area = b_pred_wh[..., 0] * b_pred_wh[..., 1]
    union_area = b_true_area + b_pred_area - intersect_area
    # calculate IoU, add epsilon in denominator to avoid dividing by 0
    iou = intersect_area / (union_area + K.epsilon())

    # get enclosed area
    enclose_mins = K.minimum(b_true_mins, b_pred_mins)
    enclose_maxes = K.maximum(b_true_maxes, b_pred_maxes)
    enclose_wh = K.maximum(enclose_maxes - enclose_mins, 0.0)
    enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]
    # calculate GIoU, add epsilon in denominator to avoid dividing by 0
    giou = iou - 1.0 * (enclose_area - union_area) / (enclose_area + K.epsilon())
    giou = K.expand_dims(giou, -1)

    return giou


def box_diou(b_true, b_pred, use_ciou=True):
    """
    Calculate DIoU/CIoU loss on anchor boxes
    Reference Paper:
        "Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression"
        https://arxiv.org/abs/1911.08287

    Parameters
    ----------
    b_true: GT boxes tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh
    b_pred: predict boxes tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh
    use_ciou: bool flag to indicate whether to use CIoU loss type

    Returns
    -------
    diou: tensor, shape=(batch, feat_w, feat_h, anchor_num, 1)
    """
    b_true_xy = b_true[..., :2]
    b_true_wh = b_true[..., 2:4]
    b_true_wh_half = b_true_wh/2.
    b_true_mins = b_true_xy - b_true_wh_half
    b_true_maxes = b_true_xy + b_true_wh_half

    b_pred_xy = b_pred[..., :2]
    b_pred_wh = b_pred[..., 2:4]
    b_pred_wh_half = b_pred_wh/2.
    b_pred_mins = b_pred_xy - b_pred_wh_half
    b_pred_maxes = b_pred_xy + b_pred_wh_half

    intersect_mins = K.maximum(b_true_mins, b_pred_mins)
    intersect_maxes = K.minimum(b_true_maxes, b_pred_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b_true_area = b_true_wh[..., 0] * b_true_wh[..., 1]
    b_pred_area = b_pred_wh[..., 0] * b_pred_wh[..., 1]
    union_area = b_true_area + b_pred_area - intersect_area
    # calculate IoU, add epsilon in denominator to avoid dividing by 0
    iou = intersect_area / (union_area + K.epsilon())

    # box center distance
    center_distance = K.sum(K.square(b_true_xy - b_pred_xy), axis=-1)
    # get enclosed area
    enclose_mins = K.minimum(b_true_mins, b_pred_mins)
    enclose_maxes = K.maximum(b_true_maxes, b_pred_maxes)
    enclose_wh = K.maximum(enclose_maxes - enclose_mins, 0.0)
    # get enclosed diagonal distance
    enclose_diagonal = K.sum(K.square(enclose_wh), axis=-1)
    # calculate DIoU, add epsilon in denominator to avoid dividing by 0
    diou = iou - 1.0 * (center_distance) / (enclose_diagonal + K.epsilon())

    if use_ciou:
        # calculate param v and alpha to extend to CIoU
        v = 4*K.square(tf.math.atan2(b_true_wh[..., 0], b_true_wh[..., 1]) - tf.math.atan2(b_pred_wh[..., 0], b_pred_wh[..., 1])) / (math.pi * math.pi)

        # a trick: here we add an non-gradient coefficient w^2+h^2 to v to customize it's back-propagate,
        #          to match related description for equation (12) in original paper
        #
        #
        #          v'/w' = (8/pi^2) * (arctan(wgt/hgt) - arctan(w/h)) * (h/(w^2+h^2))          (12)
        #          v'/h' = -(8/pi^2) * (arctan(wgt/hgt) - arctan(w/h)) * (w/(w^2+h^2))
        #
        #          The dominator w^2+h^2 is usually a small value for the cases
        #          h and w ranging in [0; 1], which is likely to yield gradient
        #          explosion. And thus in our implementation, the dominator
        #          w^2+h^2 is simply removed for stable convergence, by which
        #          the step size 1/(w^2+h^2) is replaced by 1 and the gradient direction
        #          is still consistent with Eqn. (12).
        v = v * tf.stop_gradient(b_pred_wh[..., 0] * b_pred_wh[..., 0] + b_pred_wh[..., 1] * b_pred_wh[..., 1])

        alpha = v / (1.0 - iou + v)
        diou = diou - alpha*v

    diou = K.expand_dims(diou, -1)
    return diou


def _smooth_labels(y_true, label_smoothing):
    label_smoothing = K.constant(label_smoothing, dtype=K.floatx())
    return y_true * (1.0 - label_smoothing) + 0.5 * label_smoothing


def yolo2_loss(args, anchors, num_classes, label_smoothing=0, elim_grid_sense=False, use_crossentropy_loss=False, use_crossentropy_obj_loss=False, rescore_confidence=False, use_giou_loss=False, use_diou_loss=False):
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
    pred_boxes = K.concatenate([pred_xy, pred_wh])
    pred_boxes = K.expand_dims(pred_boxes, 4)

    raw_true_boxes = y_true[...,0:4]
    raw_true_boxes = K.expand_dims(raw_true_boxes, 4)

    iou_scores = box_iou(pred_boxes, raw_true_boxes)
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
        matching_classes = _smooth_labels(matching_classes, label_smoothing)

    if use_crossentropy_loss:
        classification_loss = (class_scale * object_mask *
                           K.expand_dims(K.categorical_crossentropy(matching_classes, pred_class_prob, from_logits=False), axis=-1))
    else:
        classification_loss = (class_scale * object_mask *
                           K.square(matching_classes - pred_class_prob))

    if use_giou_loss:
        # Calculate GIoU loss as location loss
        giou = box_giou(raw_true_boxes, pred_boxes)
        giou = K.squeeze(giou, axis=-1)
        giou_loss = location_scale * object_mask * (1 - giou)
        location_loss = giou_loss
    elif use_diou_loss:
        # Calculate DIoU loss as location loss
        diou = box_diou(raw_true_boxes, pred_boxes)
        diou = K.squeeze(diou, axis=-1)
        diou_loss = location_scale * object_mask * (1 - diou)
        location_loss = diou_loss
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
    classification_loss_sum = K.sum(classification_loss) / batch_size_f
    location_loss_sum = K.sum(location_loss) / batch_size_f
    total_loss = 0.5 * (
        confidence_loss_sum + classification_loss_sum + location_loss_sum)

    # Fit for tf 2.0.0 loss shape
    total_loss = K.expand_dims(total_loss, axis=-1)

    return total_loss, location_loss_sum, confidence_loss_sum, classification_loss_sum

