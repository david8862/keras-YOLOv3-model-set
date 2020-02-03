# -*- coding=utf-8 -*-
#!/usr/bin/python3

import math
from tensorflow.keras import backend as K
from yolo2.postprocess import yolo2_head


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


def box_diou(b1, b2):
    """
    Calculate DIoU loss on anchor boxes
    Reference Paper:
        "Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression"
        https://arxiv.org/abs/1911.08287

    Parameters
    ----------
    b1: tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh
    b2: tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh

    Returns
    -------
    diou: tensor, shape=(batch, feat_w, feat_h, anchor_num, 1)
    """
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh/2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half

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
    union_area = b1_area + b2_area - intersect_area
    # calculate IoU, add epsilon in denominator to avoid dividing by 0
    iou = intersect_area / (union_area + K.epsilon())

    # box center distance
    center_distance = K.sum(K.square(b1_xy - b2_xy), axis=-1)
    # get enclosed area
    enclose_mins = K.minimum(b1_mins, b2_mins)
    enclose_maxes = K.maximum(b1_maxes, b2_maxes)
    enclose_wh = K.maximum(enclose_maxes - enclose_mins, 0.0)
    # get enclosed diagonal distance
    enclose_diagonal = K.sum(K.square(enclose_wh), axis=-1)
    # calculate DIoU, add epsilon in denominator to avoid dividing by 0
    diou = iou - 1.0 * (center_distance) / (enclose_diagonal + K.epsilon())

    # calculate param v and alpha to extend to CIoU
    #v = 4*K.square(tf.math.atan2(b1_wh[..., 0], b1_wh[..., 1]) - tf.math.atan2(b2_wh[..., 0], b2_wh[..., 1])) / (math.pi * math.pi)
    #alpha = v / (1.0 - iou + v)
    #diou = diou - alpha*v

    diou = K.expand_dims(diou, -1)
    return diou


def _smooth_labels(y_true, label_smoothing):
    label_smoothing = K.constant(label_smoothing, dtype=K.floatx())
    return y_true * (1.0 - label_smoothing) + 0.5 * label_smoothing


def yolo2_loss(args, anchors, num_classes, label_smoothing=0, use_crossentropy_loss=False, use_crossentropy_obj_loss=False, rescore_confidence=False, use_diou_loss=False):
    """YOLOv2 loss function.

    Parameters
    ----------
    yolo_output : tensor
        Final convolutional layer features.

    true_boxes : tensor
        Ground truth boxes tensor with shape [batch, num_true_boxes, 5]
        containing box x_center, y_center, width, height, and class.

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
    (yolo_output, true_boxes, y_true) = args
    num_anchors = len(anchors)
    yolo_output_shape = K.shape(yolo_output)
    input_shape = yolo_output_shape[1:3] * 32
    batch_size_f = K.cast(yolo_output_shape[0], K.dtype(yolo_output)) # batch size, float tensor
    object_scale = 5
    no_object_scale = 1
    class_scale = 1
    location_scale = 1

    pred_xy, pred_wh, pred_confidence, pred_class_prob = yolo2_head(
        yolo_output, anchors, num_classes, input_shape)

    object_mask = y_true[..., 4:5]

    # Expand pred x,y,w,h to allow comparison with ground truth.
    # batch, conv_height, conv_width, num_anchors, num_true_boxes, box_params
    pred_boxes = K.concatenate([pred_xy, pred_wh])
    pred_boxes = K.expand_dims(pred_boxes, 4)

    # reshape true_boxes to:
    # batch, conv_height, conv_width, num_anchors, num_true_boxes, box_params
    true_boxes_shape = K.shape(true_boxes)
    true_boxes = K.reshape(true_boxes, [
        true_boxes_shape[0], 1, 1, 1, true_boxes_shape[1], true_boxes_shape[2]
    ])

    iou_scores = box_iou(pred_boxes, true_boxes)
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

    if use_diou_loss:
        # Calculate DIoU loss as location loss
        diou = box_diou(pred_boxes, true_boxes)
        diou = K.squeeze(diou, axis=-1)
        diou_loss = location_scale * object_mask * (1 - diou)
        location_loss = diou_loss
    else:
        # YOLOv2 location loss for matching detection boxes.
        matching_boxes = y_true[..., 0:4]

        feats = K.reshape(yolo_output, [
            -1, yolo_output_shape[1], yolo_output_shape[2], num_anchors,
            num_classes + 5
        ])
        # Unadjusted box predictions for loss.
        # TODO: Remove extra computation shared with yolo2_head.
        raw_pred_boxes = K.concatenate(
            (K.sigmoid(feats[..., 0:2]), feats[..., 2:4]), axis=-1)

        location_loss = (location_scale * object_mask *
                            K.square(matching_boxes - raw_pred_boxes))


    confidence_loss_sum = K.sum(confidence_loss) / batch_size_f
    classification_loss_sum = K.sum(classification_loss) / batch_size_f
    location_loss_sum = K.sum(location_loss) / batch_size_f
    total_loss = 0.5 * (
        confidence_loss_sum + classification_loss_sum + location_loss_sum)

    # Fit for tf 2.0.0 loss shape
    total_loss = K.expand_dims(total_loss, axis=-1)

    return total_loss, location_loss_sum, confidence_loss_sum, classification_loss_sum

