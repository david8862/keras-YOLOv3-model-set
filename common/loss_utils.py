#!/usr/bin/python3
# -*- coding=utf-8 -*-
import math
import tensorflow as tf
from tensorflow.keras import backend as K


def softmax_focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
    """
    Compute softmax focal loss.
    Reference Paper:
        "Focal Loss for Dense Object Detection"
        https://arxiv.org/abs/1708.02002

    # Arguments
        y_true: Ground truth targets,
            tensor of shape (?, num_boxes, num_classes).
        y_pred: Predicted logits,
            tensor of shape (?, num_boxes, num_classes).
        gamma: exponent of the modulating factor (1 - p_t) ^ gamma.
        alpha: optional alpha weighting factor to balance positives vs negatives.

    # Returns
        softmax_focal_loss: Softmax focal loss, tensor of shape (?, num_boxes).
    """

    # Scale predictions so that the class probas of each sample sum to 1
    #y_pred /= K.sum(y_pred, axis=-1, keepdims=True)

    # Clip the prediction value to prevent NaN's and Inf's
    #epsilon = K.epsilon()
    #y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
    y_pred = tf.nn.softmax(y_pred)
    y_pred = tf.maximum(tf.minimum(y_pred, 1 - 1e-15), 1e-15)

    # Calculate Cross Entropy
    cross_entropy = -y_true * tf.math.log(y_pred)

    # Calculate Focal Loss
    softmax_focal_loss = alpha * tf.pow(1 - y_pred, gamma) * cross_entropy

    return softmax_focal_loss


def sigmoid_focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
    """
    Compute sigmoid focal loss.
    Reference Paper:
        "Focal Loss for Dense Object Detection"
        https://arxiv.org/abs/1708.02002

    # Arguments
        y_true: Ground truth targets,
            tensor of shape (?, num_boxes, num_classes).
        y_pred: Predicted logits,
            tensor of shape (?, num_boxes, num_classes).
        gamma: exponent of the modulating factor (1 - p_t) ^ gamma.
        alpha: optional alpha weighting factor to balance positives vs negatives.

    # Returns
        sigmoid_focal_loss: Sigmoid focal loss, tensor of shape (?, num_boxes).
    """
    sigmoid_loss = K.binary_crossentropy(y_true, y_pred, from_logits=True)

    pred_prob = tf.sigmoid(y_pred)
    p_t = ((y_true * pred_prob) + ((1 - y_true) * (1 - pred_prob)))
    modulating_factor = tf.pow(1.0 - p_t, gamma)
    alpha_weight_factor = (y_true * alpha + (1 - y_true) * (1 - alpha))

    sigmoid_focal_loss = modulating_factor * alpha_weight_factor * sigmoid_loss
    #sigmoid_focal_loss = tf.reduce_sum(sigmoid_focal_loss, axis=-1)

    return sigmoid_focal_loss


def box_iou(b1, b2, expand_dims=True):
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
    if expand_dims:
        # Expand dim to apply broadcasting.
        b1 = K.expand_dims(b1, -2)

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



def box_iou_loss(b_true, b_pred, iou_type='ciou'):
    """
    Calculate IoU style (IoU/GIoU/DIoU/CIoU/SIoU) bbox loss on anchor boxes

    Parameters
    ----------
    b_true: GT boxes tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh
    b_pred: predict boxes tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh
    iou_type: string flag to indicate IoU loss type

    Returns
    -------
    iou: iou loss tensor, shape=(batch, feat_w, feat_h, anchor_num, 1)
    """
    if iou_type:
        iou_type = iou_type.lower()

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

    if iou_type == 'iou':
        """
        normal IoU loss, not recommended
        """
        iou = K.expand_dims(iou, -1)

    elif iou_type == 'giou':
        """
        GIoU loss
        Reference Paper:
            "Generalized Intersection over Union: A Metric and A Loss for Bounding Box Regression"
            https://arxiv.org/abs/1902.09630
        """
        enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]
        # calculate GIoU, add epsilon in denominator to avoid dividing by 0
        giou = iou - 1.0 * (enclose_area - union_area) / (enclose_area + K.epsilon())
        iou = K.expand_dims(giou, -1)

    elif iou_type in ['diou', 'ciou']:
        """
        DIoU/CIoU loss
        Reference Paper:
            "Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression"
            https://arxiv.org/abs/1911.08287
        """
        # box center distance
        center_distance = K.sum(K.square(b_true_xy - b_pred_xy), axis=-1)

        # get enclosed diagonal distance
        enclose_diagonal = K.sum(K.square(enclose_wh), axis=-1)
        # calculate DIoU, add epsilon in denominator to avoid dividing by 0
        diou = iou - 1.0 * (center_distance) / (enclose_diagonal + K.epsilon())

        if iou_type == 'ciou':
            # calculate param v and alpha to extend to CIoU
            v = (4 / math.pi ** 2) * K.pow(tf.math.atan2(b_true_wh[..., 0], b_true_wh[..., 1]) - tf.math.atan2(b_pred_wh[..., 0], b_pred_wh[..., 1]), 2)
            alpha = tf.stop_gradient(v / ((1.0 + K.epsilon()) - iou + v))
            diou = diou - alpha*v

        iou = K.expand_dims(diou, -1)

    elif iou_type == 'siou':
        """
        SIoU loss
        Reference Paper:
            "SIoU Loss: More Powerful Learning for Bounding Box Regression"
            https://arxiv.org/abs/2205.12740
            https://github.com/meituan/YOLOv6/blob/main/yolov6/utils/figure_iou.py
        """
        center_distance_x = b_pred_xy[..., 0] - b_true_xy[..., 0]
        center_distance_y = b_pred_xy[..., 1] - b_true_xy[..., 1]

        # angle cost
        sigma = K.pow(center_distance_x ** 2 + center_distance_y ** 2, 0.5) + K.epsilon()
        sin_alpha_1 = K.abs(center_distance_x) / sigma
        sin_alpha_2 = K.abs(center_distance_y) / sigma
        threshold = pow(2, 0.5) / 2
        sin_alpha = K.switch(sin_alpha_1 > threshold, sin_alpha_2, sin_alpha_1)
        angle_cost = K.cos(tf.math.asin(sin_alpha) * 2 - math.pi / 2)

        # distance cost
        rho_x = (center_distance_x / enclose_wh[..., 0]) ** 2
        rho_y = (center_distance_y / enclose_wh[..., 1]) ** 2
        gamma = angle_cost - 2
        distance_cost = 2 - K.exp(gamma * rho_x) - K.exp(gamma * rho_y)

        # shape cost
        omiga_w = K.abs(b_true_wh[..., 0] - b_pred_wh[..., 0]) / K.maximum(b_true_wh[..., 0], b_pred_wh[..., 0])
        omiga_h = K.abs(b_true_wh[..., 1] - b_pred_wh[..., 1]) / K.maximum(b_true_wh[..., 1], b_pred_wh[..., 1])
        shape_cost = K.pow(1 - K.exp(-1 * omiga_w), 4) + K.pow(1 - K.exp(-1 * omiga_h), 4)

        siou = iou - 0.5 * (distance_cost + shape_cost)
        iou = K.expand_dims(siou, -1)

    else:
        raise ValueError('Unsupported iou type')

    return iou



def smooth_labels(y_true, label_smoothing):
    """
    Smoothing target GT value for crossentropy loss

    Parameters
    ----------
    y_true: target GT value tensor
    label_smoothing: smoothing factor, float value between 0 and 1

    Returns
    -------
    smoothed y_true target value
    """
    label_smoothing = K.constant(label_smoothing, dtype=K.floatx())
    return y_true * (1.0 - label_smoothing) + 0.5 * label_smoothing

