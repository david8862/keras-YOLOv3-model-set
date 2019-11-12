# -*- coding=utf-8 -*-
#!/usr/bin/python3

from tensorflow.keras import backend as K
from yolo2.postprocess import yolo2_head


def _smooth_labels(y_true, label_smoothing):
    label_smoothing = K.constant(label_smoothing, dtype=K.floatx())
    return y_true * (1.0 - label_smoothing) + 0.5 * label_smoothing


def yolo2_loss(args, anchors, num_classes, label_smoothing=0, use_crossentropy_loss=False, use_crossentropy_obj_loss=False, rescore_confidence=False):
    """YOLOv2 loss function.

    Parameters
    ----------
    yolo_output : tensor
        Final convolutional layer features.

    true_boxes : tensor
        Ground truth boxes tensor with shape [batch, num_true_boxes, 5]
        containing box x_center, y_center, width, height, and class.

    detectors_mask : array
        0/1 mask for detector positions where there is a matching ground truth.

    matching_true_boxes : array
        Corresponding ground truth boxes for positive detector positions.
        Already adjusted for conv height and width.

    anchors : tensor
        Anchor boxes for model.

    num_classes : int
        Number of object classes.

    rescore_confidence : bool, default=False
        If true then set confidence target to IOU of best predicted box with
        the closest matching ground truth box.


    Returns
    -------
    mean_loss : float
        mean localization loss across minibatch
    """
    (yolo_output, true_boxes, detectors_mask, matching_true_boxes) = args
    num_anchors = len(anchors)
    object_scale = 5
    no_object_scale = 1
    class_scale = 1
    coordinates_scale = 1
    pred_xy, pred_wh, pred_confidence, pred_class_prob = yolo2_head(
        yolo_output, anchors, num_classes)

    # Unadjusted box predictions for loss.
    # TODO: Remove extra computation shared with yolo2_head.
    yolo_output_shape = K.shape(yolo_output)
    batch_size = yolo_output_shape[0] # batch size, tensor
    batch_size_f = K.cast(batch_size, K.dtype(yolo_output))

    feats = K.reshape(yolo_output, [
        -1, yolo_output_shape[1], yolo_output_shape[2], num_anchors,
        num_classes + 5
    ])
    pred_boxes = K.concatenate(
        (K.sigmoid(feats[..., 0:2]), feats[..., 2:4]), axis=-1)

    # TODO: Adjust predictions by image width/height for non-square images?
    # IOUs may be off due to different aspect ratio.

    # Expand pred x,y,w,h to allow comparison with ground truth.
    # batch, conv_height, conv_width, num_anchors, num_true_boxes, box_params
    pred_xy = K.expand_dims(pred_xy, 4)
    pred_wh = K.expand_dims(pred_wh, 4)

    pred_wh_half = pred_wh / 2.
    pred_mins = pred_xy - pred_wh_half
    pred_maxes = pred_xy + pred_wh_half

    true_boxes_shape = K.shape(true_boxes)

    # batch, conv_height, conv_width, num_anchors, num_true_boxes, box_params
    true_boxes = K.reshape(true_boxes, [
        true_boxes_shape[0], 1, 1, 1, true_boxes_shape[1], true_boxes_shape[2]
    ])
    true_xy = true_boxes[..., 0:2]
    true_wh = true_boxes[..., 2:4]

    # Find IOU of each predicted box with each ground truth box.
    true_wh_half = true_wh / 2.
    true_mins = true_xy - true_wh_half
    true_maxes = true_xy + true_wh_half

    intersect_mins = K.maximum(pred_mins, true_mins)
    intersect_maxes = K.minimum(pred_maxes, true_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

    pred_areas = pred_wh[..., 0] * pred_wh[..., 1]
    true_areas = true_wh[..., 0] * true_wh[..., 1]

    union_areas = pred_areas + true_areas - intersect_areas
    iou_scores = intersect_areas / union_areas

    # Best IOUs for each location.
    best_ious = K.max(iou_scores, axis=4)  # Best IOU scores.
    best_ious = K.expand_dims(best_ious)

    # A detector has found an object if IOU > thresh for some true box.
    object_detections = K.cast(best_ious > 0.6, K.dtype(best_ious))

    # TODO: Darknet region training includes extra coordinate loss for early
    # training steps to encourage predictions to match anchor priors.

    # Determine confidence weights from object and no_object weights.
    # NOTE: YOLOv2 does not use binary cross-entropy. Here we try it.
    no_object_weights = (no_object_scale * (1 - object_detections) *
                         (1 - detectors_mask))
    if use_crossentropy_obj_loss:
        no_objects_loss = no_object_weights * K.binary_crossentropy(K.zeros(K.shape(pred_confidence)), pred_confidence, from_logits=False)

        if rescore_confidence:
            objects_loss = (object_scale * detectors_mask *
                            K.binary_crossentropy(best_ious, pred_confidence, from_logits=False))
        else:
            objects_loss = (object_scale * detectors_mask *
                            K.binary_crossentropy(K.ones(K.shape(pred_confidence)), pred_confidence, from_logits=False))
    else:
        no_objects_loss = no_object_weights * K.square(-pred_confidence)

        if rescore_confidence:
            objects_loss = (object_scale * detectors_mask *
                            K.square(best_ious - pred_confidence))
        else:
            objects_loss = (object_scale * detectors_mask *
                            K.square(1 - pred_confidence))
    confidence_loss = objects_loss + no_objects_loss

    # Classification loss for matching detections.
    # NOTE: YOLOv2 does not use categorical cross-entropy loss.
    #       Here we try it.
    matching_classes = K.cast(matching_true_boxes[..., 4], 'int32')
    matching_classes = K.one_hot(matching_classes, num_classes)

    if label_smoothing:
        matching_classes = _smooth_labels(matching_classes, label_smoothing)

    if use_crossentropy_loss:
        classification_loss = (class_scale * detectors_mask *
                           K.expand_dims(K.categorical_crossentropy(matching_classes, pred_class_prob, from_logits=False), axis=-1))
    else:
        classification_loss = (class_scale * detectors_mask *
                           K.square(matching_classes - pred_class_prob))

    # Coordinate loss for matching detection boxes.
    matching_boxes = matching_true_boxes[..., 0:4]
    coordinates_loss = (coordinates_scale * detectors_mask *
                        K.square(matching_boxes - pred_boxes))

    confidence_loss_sum = K.sum(confidence_loss) / batch_size_f
    classification_loss_sum = K.sum(classification_loss) / batch_size_f
    coordinates_loss_sum = K.sum(coordinates_loss) / batch_size_f
    total_loss = 0.5 * (
        confidence_loss_sum + classification_loss_sum + coordinates_loss_sum)

    # Fit for tf 2.0.0 loss shape
    total_loss = K.expand_dims(total_loss, axis=-1)

    return total_loss, coordinates_loss_sum, confidence_loss_sum, classification_loss_sum

