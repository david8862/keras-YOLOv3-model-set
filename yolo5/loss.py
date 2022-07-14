#!/usr/bin/python3
# -*- coding=utf-8 -*-
import math
import tensorflow as tf
from tensorflow.keras import backend as K

from common.loss_utils import box_iou, box_iou_loss, softmax_focal_loss, sigmoid_focal_loss, smooth_labels
from yolo5.postprocess import yolo5_decode


def yolo5_loss(args, anchors, num_classes, ignore_thresh=.5, label_smoothing=0, elim_grid_sense=False, use_focal_loss=False, use_focal_obj_loss=False, use_softmax_loss=False, iou_loss_type='ciou'):
    '''
    YOLOv5 loss function.

    Parameters
    ----------
    yolo_outputs: list of tensor, the output of yolo_body or tiny_yolo_body
    y_true: list of array, the output of preprocess_true_boxes
    anchors: array, shape=(N, 2), wh
    num_classes: integer
    ignore_thresh: float, the iou threshold whether to ignore object confidence loss

    Returns
    -------
    loss: tensor, shape=(1,)

    '''
    num_layers = len(anchors)//3 # default setting
    yolo_outputs = args[:num_layers]
    y_true = args[num_layers:]

    # gains for box, class and confidence loss
    # from https://github.com/ultralytics/yolov5/blob/master/data/hyp.scratch.yaml
    #box_loss_gain = 0.05
    #class_loss_gain = 0.5
    #confidence_loss_gain = 1.0

    box_loss_gain = 1.0
    class_loss_gain = 1.0
    confidence_loss_gain = 1.0

    # balance weights for confidence (objectness) loss
    # on different predict heads (x/32, x/16, x/8),
    # here the order is reversed from ultralytics PyTorch version
    # from https://github.com/ultralytics/yolov5/blob/master/utils/loss.py#L109
    #confidence_balance_weights = [0.4, 1.0, 4.0]
    confidence_balance_weights = [1.0, 1.0, 1.0]

    if num_layers == 3:
        anchor_mask = [[6,7,8], [3,4,5], [0,1,2]]
        # YOLOv5 enable "elim_grid_sense" by default
        scale_x_y = [2.0, 2.0, 2.0] #if elim_grid_sense else [None, None, None]
    else:
        anchor_mask = [[3,4,5], [0,1,2]]
        scale_x_y = [1.05, 1.05] #if elim_grid_sense else [None, None]

    input_shape = K.cast(K.shape(yolo_outputs[0])[1:3] * 32, K.dtype(y_true[0]))
    grid_shapes = [K.cast(K.shape(yolo_outputs[i])[1:3], K.dtype(y_true[0])) for i in range(num_layers)]
    loss = 0
    total_location_loss = 0
    total_confidence_loss = 0
    total_class_loss = 0
    batch_size = K.shape(yolo_outputs[0])[0] # batch size, tensor
    batch_size_f = K.cast(batch_size, K.dtype(yolo_outputs[0]))

    for i in range(num_layers):
        object_mask = y_true[i][..., 4:5]
        true_class_probs = y_true[i][..., 5:]
        if label_smoothing:
            true_class_probs = smooth_labels(true_class_probs, label_smoothing)
            #true_objectness_probs = smooth_labels(object_mask, label_smoothing)
        #else:
            #true_objectness_probs = object_mask

        grid, raw_pred, pred_xy, pred_wh = yolo5_decode(yolo_outputs[i],
             anchors[anchor_mask[i]], num_classes, input_shape, scale_x_y=scale_x_y[i], calc_loss=True)
        pred_box = K.concatenate([pred_xy, pred_wh])

        # Darknet raw box to calculate loss.
        raw_true_xy = y_true[i][..., :2]*grid_shapes[i][::-1] - grid
        raw_true_wh = K.log(y_true[i][..., 2:4] / anchors[anchor_mask[i]] * input_shape[::-1])
        raw_true_wh = K.switch(object_mask, raw_true_wh, K.zeros_like(raw_true_wh)) # avoid log(0)=-inf
        box_loss_scale = 2 - y_true[i][...,2:3]*y_true[i][...,3:4]

        # Find ignore mask, iterate over each of batch.
        #ignore_mask = tf.TensorArray(K.dtype(y_true[0]), size=1, dynamic_size=True)
        #object_mask_bool = K.cast(object_mask, 'bool')
        #def loop_body(b, ignore_mask):
            #true_box = tf.boolean_mask(y_true[i][b,...,0:4], object_mask_bool[b,...,0])
            #iou = box_iou(pred_box[b], true_box)
            #best_iou = K.max(iou, axis=-1)
            #ignore_mask = ignore_mask.write(b, K.cast(best_iou<ignore_thresh, K.dtype(true_box)))
            #return b+1, ignore_mask
        #_, ignore_mask = tf.while_loop(lambda b,*args: b<batch_size, loop_body, [0, ignore_mask])
        #ignore_mask = ignore_mask.stack()
        #ignore_mask = K.expand_dims(ignore_mask, -1)


        if iou_loss_type:
            # Calculate IoU style loss as location loss
            raw_true_box = y_true[i][...,0:4]
            iou = box_iou_loss(raw_true_box, pred_box, iou_type=iou_loss_type)
            iou_loss = object_mask * box_loss_scale * (1 - iou)
            location_loss = iou_loss
        else:
            raise ValueError('Unsupported IOU loss type')
            # Standard YOLOv3 location loss
            # K.binary_crossentropy is helpful to avoid exp overflow.
            #xy_loss = object_mask * box_loss_scale * K.binary_crossentropy(raw_true_xy, raw_pred[...,0:2], from_logits=True)
            #wh_loss = object_mask * box_loss_scale * 0.5 * K.square(raw_true_wh-raw_pred[...,2:4])
            #xy_loss = K.sum(xy_loss) / batch_size_f
            #wh_loss = K.sum(wh_loss) / batch_size_f
            #location_loss = xy_loss + wh_loss

        # use box iou for positive sample as objectness ground truth (need to detach gradient),
        # to calculate confidence loss
        # from https://github.com/ultralytics/yolov5/blob/master/utils/loss.py#L127
        true_objectness_probs = tf.stop_gradient(K.maximum(iou, 0))

        if use_focal_obj_loss:
            # Focal loss for objectness confidence
            confidence_loss = confidence_balance_weights[i] * sigmoid_focal_loss(true_objectness_probs, raw_pred[...,4:5])
        else:
            #confidence_loss = K.binary_crossentropy(true_objectness_probs, raw_pred[...,4:5], from_logits=True) * confidence_balance_weights[i]
            confidence_loss = confidence_balance_weights[i] * (object_mask * K.binary_crossentropy(true_objectness_probs, raw_pred[...,4:5], from_logits=True)+ \
                (1-object_mask) * K.binary_crossentropy(object_mask, raw_pred[...,4:5], from_logits=True))# * ignore_mask

        if use_focal_loss:
            # Focal loss for classification score
            if use_softmax_loss:
                class_loss = softmax_focal_loss(true_class_probs, raw_pred[...,5:])
            else:
                class_loss = sigmoid_focal_loss(true_class_probs, raw_pred[...,5:])
        else:
            if use_softmax_loss:
                # use softmax style classification output
                class_loss = object_mask * K.expand_dims(K.categorical_crossentropy(true_class_probs, raw_pred[...,5:], from_logits=True), axis=-1)
            else:
                # use sigmoid style classification output
                class_loss = object_mask * K.binary_crossentropy(true_class_probs, raw_pred[...,5:], from_logits=True)


        confidence_loss = confidence_loss_gain * K.sum(confidence_loss) / batch_size_f
        location_loss = box_loss_gain * K.sum(location_loss) / batch_size_f
        # only involve class loss for multiple classes
        if num_classes == 1:
            class_loss = K.constant(0)
        else:
            class_loss = class_loss_gain * K.sum(class_loss) / batch_size_f


        #object_number = K.sum(object_mask)
        #divided_factor = K.switch(object_number > 1, object_number, K.constant(1, K.dtype(object_number)))

        #confidence_loss = confidence_loss_gain * K.mean(confidence_loss)
        #location_loss = box_loss_gain * K.sum(location_loss) / divided_factor

        # only involve class loss for multiple classes
        #if num_classes == 1:
            #class_loss = K.constant(0)
        #else:
            #class_loss = class_loss_gain * K.sum(class_loss) / divided_factor

        loss += location_loss + confidence_loss + class_loss
        total_location_loss += location_loss
        total_confidence_loss += confidence_loss
        total_class_loss += class_loss

    # Fit for tf 2.0.0 loss shape
    loss = K.expand_dims(loss, axis=-1)

    return loss, total_location_loss, total_confidence_loss, total_class_loss

