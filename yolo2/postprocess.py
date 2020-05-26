#!/usr/bin/python3
# -*- coding=utf-8 -*-

import tensorflow as tf
from tensorflow.keras import backend as K


def yolo2_boxes_to_corners(box_xy, box_wh):
    """Convert YOLOv2 box predictions to bounding box corners."""
    box_mins = box_xy - (box_wh / 2.)
    box_maxes = box_xy + (box_wh / 2.)

    return K.concatenate([
        box_mins[..., 1:2],  # y_min
        box_mins[..., 0:1],  # x_min
        box_maxes[..., 1:2],  # y_max
        box_maxes[..., 0:1]  # x_max
    ])



def yolo2_filter_boxes(boxes, box_confidence, box_class_probs, threshold=.6):
    """Filter YOLOv2 boxes based on object and class confidence."""
    box_scores = box_confidence * box_class_probs
    box_classes = K.argmax(box_scores, axis=-1)
    box_class_scores = K.max(box_scores, axis=-1)
    prediction_mask = box_class_scores >= threshold

    # TODO: Expose tf.boolean_mask to Keras backend?
    boxes = tf.boolean_mask(boxes, prediction_mask)
    scores = tf.boolean_mask(box_class_scores, prediction_mask)
    classes = tf.boolean_mask(box_classes, prediction_mask)
    return boxes, scores, classes


def yolo2_head(feats, anchors, num_classes, input_shape, calc_loss=False):
    """Convert final layer features to bounding box parameters."""
    num_anchors = len(anchors)
    # Reshape to batch, height, width, num_anchors, box_params.
    anchors_tensor = K.reshape(K.constant(anchors), [1, 1, 1, num_anchors, 2])

    grid_shape = K.shape(feats)[1:3] # height, width
    grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
        [1, grid_shape[1], 1, 1])
    grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
        [grid_shape[0], 1, 1, 1])
    grid = K.concatenate([grid_x, grid_y])
    grid = K.cast(grid, K.dtype(feats))

    feats = K.reshape(
        feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])

    # Adjust preditions to each spatial grid point and anchor size.
    box_xy = (K.sigmoid(feats[..., :2]) + grid) / K.cast(grid_shape[::-1], K.dtype(feats))
    #box_wh = K.exp(feats[..., 2:4]) * anchors_tensor / K.cast(grid_shape[::-1], K.dtype(feats))
    box_wh = K.exp(feats[..., 2:4]) * anchors_tensor / K.cast(input_shape[::-1], K.dtype(feats))
    box_confidence = K.sigmoid(feats[..., 4:5])
    box_class_probs = K.softmax(feats[..., 5:])

    if calc_loss == True:
        return grid, feats, box_xy, box_wh
    return box_xy, box_wh, box_confidence, box_class_probs


def yolo2_eval(yolo_outputs,
              image_shape,
              max_boxes=10,
              score_threshold=.6,
              iou_threshold=.5):
    """Evaluate YOLOv2 model on given input batch and return filtered boxes."""
    box_xy, box_wh, box_confidence, box_class_probs = yolo_outputs
    boxes = yolo2_boxes_to_corners(box_xy, box_wh)
    boxes, scores, classes = yolo2_filter_boxes(
        boxes, box_confidence, box_class_probs, threshold=score_threshold)

    # Scale boxes back to original image shape.
    height = image_shape[0]
    width = image_shape[1]
    image_dims = K.stack([height, width, height, width])
    image_dims = K.reshape(image_dims, [1, 4])
    boxes = boxes * image_dims

    # TODO: Something must be done about this ugly hack!
    max_boxes_tensor = K.constant(max_boxes, dtype='int32')
    K.get_session().run(tf.variables_initializer([max_boxes_tensor]))
    nms_index = tf.image.non_max_suppression(
        boxes, scores, max_boxes_tensor, iou_threshold=iou_threshold)
    boxes = K.gather(boxes, nms_index)
    scores = K.gather(scores, nms_index)
    classes = K.gather(classes, nms_index)
    return boxes, scores, classes


def yolo2_correct_boxes(box_xy, box_wh, input_shape, image_shape):
    '''Get corrected boxes'''
    input_shape = K.cast(input_shape, K.dtype(box_xy))
    image_shape = K.cast(image_shape, K.dtype(box_xy))

    #reshape the image_shape tensor to align with boxes dimension
    image_shape = K.reshape(image_shape, [-1, 1, 1, 1, 2])

    new_shape = K.round(image_shape * K.min(input_shape/image_shape))
    offset = (input_shape-new_shape)/2./input_shape
    scale = input_shape/new_shape
    # reverse offset/scale to match (w,h) order
    offset = offset[..., ::-1]
    scale = scale[..., ::-1]

    box_xy = (box_xy - offset) * scale
    box_wh *= scale

    box_mins = box_xy - (box_wh / 2.)
    box_maxes = box_xy + (box_wh / 2.)
    boxes =  K.concatenate([
        box_mins[..., 0:1],  # x_min
        box_mins[..., 1:2],  # y_min
        box_maxes[..., 0:1],  # x_max
        box_maxes[..., 1:2]  # y_max
    ])

    # Scale boxes back to original image shape.
    image_wh = image_shape[..., ::-1]
    boxes *= K.concatenate([image_wh, image_wh])
    return boxes


def batched_yolo2_boxes_and_scores(feats, anchors, num_classes, input_shape, image_shape):
    '''Process Conv layer output'''
    box_xy, box_wh, box_confidence, box_class_probs = yolo2_head(feats,
        anchors, num_classes, input_shape)

    num_anchors = len(anchors)
    grid_shape = K.shape(feats)[1:3] # height, width
    total_anchor_num = grid_shape[0] * grid_shape[1] * num_anchors

    boxes = yolo2_correct_boxes(box_xy, box_wh, input_shape, image_shape)
    boxes = K.reshape(boxes, [-1, total_anchor_num, 4])
    box_scores = box_confidence * box_class_probs
    box_scores = K.reshape(box_scores, [-1, total_anchor_num, num_classes])
    return boxes, box_scores

def batched_yolo2_postprocess(args,
              anchors,
              num_classes,
              max_boxes=100,
              confidence=0.1,
              iou_threshold=0.4):
    """Postprocess for YOLOv2 model on given input and return filtered boxes."""
    yolo_outputs = args[0]
    image_shape = args[1]

    input_shape = K.shape(yolo_outputs)[1:3] * 32
    batch_size = K.shape(image_shape)[0] # batch size, tensor

    boxes, box_scores = batched_yolo2_boxes_and_scores(yolo_outputs,
        anchors, num_classes, input_shape, image_shape)

    mask = box_scores >= confidence
    max_boxes_tensor = K.constant(max_boxes, dtype='int32')

    def single_image_nms(b, batch_boxes, batch_scores, batch_classes):
        boxes_ = []
        scores_ = []
        classes_ = []
        for c in range(num_classes):
            # TODO: use keras backend instead of tf.
            class_boxes = tf.boolean_mask(boxes[b], mask[b, :, c])
            class_box_scores = tf.boolean_mask(box_scores[b, :, c], mask[b, :, c])
            nms_index = tf.image.non_max_suppression(
                class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=iou_threshold)
            class_boxes = K.gather(class_boxes, nms_index)
            class_box_scores = K.gather(class_box_scores, nms_index)
            classes = K.ones_like(class_box_scores, 'int32') * c
            boxes_.append(class_boxes)
            scores_.append(class_box_scores)
            classes_.append(classes)

        boxes_ = K.concatenate(boxes_, axis=0)
        scores_ = K.concatenate(scores_, axis=0)
        classes_ = K.concatenate(classes_, axis=0)

        batch_boxes = batch_boxes.write(b, boxes_)
        batch_scores = batch_scores.write(b, scores_)
        batch_classes = batch_classes.write(b, classes_)

        return b+1, batch_boxes, batch_scores, batch_classes

    batch_boxes = tf.TensorArray(K.dtype(boxes), size=1, dynamic_size=True)
    batch_scores = tf.TensorArray(K.dtype(box_scores), size=1, dynamic_size=True)
    batch_classes = tf.TensorArray(dtype=tf.int32, size=1, dynamic_size=True)
    _, batch_boxes, batch_scores, batch_classes = tf.while_loop(lambda b,*args: b<batch_size, single_image_nms, [0, batch_boxes, batch_scores, batch_classes])

    batch_boxes = batch_boxes.stack()
    batch_scores = batch_scores.stack()
    batch_classes = batch_classes.stack()

    return batch_boxes, batch_scores, batch_classes

