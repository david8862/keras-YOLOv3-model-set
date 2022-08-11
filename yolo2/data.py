#!/usr/bin/python3
# -*- coding=utf-8 -*-
"""training data generation functions."""
import numpy as np
import random, math
from PIL import Image
from tensorflow.keras.utils import Sequence
from common.data_utils import random_mosaic_augment, random_mosaic_augment_v5
from common.utils import get_multiscale_list
from yolo3.data import get_ground_truth_data


def transform_box_info(boxes, image_size):
    """
    Transform box info to (x_center, y_center, box_width, box_height, cls_id)
    and image relative coordinate. This is for YOLOv2 y_true data
    """
    # center-lized box coordinate
    boxes_xy = 0.5 * (boxes[..., 0:2] + boxes[..., 2:4])
    boxes_wh = boxes[..., 2:4] - boxes[..., 0:2]
    # transform to relative coordinate
    boxes_xy = boxes_xy / image_size
    boxes_wh = boxes_wh / image_size
    boxes = np.concatenate((boxes_xy, boxes_wh, boxes[..., 4:5]), axis=-1)

    return boxes


def preprocess_true_boxes(true_boxes, anchors, input_shape, num_classes, multi_anchor_assign, iou_thresh=0.2):
    """Find detector in YOLO where ground truth box should appear.

    Parameters
    ----------
    true_boxes : array
        List of ground truth boxes in form of (xmin, ymin, xmax, ymax, cls_id).
        Relative coordinates are in the range [0, 1] indicating a percentage
        of the original image dimensions.
    anchors : array
        List of anchors in form of w, h.
        Anchors are assumed to be in the range [0, conv_size] where conv_size
        is the spatial dimension of the final convolutional features.
    input_shape : array-like
        List of model input image dimensions in form of h, w in pixels.
    num_classes : scalar
        Number of train classes
    multi_anchor_assign: boolean, whether to use iou_thresh to assign multiple
                         anchors for a single ground truth

    Returns
    -------
    y_true: array
        y_true feature map array with shape [conv_height, conv_width, num_anchors, 6]
        in form of relative x, y, w, h, objectness, class
    """
    assert (true_boxes[..., 4] < num_classes).all(), 'class id must be less than num_classes'
    height, width = input_shape
    num_anchors = len(anchors)
    # Downsampling factor of 5x 2-stride max_pools == 32.
    # TODO: Remove hardcoding of downscaling calculations.
    assert height % 32 == 0, 'Image sizes in YOLO_v2 must be multiples of 32.'
    assert width % 32 == 0, 'Image sizes in YOLO_v2 must be multiples of 32.'
    conv_height = height // 32
    conv_width = width // 32

    #Transform box info to (x_center, y_center, box_width, box_height, cls_id)
    #and image relative coordinate.
    true_boxes = np.array(true_boxes, dtype='float32')
    input_shape = np.array(input_shape, dtype='int32')
    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]
    true_boxes[..., 0:2] = boxes_xy / input_shape[::-1]
    true_boxes[..., 2:4] = boxes_wh / input_shape[::-1]

    num_box_params = true_boxes.shape[1]
    y_true = np.zeros(
        (conv_height, conv_width, num_anchors, num_box_params+1),
        dtype=np.float32)

    for box in true_boxes:
        # bypass invalid true_box, if w,h is 0
        if box[2] == 0 and box[3] == 0:
            continue

        box_class = box[4:5]
        i = np.floor(box[1] * conv_height).astype('int')
        j = np.floor(box[0] * conv_width).astype('int')
        best_iou = 0
        #best_anchor = 0
        selected_anchors = []
        for k, anchor in enumerate(anchors):
            # Find IOU between box shifted to origin and anchor box.
            box_maxes = box[2:4] * input_shape[::-1] / 2.
            box_mins = -box_maxes
            anchor_maxes = (anchor / 2.)
            anchor_mins = -anchor_maxes

            intersect_mins = np.maximum(box_mins, anchor_mins)
            intersect_maxes = np.minimum(box_maxes, anchor_maxes)
            intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
            intersect_area = intersect_wh[0] * intersect_wh[1]
            box_area = (box[2] * input_shape[1]) * (box[3] * input_shape[0])
            anchor_area = anchor[0] * anchor[1]
            iou = intersect_area / (box_area + anchor_area - intersect_area)

            # select 1 best anchor or multiple anchors
            if multi_anchor_assign:
                if iou > iou_thresh:
                    best_iou = iou
                    selected_anchors.append(k)
            else:
                if iou > best_iou:
                    best_iou = iou
                    # make sure we only pick the best 1 anchor
                    if len(selected_anchors) == 0:
                        selected_anchors.append(k)
                    else:
                        selected_anchors[0] = k
                    assert len(selected_anchors) == 1, 'over select anchors!'

        # Got selected anchor and assign to true box
        if best_iou > 0:
            adjusted_box = np.array(
                [
                    box[0], box[1],
                    box[2],
                    box[3],
                    1,
                    box_class
                ],
                dtype=np.float32)
            for selected_anchor in selected_anchors:
                y_true[i, j, selected_anchor] = adjusted_box
    return y_true


def get_y_true_data(box_data, anchors, input_shape, num_classes, multi_anchor_assign):
    '''
    Precompute y_true feature map data on a batch for training.
    y_true feature map array gives the regression targets for the ground truth
    box with shape [conv_height, conv_width, num_anchors, 6]
    '''
    y_true_data = [0 for i in range(len(box_data))]
    for i, boxes in enumerate(box_data):
        y_true_data[i] = preprocess_true_boxes(boxes, anchors, input_shape, num_classes, multi_anchor_assign)

    return np.array(y_true_data)


class Yolo2DataGenerator(Sequence):
    def __init__(self, annotation_lines, batch_size, input_shape, anchors, num_classes, enhance_augment=None, rescale_interval=-1, multi_anchor_assign=False, shuffle=True, **kwargs):
        self.annotation_lines = annotation_lines
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.anchors = anchors
        self.num_classes = num_classes
        self.enhance_augment = enhance_augment
        self.multi_anchor_assign = multi_anchor_assign
        self.indexes = np.arange(len(self.annotation_lines))
        self.shuffle = shuffle
        # prepare multiscale config
        # TODO: error happens when using Sequence data generator with
        #       multiscale input shape, disable multiscale first
        if rescale_interval != -1:
            raise ValueError("tf.keras.Sequence generator doesn't support multiscale input, pls remove related config")
        #self.rescale_interval = rescale_interval
        self.rescale_interval = -1

        self.rescale_step = 0
        self.input_shape_list = get_multiscale_list()

    def __len__(self):
        # get iteration loops on each epoch
        return max(1, math.ceil(len(self.annotation_lines) / float(self.batch_size)))

    def __getitem__(self, index):
        # generate annotation indexes for every batch
        batch_indexs = self.indexes[index*self.batch_size : (index+1)*self.batch_size]
        # fetch annotation lines based on index
        batch_annotation_lines = [self.annotation_lines[i] for i in batch_indexs]

        if self.rescale_interval > 0:
            # Do multi-scale training on different input shape
            self.rescale_step = (self.rescale_step + 1) % self.rescale_interval
            if self.rescale_step == 0:
                self.input_shape = self.input_shape_list[random.randint(0, len(self.input_shape_list)-1)]

        image_data = []
        box_data = []
        for b in range(self.batch_size):
            image, box = get_ground_truth_data(batch_annotation_lines[b], self.input_shape, augment=True)
            image_data.append(image)
            box_data.append(box)
        image_data = np.array(image_data)
        box_data = np.array(box_data)

        if self.enhance_augment == 'mosaic':
            # add random mosaic augment on batch ground truth data
            image_data, box_data = random_mosaic_augment(image_data, box_data, prob=0.2)
        #elif self.enhance_augment == 'mosaic_v5':
            # mosaic augment from YOLOv5
            #image_data, box_data = random_mosaic_augment_v5(image_data, box_data, prob=0.2)

        y_true_data = get_y_true_data(box_data, self.anchors, self.input_shape, self.num_classes, self.multi_anchor_assign)

        return [image_data, y_true_data], np.zeros(self.batch_size)

    def on_epoch_end(self):
        # shuffle annotation data on epoch end
        if self.shuffle == True:
            np.random.shuffle(self.annotation_lines)



def yolo2_data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes, enhance_augment, rescale_interval, multi_anchor_assign):
    '''data generator for fit_generator'''
    n = len(annotation_lines)
    i = 0
    # prepare multiscale config
    rescale_step = 0
    input_shape_list = get_multiscale_list()
    while True:
        if rescale_interval > 0:
            # Do multi-scale training on different input shape
            rescale_step = (rescale_step + 1) % rescale_interval
            if rescale_step == 0:
                input_shape = input_shape_list[random.randint(0,len(input_shape_list)-1)]

        image_data = []
        box_data = []
        for b in range(batch_size):
            if i==0:
                np.random.shuffle(annotation_lines)
            image, box = get_ground_truth_data(annotation_lines[i], input_shape, augment=True)
            image_data.append(image)
            box_data.append(box)
            i = (i+1) % n
        image_data = np.array(image_data)
        box_data = np.array(box_data)

        if enhance_augment == 'mosaic':
            # add random mosaic augment on batch ground truth data
            image_data, box_data = random_mosaic_augment(image_data, box_data, prob=0.2)
        #elif enhance_augment == 'mosaic_v5':
            # mosaic augment from YOLOv5
            #image_data, box_data = random_mosaic_augment_v5(image_data, box_data, prob=0.2)

        y_true_data = get_y_true_data(box_data, anchors, input_shape, num_classes, multi_anchor_assign)

        yield [image_data, y_true_data], np.zeros(batch_size)


def yolo2_data_generator_wrapper(annotation_lines, batch_size, input_shape, anchors, num_classes, enhance_augment=None, rescale_interval=-1, multi_anchor_assign=False, **kwargs):
    n = len(annotation_lines)
    if n == 0 or batch_size <= 0:
        return None

    return yolo2_data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes, enhance_augment, rescale_interval, multi_anchor_assign)

