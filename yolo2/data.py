#!/usr/bin/python3
# -*- coding=utf-8 -*-
"""training data generation functions."""

from PIL import Image
import numpy as np
import random, math
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
from tensorflow.keras.utils import Sequence
from common.utils import get_multiscale_list


def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a


def get_random_data(annotation_line, input_shape, random=True, max_boxes=20, jitter=.3, hue=.1, sat=1.5, val=1.5, proc_img=True):
    '''random preprocessing for real-time data augmentation'''
    line = annotation_line.split()
    image = Image.open(line[0])
    iw, ih = image.size
    h, w = input_shape
    img_size = np.array([w, h])
    boxes = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])

    if not random:
        # resize image
        scale = min(w/iw, h/ih)
        nw = int(iw*scale)
        nh = int(ih*scale)
        dx = (w-nw)//2
        dy = (h-nh)//2
        image_data=0
        if proc_img:
            image = image.resize((nw,nh), Image.BICUBIC)
            new_image = Image.new('RGB', (w,h), (128,128,128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image)/255.

        # correct boxes
        box_data = np.zeros((max_boxes,5))
        if len(boxes)>0:
            np.random.shuffle(boxes)
            if len(boxes)>max_boxes: boxes = boxes[:max_boxes]
            boxes[:, [0,2]] = boxes[:, [0,2]]*scale + dx
            boxes[:, [1,3]] = boxes[:, [1,3]]*scale + dy

            # Get box parameters as x_center, y_center, box_width, box_height, class.
            boxes_xy = 0.5 * (boxes[:, 0:2] + boxes[:, 2:4])
            boxes_wh = boxes[:, 2:4] - boxes[:, 0:2]
            boxes_xy = boxes_xy / img_size
            boxes_wh = boxes_wh / img_size
            boxes = np.concatenate((boxes_xy[i], boxes_wh[i], boxes[:, 4:5]), axis=1)

            box_data[:len(boxes)] = boxes

        return image_data, box_data

    # resize image
    new_ar = w/h * rand(1-jitter,1+jitter)/rand(1-jitter,1+jitter)
    scale = rand(.25, 2)
    if new_ar < 1:
        nh = int(scale*h)
        nw = int(nh*new_ar)
    else:
        nw = int(scale*w)
        nh = int(nw/new_ar)
    image = image.resize((nw,nh), Image.BICUBIC)

    # place image
    dx = int(rand(0, w-nw))
    dy = int(rand(0, h-nh))
    new_image = Image.new('RGB', (w,h), (128,128,128))
    new_image.paste(image, (dx, dy))
    image = new_image

    # flip image or not
    flip = rand()<.5
    if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

    # distort image
    hue = rand(-hue, hue)
    sat = rand(1, sat) if rand()<.5 else 1/rand(1, sat)
    val = rand(1, val) if rand()<.5 else 1/rand(1, val)
    x = rgb_to_hsv(np.array(image)/255.)
    x[..., 0] += hue
    x[..., 0][x[..., 0]>1] -= 1
    x[..., 0][x[..., 0]<0] += 1
    x[..., 1] *= sat
    x[..., 2] *= val
    x[x>1] = 1
    x[x<0] = 0
    image_data = hsv_to_rgb(x) # numpy array, 0 to 1

    # correct boxes
    box_data = np.zeros((max_boxes,5))
    if len(boxes)>0:
        np.random.shuffle(boxes)
        boxes[:, [0,2]] = boxes[:, [0,2]]*nw/iw + dx
        boxes[:, [1,3]] = boxes[:, [1,3]]*nh/ih + dy
        if flip: boxes[:, [0,2]] = w - boxes[:, [2,0]]
        boxes[:, 0:2][boxes[:, 0:2]<0] = 0
        boxes[:, 2][boxes[:, 2]>w] = w
        boxes[:, 3][boxes[:, 3]>h] = h
        box_w = boxes[:, 2] - boxes[:, 0]
        box_h = boxes[:, 3] - boxes[:, 1]
        boxes = boxes[np.logical_and(box_w>1, box_h>1)] # discard invalid box
        if len(boxes)>max_boxes: boxes = boxes[:max_boxes]

        # Get box parameters as x_center, y_center, box_width, box_height, class.
        boxes_xy = 0.5 * (boxes[:, 0:2] + boxes[:, 2:4])
        boxes_wh = boxes[:, 2:4] - boxes[:, 0:2]
        boxes_xy = boxes_xy / img_size
        boxes_wh = boxes_wh / img_size
        boxes = np.concatenate((boxes_xy, boxes_wh, boxes[:, 4:5]), axis=1)

        box_data[:len(boxes)] = boxes

    return image_data, box_data


def process_data(images, input_shape, boxes=None):
    '''processes the data'''
    images = [Image.fromarray(i) for i in images]
    orig_sizes = [np.array([i.width, i.height]) for i in images]

    # Image preprocessing.
    processed_images = [i.resize(input_shape, Image.BICUBIC) for i in images]
    processed_images = [np.array(image, dtype=np.float) for image in processed_images]
    processed_images = [image/255. for image in processed_images]

    if boxes is not None:
        # Box preprocessing.
        # Original boxes stored as 1D list of x_min, y_min, x_max, y_max, class.
        boxes = [box.reshape((-1, 5)) for box in boxes]

        # Get box parameters as x_center, y_center, box_width, box_height, class.
        boxes_xy = [0.5 * (box[:, 0:2] + box[:, 2:4]) for box in boxes]
        boxes_wh = [box[:, 2:4] - box[:, 0:2] for box in boxes]
        boxes_xy = [box_xy / orig_sizes[i] for i, box_xy in enumerate(boxes_xy)]
        boxes_wh = [box_wh / orig_sizes[i] for i, box_wh in enumerate(boxes_wh)]
        boxes = [np.concatenate((boxes_xy[i], boxes_wh[i], box[:, 4:5]), axis=1) for i, box in enumerate(boxes)]

        # find the max number of boxes
        max_boxes = 0
        for boxz in boxes:
            if boxz.shape[0] > max_boxes:
                max_boxes = boxz.shape[0]

        # add zero pad for training
        for i, boxz in enumerate(boxes):
            if boxz.shape[0]  < max_boxes:
                zero_padding = np.zeros( (max_boxes-boxz.shape[0], 5), dtype=np.float32)
                boxes[i] = np.vstack((boxz, zero_padding))

        return np.array(processed_images), np.array(boxes)
    else:
        return np.array(processed_images)


def preprocess_true_boxes(true_boxes, anchors, input_shape, num_classes):
    """Find detector in YOLO where ground truth box should appear.

    Parameters
    ----------
    true_boxes : array
        List of ground truth boxes in form of relative x, y, w, h, class.
        Relative coordinates are in the range [0, 1] indicating a percentage
        of the original image dimensions.
    anchors : array
        List of anchors in form of w, h.
        Anchors are assumed to be in the range [0, conv_size] where conv_size
        is the spatial dimension of the final convolutional features.
    input_shape : array-like
        List of model input image dimensions in form of h, w in pixels.

    Returns
    -------
    detectors_mask : array
        0/1 mask for detectors in [conv_height, conv_width, num_anchors, 1]
        that should be compared with a matching ground truth box.
    matching_true_boxes: array
        Same shape as detectors_mask with the corresponding ground truth box
        adjusted for comparison with predicted parameters at training time.
    """
    assert (true_boxes[..., 4]<num_classes).all(), 'class id must be less than num_classes'
    height, width = input_shape
    num_anchors = len(anchors)
    # Downsampling factor of 5x 2-stride max_pools == 32.
    # TODO: Remove hardcoding of downscaling calculations.
    assert height % 32 == 0, 'Image sizes in YOLO_v2 must be multiples of 32.'
    assert width % 32 == 0, 'Image sizes in YOLO_v2 must be multiples of 32.'
    conv_height = height // 32
    conv_width = width // 32
    num_box_params = true_boxes.shape[1]
    detectors_mask = np.zeros(
        (conv_height, conv_width, num_anchors, 1), dtype=np.float32)
    matching_true_boxes = np.zeros(
        (conv_height, conv_width, num_anchors, num_box_params),
        dtype=np.float32)

    for box in true_boxes:
        # scale box to convolutional feature spatial dimensions
        box_class = box[4:5]
        box = box[0:4] * np.array(
            [conv_width, conv_height, conv_width, conv_height])
        i = np.floor(box[1]).astype('int')
        j = np.floor(box[0]).astype('int')
        best_iou = 0
        best_anchor = 0
        for k, anchor in enumerate(anchors):
            # Find IOU between box shifted to origin and anchor box.
            box_maxes = box[2:4] / 2.
            box_mins = -box_maxes
            anchor_maxes = (anchor / 2.)
            anchor_mins = -anchor_maxes

            intersect_mins = np.maximum(box_mins, anchor_mins)
            intersect_maxes = np.minimum(box_maxes, anchor_maxes)
            intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
            intersect_area = intersect_wh[0] * intersect_wh[1]
            box_area = box[2] * box[3]
            anchor_area = anchor[0] * anchor[1]
            iou = intersect_area / (box_area + anchor_area - intersect_area)
            if iou > best_iou:
                best_iou = iou
                best_anchor = k

        if best_iou > 0:
            detectors_mask[i, j, best_anchor] = 1
            adjusted_box = np.array(
                [
                    box[0] - j, box[1] - i,
                    np.log(box[2] / (anchors[best_anchor][0] / 32)),
                    np.log(box[3] / (anchors[best_anchor][1] / 32)), box_class
                ],
                dtype=np.float32)
            matching_true_boxes[i, j, best_anchor] = adjusted_box
    return detectors_mask, matching_true_boxes


def get_detector_mask(boxes, anchors, input_shape, num_classes):
    '''
    Precompute detectors_mask and matching_true_boxes for training.
    Detectors mask is 1 for each spatial position in the final conv layer and
    anchor that should be active for the given boxes and 0 otherwise.
    Matching true boxes gives the regression targets for the ground truth box
    that caused a detector to be active or 0 otherwise.
    '''
    detectors_mask = [0 for i in range(len(boxes))]
    matching_true_boxes = [0 for i in range(len(boxes))]
    for i, box in enumerate(boxes):
        detectors_mask[i], matching_true_boxes[i] = preprocess_true_boxes(box, anchors, input_shape, num_classes)

    return np.array(detectors_mask), np.array(matching_true_boxes)


class Yolo2DataGenerator(Sequence):
    def __init__(self, annotation_lines, batch_size, input_shape, anchors, num_classes, rescale_interval=-1, shuffle=True):
        self.annotation_lines = annotation_lines
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.anchors = anchors
        self.num_classes = num_classes
        self.indexes = np.arange(len(self.annotation_lines))
        self.shuffle = shuffle
        # prepare multiscale config
        # TODO: error happens when using Sequence data generator with
        #       multiscale input shape, disable multiscale first
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
            image, box = get_random_data(batch_annotation_lines[b], self.input_shape, random=True)
            image_data.append(image)
            box_data.append(box)
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        detectors_mask, matching_true_boxes = get_detector_mask(box_data, self.anchors, self.input_shape, self.num_classes)

        return [image_data, box_data, detectors_mask, matching_true_boxes], np.zeros(self.batch_size)

    def on_epoch_end(self):
        # shuffle annotation data on epoch end
        if self.shuffle == True:
            np.random.shuffle(self.annotation_lines)


def yolo2_data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes, rescale_interval):
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

            image, box = get_random_data(annotation_lines[i], input_shape, random=True)
            #line = annotation_lines[i].split()
            #image = np.array(Image.open(line[0]))
            #box = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])

            image_data.append(image)
            box_data.append(box)
            i = (i+1) % n
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        #image_data, box_data = process_data(image_data, input_shape, box_data)
        detectors_mask, matching_true_boxes = get_detector_mask(box_data, anchors, input_shape, num_classes)

        yield [image_data, box_data, detectors_mask, matching_true_boxes], np.zeros(batch_size)


def yolo2_data_generator_wrapper(annotation_lines, batch_size, input_shape, anchors, num_classes, rescale_interval=-1):
    n = len(annotation_lines)
    if n==0 or batch_size<=0: return None
    return yolo2_data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes, rescale_interval)

