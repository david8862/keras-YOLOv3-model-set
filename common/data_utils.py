#!/usr/bin/python3
# -*- coding=utf-8 -*-
"""Data process utility functions."""
import numpy as np
from PIL import Image, ImageEnhance
import cv2
#from matplotlib.colors import rgb_to_hsv, hsv_to_rgb


def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a


def letterbox_resize(image, target_size, return_padding_info=False):
    """
    Resize image with unchanged aspect ratio using padding

    # Arguments
        image: origin image to be resize
            PIL Image object containing image data
        target_size: target image size,
            tuple of format (width, height).
        return_padding_info: whether to return padding size & offset info
            Boolean flag to control return value

    # Returns
        new_image: resized PIL Image object.

        padding_size: padding image size (keep aspect ratio).
            will be used to reshape the ground truth bounding box
        offset: top-left offset in target image padding.
            will be used to reshape the ground truth bounding box
    """
    src_w, src_h = image.size
    target_w, target_h = target_size

    # calculate padding scale and padding offset
    scale = min(target_w/src_w, target_h/src_h)
    padding_w = int(src_w * scale)
    padding_h = int(src_h * scale)
    padding_size = (padding_w, padding_h)

    dx = (target_w - padding_w)//2
    dy = (target_h - padding_h)//2
    offset = (dx, dy)

    # create letterbox resized image
    image = image.resize(padding_size, Image.BICUBIC)
    new_image = Image.new('RGB', target_size, (128,128,128))
    new_image.paste(image, offset)

    if return_padding_info:
        return new_image, padding_size, offset
    else:
        return new_image


def random_resize_crop_pad(image, target_size, aspect_ratio_jitter=0.3, scale_jitter=0.5):
    """
    Randomly resize image and crop|padding to target size. It can
    be used for data augment in training data preprocess

    # Arguments
        image: origin image to be resize
            PIL Image object containing image data
        target_size: target image size,
            tuple of format (width, height).
        aspect_ratio_jitter: jitter range for random aspect ratio,
            scalar to control the aspect ratio of random resized image.
        scale_jitter: jitter range for random resize scale,
            scalar to control the resize scale of random resized image.

    # Returns
        new_image: target sized PIL Image object.
        padding_size: random generated padding image size.
            will be used to reshape the ground truth bounding box
        padding_offset: random generated offset in target image padding.
            will be used to reshape the ground truth bounding box
    """
    target_w, target_h = target_size

    # generate random aspect ratio & scale for resize
    rand_aspect_ratio = target_w/target_h * rand(1-aspect_ratio_jitter,1+aspect_ratio_jitter)/rand(1-aspect_ratio_jitter,1+aspect_ratio_jitter)
    rand_scale = rand(scale_jitter, 1/scale_jitter)

    # calculate random padding size and resize
    if rand_aspect_ratio < 1:
        padding_h = int(rand_scale * target_h)
        padding_w = int(padding_h * rand_aspect_ratio)
    else:
        padding_w = int(rand_scale * target_w)
        padding_h = int(padding_w / rand_aspect_ratio)
    padding_size = (padding_w, padding_h)
    image = image.resize(padding_size, Image.BICUBIC)

    # get random offset in padding image
    dx = int(rand(0, target_w - padding_w))
    dy = int(rand(0, target_h - padding_h))
    padding_offset = (dx, dy)

    # create target image
    new_image = Image.new('RGB', (target_w, target_h), (128,128,128))
    new_image.paste(image, padding_offset)

    return new_image, padding_size, padding_offset


def reshape_boxes(boxes, src_shape, target_shape, padding_shape, offset, horizontal_flip=False, vertical_flip=False):
    """
    Reshape bounding boxes from src_shape image to target_shape image,
    usually for training data preprocess

    # Arguments
        boxes: Ground truth object bounding boxes,
            numpy array of shape (num_boxes, 5),
            box format (xmin, ymin, xmax, ymax, cls_id).
        src_shape: origin image shape,
            tuple of format (width, height).
        target_shape: target image shape,
            tuple of format (width, height).
        padding_shape: padding image shape,
            tuple of format (width, height).
        offset: top-left offset when padding target image.
            tuple of format (dx, dy).
        horizontal_flip: whether to do horizontal flip.
            boolean flag.
        vertical_flip: whether to do vertical flip.
            boolean flag.

    # Returns
        boxes: reshaped bounding box numpy array
    """
    if len(boxes)>0:
        src_w, src_h = src_shape
        target_w, target_h = target_shape
        padding_w, padding_h = padding_shape
        dx, dy = offset

        # shuffle and reshape boxes
        np.random.shuffle(boxes)
        boxes[:, [0,2]] = boxes[:, [0,2]]*padding_w/src_w + dx
        boxes[:, [1,3]] = boxes[:, [1,3]]*padding_h/src_h + dy
        # horizontal flip boxes if needed
        if horizontal_flip:
            boxes[:, [0,2]] = target_w - boxes[:, [2,0]]
        # vertical flip boxes if needed
        if vertical_flip:
            boxes[:, [1,3]] = target_h - boxes[:, [3,1]]

        # check box coordinate range
        boxes[:, 0:2][boxes[:, 0:2] < 0] = 0
        boxes[:, 2][boxes[:, 2] > target_w] = target_w
        boxes[:, 3][boxes[:, 3] > target_h] = target_h

        # check box width and height to discard invalid box
        boxes_w = boxes[:, 2] - boxes[:, 0]
        boxes_h = boxes[:, 3] - boxes[:, 1]
        boxes = boxes[np.logical_and(boxes_w>1, boxes_h>1)] # discard invalid box

    return boxes


def random_hsv_distort(image, hue=.1, sat=1.5, val=1.5):
    """
    Random distort image in HSV color space
    usually for training data preprocess

    # Arguments
        image: origin image to be resize
            PIL Image object containing image data
        hue: distort range for Hue
            scalar
        sat: distort range for Saturation
            scalar
        val: distort range for Value(Brightness)
            scalar

    # Returns
        new_image: distorted PIL Image object.
    """
    # get random HSV param
    hue = rand(-hue, hue)
    sat = rand(1, sat) if rand()<.5 else 1/rand(1, sat)
    val = rand(1, val) if rand()<.5 else 1/rand(1, val)

    # transform color space from RGB to HSV
    x = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2HSV)

    # distort image
    # cv2 HSV value range:
    #     H: [0, 179]
    #     S: [0, 255]
    #     V: [0, 255]
    x = x.astype(np.float64)
    x[..., 0] = (x[..., 0] * (1 + hue)) % 180
    x[..., 1] = x[..., 1] * sat
    x[..., 2] = x[..., 2] * val
    x[..., 1:3][x[..., 1:3]>255] = 255
    x[..., 1:3][x[..., 1:3]<0] = 0
    x = x.astype(np.uint8)

    # back to PIL RGB distort image
    x = cv2.cvtColor(x, cv2.COLOR_HSV2RGB)
    new_image = Image.fromarray(x)

    return new_image


def random_brightness(image, jitter=.5):
    """
    Random adjust brightness for image

    # Arguments
        image: origin image for brightness change
            PIL Image object containing image data
        jitter: jitter range for random brightness,
            scalar to control the random brightness level.

    # Returns
        new_image: adjusted PIL Image object.
    """
    enh_bri = ImageEnhance.Brightness(image)
    brightness = rand(jitter, 1/jitter)
    new_image = enh_bri.enhance(brightness)

    return new_image


def random_chroma(image, jitter=.5):
    """
    Random adjust chroma (color level) for image

    # Arguments
        image: origin image for chroma change
            PIL Image object containing image data
        jitter: jitter range for random chroma,
            scalar to control the random color level.

    # Returns
        new_image: adjusted PIL Image object.
    """
    enh_col = ImageEnhance.Color(image)
    color = rand(jitter, 1/jitter)
    new_image = enh_col.enhance(color)

    return new_image


def random_contrast(image, jitter=.5):
    """
    Random adjust contrast for image

    # Arguments
        image: origin image for contrast change
            PIL Image object containing image data
        jitter: jitter range for random contrast,
            scalar to control the random contrast level.

    # Returns
        new_image: adjusted PIL Image object.
    """
    enh_con = ImageEnhance.Contrast(image)
    contrast = rand(jitter, 1/jitter)
    new_image = enh_con.enhance(contrast)

    return new_image


def random_sharpness(image, jitter=.5):
    """
    Random adjust sharpness for image

    # Arguments
        image: origin image for sharpness change
            PIL Image object containing image data
        jitter: jitter range for random sharpness,
            scalar to control the random sharpness level.

    # Returns
        new_image: adjusted PIL Image object.
    """
    enh_sha = ImageEnhance.Sharpness(image)
    sharpness = rand(jitter, 1/jitter)
    new_image = enh_sha.enhance(sharpness)

    return new_image


def random_horizontal_flip(image, jitter=.5):
    """
    Random horizontal flip for image

    # Arguments
        image: origin image for horizontal flip
            PIL Image object containing image data
        jitter: jitter range for flip probability,
            scalar to control the flip probability.

    # Returns
        image: adjusted PIL Image object.
        flip: boolean flag for horizontal flip action
    """
    flip = rand() < jitter
    if flip:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)

    return image, flip


def random_vertical_flip(image, jitter=.2):
    """
    Random vertical flip for image

    # Arguments
        image: origin image for vertical flip
            PIL Image object containing image data
        jitter: jitter range for flip probability,
            scalar to control the flip probability.

    # Returns
        image: adjusted PIL Image object.
        flip: boolean flag for vertical flip action
    """
    flip = rand() < jitter
    if flip:
        image = image.transpose(Image.FLIP_TOP_BOTTOM)

    return image, flip


def random_grayscale(image, jitter=.2):
    """
    Random convert image to grayscale

    # Arguments
        image: origin image for grayscale convert
            PIL Image object containing image data
        jitter: jitter range for convert probability,
            scalar to control the convert probability.

    # Returns
        image: adjusted PIL Image object.
    """
    convert = rand() < jitter
    if convert:
        #convert to grayscale first, and then
        #back to 3 channels fake RGB
        image = image.convert('L')
        image = image.convert('RGB')

    return image


def normalize_image(image):
    """
    normalize image array from 0 ~ 255
    to 0.0 ~ 1.0

    # Arguments
        image: origin input image
            numpy image array with dtype=float, 0.0 ~ 255.0

    # Returns
        image: numpy image array with dtype=float, 0.0 ~ 1.0
    """
    image = image / 255.0

    return image


def preprocess_image(image, model_image_size):
    """
    Prepare model input image data with letterbox
    resize, normalize and dim expansion

    # Arguments
        image: origin input image
            PIL Image object containing image data
        model_image_size: model input image size
            tuple of format (height, width).

    # Returns
        image_data: numpy array of image data for model input.
    """
    #resized_image = cv2.resize(image, tuple(reversed(model_image_size)), cv2.INTER_AREA)
    resized_image = letterbox_resize(image, tuple(reversed(model_image_size)))
    image_data = np.asarray(resized_image).astype('float32')
    image_data = normalize_image(image_data)
    image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
    return image_data

