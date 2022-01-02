#!/usr/bin/python3
# -*- coding=utf-8 -*-
"""Data process utility functions."""
import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageFilter


def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a


def random_grayscale(image, prob=.2):
    """
    Random convert image to grayscale

    # Arguments
        image: origin image for grayscale convert
            numpy image array
        prob: probability for grayscale convert,
            scalar to control the convert probability.

    # Returns
        image: adjusted numpy image array.
    """
    convert = rand() < prob
    if convert:
        #convert to grayscale first, and then
        #back to 3 channels fake RGB
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    return image


def random_chroma(image, jitter=.5):
    """
    Random adjust chroma (color level) for image

    # Arguments
        image: origin image for grayscale convert
            numpy image array
        jitter: jitter range for random chroma,
            scalar to control the random color level.

    # Returns
        new_image: adjusted numpy image array.
    """
    enh_col = ImageEnhance.Color(Image.fromarray(image.astype(np.uint8)))
    color = rand(jitter, 1/jitter)
    new_image = enh_col.enhance(color)

    return np.array(new_image)


def random_contrast(image, jitter=.5):
    """
    Random adjust contrast for image

    # Arguments
        image: origin image for contrast change
            numpy image array
        jitter: jitter range for random contrast,
            scalar to control the random contrast level.

    # Returns
        new_image: adjusted numpy image array.
    """
    enh_con = ImageEnhance.Contrast(Image.fromarray(image.astype(np.uint8)))
    contrast = rand(jitter, 1/jitter)
    new_image = enh_con.enhance(contrast)

    return np.array(new_image)


def random_sharpness(image, jitter=.5):
    """
    Random adjust sharpness for image

    # Arguments
        image: origin image for sharpness change
            numpy image array
        jitter: jitter range for random sharpness,
            scalar to control the random sharpness level.

    # Returns
        new_image: adjusted numpy image array.
    """
    enh_sha = ImageEnhance.Sharpness(Image.fromarray(image.astype(np.uint8)))
    sharpness = rand(jitter, 1/jitter)
    new_image = enh_sha.enhance(sharpness)

    return np.array(new_image)


def normalize_image(image):
    """
    normalize image array from 0 ~ 255
    to -1.0 ~ 1.0

    # Arguments
        image: origin input image
            numpy image array with dtype=float, 0.0 ~ 255.0

    # Returns
        image: numpy image array with dtype=float, -1.0 ~ 1.0
    """
    image = image.astype(np.float32) / 127.5 - 1

    return image


def denormalize_image(image):
    """
    Denormalize image array from -1.0 ~ 1.0
    to 0 ~ 255

    # Arguments
        image: normalized image array with dtype=float, -1.0 ~ 1.0

    # Returns
        image: numpy image array with dtype=uint8, 0 ~ 255
    """
    image = (image * 127.5 + 127.5).astype(np.uint8)

    return image


def preprocess_image(image, model_input_shape):
    """
    Prepare model input image data with
    resize, normalize and dim expansion

    # Arguments
        image: origin input image
            PIL Image object containing image data
        model_input_shape: model input image shape
            tuple of format (height, width).

    # Returns
        image_data: numpy array of image data for model input.
    """
    resized_image = image.resize(model_input_shape[::-1], Image.BICUBIC)
    image_data = np.asarray(resized_image).astype('float32')
    image_data = normalize_image(image_data)
    image_data = np.expand_dims(image_data, 0)
    return image_data

