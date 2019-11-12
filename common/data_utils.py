#!/usr/bin/python3
# -*- coding=utf-8 -*-
"""Data process utility functions."""

from PIL import Image
import numpy as np
import cv2


def letterbox_image(image, size):
    '''resize image with unchanged aspect ratio using padding'''
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    return new_image


def preprocess_image(image, model_image_size):
    #resized_image = cv2.resize(image, tuple(reversed(model_image_size)), cv2.INTER_AREA)
    resized_image = letterbox_image(image, tuple(reversed(model_image_size)))
    image_data = np.asarray(resized_image).astype('float32')
    image_data /= 255.
    image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
    return image_data

