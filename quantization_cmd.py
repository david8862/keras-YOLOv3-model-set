import logging
from zipfile import Path
logging.getLogger('tensorflow').setLevel(logging.DEBUG)

import tensorflow as tf
import numpy as np
assert float(tf.__version__[:3]) >= 2.3

import matplotlib.pylab as plt
import os
import cv2
import argparse

def letterbox_image(image, size):
    '''resize image with unchanged aspect ratio using padding'''
    iw, ih = image.shape[0:2][::-1]
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)
    image = cv2.resize(image, (nw,nh), interpolation=cv2.INTER_CUBIC)
    new_image = np.zeros((size[1], size[0], 3), np.uint8)
    new_image.fill(128)
    dx = (w-nw)//2
    dy = (h-nh)//2
    new_image[dy:dy+nh, dx:dx+nw,:] = image
    return new_image

parser = argparse.ArgumentParser('Convert floating Keras model to integer-only tflite model.')
parser.add_argument(
    '--model', 
    default=os.path.join(os.getcwd(), 'weights', 'yolov3-tiny.h5'), 
    help='Keras model to be converted.'
)
parser.add_argument(
    '--representative_dataset', 
    required=True, 
    # default=os.path.join(os.getcwd(), 'quantization_dataset'),
    help='Path of representative dataset.'
)
parser.add_argument(
    '--output_path', 
    default=os.getcwd(), 
    help='Path of output quantized tflite model.'
)
args = parser.parse_args()

learning_set = []
for filename in os.listdir(args.representative_dataset):
    img = cv2.imread(os.path.join(args.representative_dataset, filename))
    # learning_set.append(cv2.resize(img, (416, 416)))
    learning_set.append(letterbox_image(img.copy(), (416, 416)))

def representative_dataset():
    for img in tf.data.Dataset.from_tensor_slices(learning_set).batch(1).take(100):
        yield [tf.cast(img, tf.float32) / 255.0]

saved_keras_model = tf.keras.models.load_model(args.model)

converter = tf.lite.TFLiteConverter.from_keras_model(saved_keras_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
tflite_quant_model = converter.convert()

with open(os.path.join(args.output_path, 'quant_model_new.tflite'), 'wb') as f:
    f.write(tflite_quant_model)
