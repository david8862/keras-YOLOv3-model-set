import logging
from zipfile import Path
logging.getLogger('tensorflow').setLevel(logging.DEBUG)

import tensorflow as tf
import numpy as np
assert float(tf.__version__[:3]) >= 2.3

import matplotlib.pylab as plt
import os
import cv2

learning_set_dir = os.path.join(os.getcwd(), 'quantization_dataset')
learning_set = []

for filename in os.listdir(learning_set_dir):
    img = cv2.imread(os.path.join(learning_set_dir, filename))
    learning_set.append(cv2.resize(img, (416, 416)))

def representative_dataset():
    for img in tf.data.Dataset.from_tensor_slices(learning_set).batch(1).take(100):
        yield [tf.cast(img, tf.float32) / 255.0]

saved_keras_dir = os.path.join(os.getcwd(), 'weights', 'yolov3-tiny.h5')
saved_keras_model = tf.keras.models.load_model(saved_keras_dir)

converter = tf.lite.TFLiteConverter.from_keras_model(saved_keras_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
tflite_quant_model = converter.convert()

with open('quant_model.tflite', 'wb') as f:
    f.write(tflite_quant_model)

# tflite_model = converter.convert()
# with open('tfl_model.tflite', 'wb') as f:
#     f.write(tflite_model)
