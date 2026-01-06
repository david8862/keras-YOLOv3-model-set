#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to convert YOLO tf.keras model to tflite model
"""
import os, sys, argparse
import tensorflow as tf
from tensorflow.keras.models import load_model

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))
from common.utils import get_custom_objects


def keras_to_tflite(keras_model_file, output_file):
    # load tf.keras model
    custom_object_dict = get_custom_objects()
    model = load_model(keras_model_file, compile=False, custom_objects=custom_object_dict)

    # convert the model
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # save the model
    with open(output_file, 'wb') as f:
        f.write(tflite_model)
        print('\nDone. TFLITE model has been saved to', output_file)


def main():
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS, description='Convert YOLO tf.keras model to tflite model')
    parser.add_argument('--keras_model_file', required=True, type=str, help='path to keras model file')
    parser.add_argument('--output_file', required=False, type=str, default='yolo.tflite', help='output tflite model file, default=%(default)s')

    args = parser.parse_args()

    keras_to_tflite(args.keras_model_file, args.output_file)


if __name__ == '__main__':
    main()
