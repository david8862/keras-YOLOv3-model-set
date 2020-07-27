#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to convert YOLO keras model to an integer quantized tflite model
using latest Post-Training Integer Quantization Toolkit released in
tensorflow 2.0.0 build
"""
import os, sys, argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))
from yolo3.data import get_ground_truth_data
from common.utils import get_custom_objects

#tf.enable_eager_execution()


def post_train_quant_convert(keras_model_file, annotation_file, sample_num, model_input_shape, output_file):
    #get input_shapes for converter
    input_shapes=list((1,)+model_input_shape+(3,))

    with open(annotation_file) as f:
        annotation_lines = f.readlines()

    custom_object_dict = get_custom_objects()

    model = load_model(keras_model_file, custom_objects=custom_object_dict)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    def data_generator():
        n = len(annotation_lines)
        i = 0
        for num in range(sample_num):
            image, _ = get_ground_truth_data(annotation_lines[i], model_input_shape, augment=True)
            i = (i+1) % n
            image = np.array([image], dtype=np.float32)
            yield [image]


    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    #converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
    #converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_LATENCY]
    converter.representative_dataset = tf.lite.RepresentativeDataset(data_generator)

    #converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

    tflite_model = converter.convert()
    with open(output_file, "wb") as f:
        f.write(tflite_model)



def main():
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS, description='TF 2.x post training integer quantization converter')

    parser.add_argument('--keras_model_file', required=True, type=str, help='path to keras model file')
    parser.add_argument('--annotation_file', required=True, type=str, help='annotation txt file to feed the converter')
    parser.add_argument('--sample_num', type=int, help='annotation sample number to feed the converter,default=%(default)s', default=30)
    parser.add_argument('--model_input_shape', type=str, help='model image input shape as <height>x<width>, default=%(default)s', default='416x416')
    parser.add_argument('--output_file', required=True, type=str, help='output tflite model file')

    args = parser.parse_args()
    height, width = args.model_input_shape.split('x')
    model_input_shape = (int(height), int(width))
    assert (model_input_shape[0]%32 == 0 and model_input_shape[1]%32 == 0), 'model_input_shape should be multiples of 32'

    post_train_quant_convert(args.keras_model_file, args.annotation_file, args.sample_num, model_input_shape, args.output_file)



if __name__ == '__main__':
    main()

