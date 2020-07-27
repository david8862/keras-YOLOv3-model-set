#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to convert YOLO keras model to ONNX model
"""
import os
import sys
import argparse
import shutil
import subprocess
import tensorflow as tf
from tensorflow.keras.models import load_model
import keras2onnx
import onnx

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))
from common.utils import get_custom_objects

os.environ['TF_KERAS'] = '1'

def onnx_convert(keras_model_file, output_file, op_set):
    custom_object_dict = get_custom_objects()
    model = load_model(keras_model_file, custom_objects=custom_object_dict)

    # convert to onnx model
    onnx_model = keras2onnx.convert_keras(model, model.name, custom_op_conversions=custom_object_dict, target_opset=op_set)

    # save converted onnx model
    onnx.save_model(onnx_model, output_file)


def onnx_convert_with_savedmodel(keras_model_file, output_file, op_set):
    # only available for TF 2.x
    if not tf.__version__.startswith('2'):
        raise ValueError('savedmodel convert only support in TF 2.x env')

    custom_object_dict = get_custom_objects()
    model = load_model(keras_model_file, custom_objects=custom_object_dict)

    # export to saved model
    model.save('tmp_savedmodel', save_format='tf')

    # use tf2onnx to convert to onnx model
    cmd = 'python -m tf2onnx.convert --saved-model tmp_savedmodel --output {} --opset {}'.format(output_file, op_set)
    process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()

    # clean saved model
    shutil.rmtree('tmp_savedmodel')


def main():
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS, description='Convert YOLO tf.keras model to ONNX model')
    parser.add_argument('--keras_model_file', required=True, type=str, help='path to keras model file')
    parser.add_argument('--output_file', required=True, type=str, help='output onnx model file')
    parser.add_argument('--op_set', required=False, type=int, help='onnx op set, default=%(default)s', default=10)
    parser.add_argument('--with_savedmodel', default=False, action="store_true", help='use a temp savedmodel for convert')

    args = parser.parse_args()

    if args.with_savedmodel:
        onnx_convert_with_savedmodel(args.keras_model_file, args.output_file, args.op_set)
    else:
        onnx_convert(args.keras_model_file, args.output_file, args.op_set)


if __name__ == '__main__':
    main()

