#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to convert YOLO keras model to ONNX model
"""
import os, sys, argparse
from tensorflow.keras.models import load_model
import keras2onnx
import onnx

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
from common.utils import get_custom_objects

#tf.enable_eager_execution()

os.environ['TF_KERAS'] = '1'

def onnx_convert(keras_model_file, output_file, op_set):
    custom_object_dict = get_custom_objects()
    model = load_model(keras_model_file, custom_objects=custom_object_dict)

    # convert to onnx model
    onnx_model = keras2onnx.convert_keras(model, model.name, custom_op_conversions=custom_object_dict, target_opset=op_set)

    # save converted onnx model
    onnx.save_model(onnx_model, output_file)


def main():
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS, description='Convert YOLO tf.keras model to ONNX model')
    parser.add_argument('--keras_model_file', required=True, type=str, help='path to keras model file')
    parser.add_argument('--output_file', required=True, type=str, help='output onnx model file')
    parser.add_argument('--op_set', required=False, type=int, help='onnx op set', default=None)

    args = parser.parse_args()

    onnx_convert(args.keras_model_file, args.output_file, args.op_set)


if __name__ == '__main__':
    main()

