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
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
from common.utils import get_custom_objects


def mish(x):
    return x * K.tanh(K.log(1 + K.exp(K.abs(-x))) + K.maximum(x, 0))


def main():
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS, description='Convert YOLO tf.keras model to ONNX model')
    parser.add_argument('--keras_model_file', required=True, type=str, help='path to keras model file')
    parser.add_argument('--output_file', required=True, type=str, help='output onnx model file')
    parser.add_argument('--op_set', required=False, type=int, help='onnx op set', default=10)

    args = parser.parse_args()
    custom_ob_dict = get_custom_objects()
    custom_ob_dict['mish'] = mish
    model = load_model(args.keras_model_file, custom_objects=custom_object_dict)
    model.save('./tmp/')

    cmd = 'python -m tf2onnx.convert --saved_model ./tmp/ --output {} --opset {}'.format(args.output_file, args.op_set)
    process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()

    shutil.rmtree('./tmp/')


if __name__ == '__main__':
    main()

