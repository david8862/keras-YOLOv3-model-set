#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tool to convert opset version for onnx model

onnx could be installed with following cmd:
pip install onnx
"""
import os, sys, argparse
import onnx
from onnx import version_converter


def onnx_version_convert(input_model, target_version, output_model):
    onnx_model = onnx.load(input_model)

    converted_model = version_converter.convert_version(onnx_model, target_version)

    onnx.checker.check_model(converted_model)
    onnx.save(converted_model, output_model)
    print('Done. onnx model with opset version {} has been saved to {}'.format(target_version, output_model))


def main():
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS, description='tool to convert opset version for onnx model')
    parser.add_argument('--input_model', type=str, required=True,
           help='input onnx model file to convert')
    parser.add_argument('--target_version', type=int, required=False, default=11,
           help='target opset version. default=%(default)s')
    parser.add_argument('--output_model', type=str, required=True,
           help='output onnx model file to save')

    args = parser.parse_args()

    onnx_version_convert(args.input_model, args.target_version, args.output_model)


if __name__ == "__main__":
    main()
