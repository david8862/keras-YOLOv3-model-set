#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to simplify converted YOLO ONNX model with onnxsim

Reference from:
https://blog.csdn.net/weixin_34910922/article/details/113956622

onnxsim could be installed with following cmd:
pip install onnx-simplifier
"""
import os, sys, argparse
import onnx
from onnxsim import simplify


def onnx_simplify(input_model, output_model):
    onnx_model = onnx.load(input_model)

    model_simp, check = simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated"

    onnx.checker.check_model(model_simp)
    onnx.save(model_simp, output_model)
    print('Done. simplified model has been saved to', output_model)


def main():
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS, description='simplify YOLO onnx model with onnxsim')
    parser.add_argument('--input_model', help='input YOLO onnx model file to simplify', type=str, required=True)
    parser.add_argument('--output_model', help='output onnx model file to save', type=str, required=True)

    args = parser.parse_args()

    onnx_simplify(args.input_model, args.output_model)


if __name__ == "__main__":
    main()
