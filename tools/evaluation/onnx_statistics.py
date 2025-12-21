#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to calculate FLOPs & PARAMs of a ONNX model.

Reference:
https://pypi.org/project/onnx-tool/
https://www.zhihu.com/question/518300549/answer/3212625669
https://zhuanlan.zhihu.com/p/656514769

onnx-tool could be installed with following cmd:
pip install onnx-tool
"""
import os, sys, argparse
import onnx
import onnx_tool
from onnx_tool import create_ndarray_f32


def main():
    parser = argparse.ArgumentParser(description='ONNX model profiling tool')
    parser.add_argument('--model_path', type=str, required=True, help='model file to evaluate')
    parser.add_argument('--model_input_shape', type=str, required=False, default=None, help='model input shape. separate with comma for each dim. optional')

    args = parser.parse_args()

    # load & check onnx model input tensors
    onnx_model = onnx.load(args.model_path)
    graph = onnx_model.graph

    for input_tensor in graph.input:
        print('input tensor:', input_tensor.name)
        print('Attribute:\n', input_tensor)
    if len(graph.input) > 1:
        print('NOTE! model has more than 1 input tensor. May need to update script to assign dynamic shape')

    # here we assume model has only 1 input tensor
    if args.model_input_shape:
        input_shape_list = [int(input_shape) for input_shape in args.model_input_shape.split(',')]
        onnx_tool.model_profile(args.model_path, {graph.input[0].name: create_ndarray_f32(tuple(input_shape_list))})
    else:
        onnx_tool.model_profile(args.model_path)


if __name__ == '__main__':
    main()
