#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to convert YOLO tensorflow model to RKNN model

You need to install rknn-toolkit to run this script. And may
have version dependency with rknn-toolkit and board SDK.

Now it's verified on rknn-toolkit-v1.6.0:
https://github.com/rockchip-linux/rknn-toolkit/releases/download/v1.6.0/rknn-toolkit-v1.6.0-packages.tar.gz
"""
import argparse
from rknn.api import RKNN


def rknn_convert(input_model, output_model, model_input_shape, output_tensor_num, dataset_file, target_platform):
    # Create RKNN object
    rknn = RKNN()
    print('--> config model')
    rknn.config(channel_mean_value='0 0 0 255', reorder_channel='0 1 2', batch_size=1, target_platform=target_platform)

    # Load tensorflow model
    print('--> Loading model')
    if output_tensor_num == 1:
        output_tensor_names = ['predict_conv/BiasAdd']
    elif output_tensor_num == 2:
        output_tensor_names = ['predict_conv_1/BiasAdd', 'predict_conv_2/BiasAdd']
    elif output_tensor_num == 3:
        output_tensor_names = ['predict_conv_1/BiasAdd', 'predict_conv_2/BiasAdd', 'predict_conv_3/BiasAdd']
    else:
        raise ValueError('invalid output tensor number ', output_tensor_num)


    ret = rknn.load_tensorflow(tf_pb=input_model,
                               inputs=['image_input'],
                               outputs=output_tensor_names,
                               input_size_list=[model_input_shape+(3,)],
                               predef_file=None)
    #ret = rknn.load_onnx(model=input_model)
    if ret != 0:
        print('Load failed!')
        exit(ret)

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=True, dataset=dataset_file, pre_compile=True)
    if ret != 0:
        print('Build  failed!')
        exit(ret)

    # Export rknn model
    print('--> Export RKNN model')
    ret = rknn.export_rknn(output_model)
    if ret != 0:
        print('Export .rknn failed!')
        exit(ret)

    # Release RKNN object
    rknn.release()


def main():
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS, description='Convert YOLO tensorflow model to RKNN model')
    parser.add_argument('--input_model', required=True, type=str, help='tensorflow pb model file')
    parser.add_argument('--output_model', required=True, type=str, help='output rknn model file')
    parser.add_argument('--model_input_shape', type=str, help='model image input shape as <height>x<width>, default=%(default)s', default='416x416')
    parser.add_argument('--output_tensor_num', required=False, type=int, default=3, choices=[1, 2, 3], help='YOLO output tensor number, default=%(default)s')
    parser.add_argument('--dataset_file', required=True, type=str, help='data samples txt file')
    parser.add_argument('--target_platform', required=False, type=str, default='rv1126', choices=['rk1808', 'rk3399pro', 'rv1109', 'rv1126'], help='target Rockchip platform, default=%(default)s')

    args = parser.parse_args()
    height, width = args.model_input_shape.split('x')
    model_input_shape = (int(height), int(width))

    rknn_convert(args.input_model, args.output_model, model_input_shape, args.output_tensor_num, args.dataset_file, args.target_platform)


if __name__ == '__main__':
    main()

