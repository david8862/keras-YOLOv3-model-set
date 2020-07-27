#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to convert YOLO h5 or pb model to CoreML model
"""
import os, sys, argparse
from tensorflow.keras.models import load_model
import tensorflow as tf
import tfcoreml

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))
from common.utils import get_custom_objects

def coreml_convert(input_model_file, output_file, model_image_size):
    if input_model_file.endswith('.h5'):
        if not tf.__version__.startswith('2'):
            raise ValueError('tf.keras model convert only support in TF 2.x env')
        # tf.keras h5 model
        custom_object_dict = get_custom_objects()
        keras_model = load_model(input_model_file, custom_objects=custom_object_dict)

        # get input, output node names for the TF graph from tf.keras model
        # assume only 1 input
        input_name = keras_model.inputs[0].name.split(':')[0]
        output_names = [output.name.split(':')[0].split('/')[-1] for output in keras_model.outputs]

        assert len(output_names) == 1, 'h5 model convert only support YOLOv2 family with 1 prediction output.'

    elif input_model_file.endswith('.pb'):
        # NOTE: TF 1.x frozen pb graph need to specify input/output tensor name
        # so we need to hardcode the input/output tensor names here to get them from model
        input_name = 'image_input'

        # YOLOv2 model with 1 prediction output
        #output_names = ['predict_conv/BiasAdd']

        # Tiny YOLOv3 model with 2 prediction outputs
        output_names = ['predict_conv_1/BiasAdd', 'predict_conv_2/BiasAdd']

        # YOLOv3 model with 3 prediction outputs
        #output_names = ['predict_conv_1/BiasAdd', 'predict_conv_2/BiasAdd', 'predict_conv_3/BiasAdd']
    else:
        raise ValueError('unsupported model type')

    input_name_shape_dict={input_name: (1,) + model_image_size + (3,)}
    # convert to CoreML model file
    model = tfcoreml.convert(tf_model_path=input_model_file,
                             mlmodel_path=output_file,
                             input_name_shape_dict=input_name_shape_dict,
                             output_feature_names=output_names,
                             minimum_ios_deployment_target='13')

    # save converted CoreML model
    #model.save(output_file)


def main():
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS, description='Convert YOLO h5/pb model to CoreML model')
    parser.add_argument('--input_model_file', required=True, type=str, help='path to input h5/pb model file')
    parser.add_argument('--output_file', required=True, type=str, help='output CoreML .mlmodel file')
    parser.add_argument('--model_image_size', required=False, type=str, help='model image input size as <height>x<width>, default=%(default)s', default='416x416')

    args = parser.parse_args()
    height, width = args.model_image_size.split('x')
    model_image_size = (int(height), int(width))
    assert (model_image_size[0]%32 == 0 and model_image_size[1]%32 == 0), 'model_image_size should be multiples of 32'

    coreml_convert(args.input_model_file, args.output_file, model_image_size)


if __name__ == '__main__':
    main()

