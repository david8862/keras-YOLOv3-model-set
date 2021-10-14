#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to convert YOLO keras model to ONNX model
"""
import os, sys, argparse
import shutil
import subprocess
import tensorflow as tf
from tensorflow.keras.models import load_model
import onnx

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))
from common.utils import get_custom_objects

os.environ['TF_KERAS'] = '1'

def onnx_convert_old(keras_model_file, output_file, op_set):
    """
    old implementation to convert keras model to onnx,
    using deprecated keras2onnx package
    """
    import keras2onnx
    custom_object_dict = get_custom_objects()
    model = load_model(keras_model_file, custom_objects=custom_object_dict)

    # convert to onnx model
    onnx_model = keras2onnx.convert_keras(model, model.name, custom_op_conversions=custom_object_dict, target_opset=op_set)

    # save converted onnx model
    onnx.save_model(onnx_model, output_file)


def onnx_convert(keras_model_file, output_file, op_set, inputs_as_nchw):
    import tf2onnx
    custom_object_dict = get_custom_objects()
    model = load_model(keras_model_file, custom_objects=custom_object_dict)

    # assume only 1 input tensor for image
    assert len(model.inputs) == 1, 'invalid input tensor number.'

    spec = (tf.TensorSpec(model.inputs[0].shape, tf.float32, name="image_input"),)

    if inputs_as_nchw:
        nchw_inputs_list = [model.inputs[0].name]
    else:
        nchw_inputs_list = None

    # Reference:
    # https://github.com/onnx/tensorflow-onnx#python-api-reference
    model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, custom_ops=None, opset=op_set, inputs_as_nchw=nchw_inputs_list, output_path=output_file)


def onnx_convert_with_savedmodel(keras_model_file, output_file, op_set, inputs_as_nchw):
    # only available for TF 2.x
    if not tf.__version__.startswith('2'):
        raise ValueError('savedmodel convert only support in TF 2.x env')

    custom_object_dict = get_custom_objects()
    model = load_model(keras_model_file, custom_objects=custom_object_dict)

    # assume only 1 input tensor for image
    assert len(model.inputs) == 1, 'invalid input tensor number.'

    # export to saved model
    model.save('tmp_savedmodel', save_format='tf')

    # use tf2onnx to convert to onnx model
    if inputs_as_nchw:
        cmd = 'python -m tf2onnx.convert --saved-model tmp_savedmodel --inputs-as-nchw {} --output {} --opset {}'.format(model.inputs[0].name, output_file, op_set)
    else:
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
    parser.add_argument('--inputs_as_nchw', help="convert input layout to NCHW", default=False, action="store_true")
    parser.add_argument('--with_savedmodel', default=False, action="store_true", help='use a temp savedmodel for convert')

    args = parser.parse_args()

    if args.with_savedmodel:
        onnx_convert_with_savedmodel(args.keras_model_file, args.output_file, args.op_set, args.inputs_as_nchw)
    else:
        onnx_convert(args.keras_model_file, args.output_file, args.op_set, args.inputs_as_nchw)


if __name__ == '__main__':
    main()

