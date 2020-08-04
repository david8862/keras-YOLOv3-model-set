#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to calculate FLOPs & PARAMs of a tf.keras model.
Compatible with TF 1.x and TF 2.x
"""
import os, sys, argparse
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))
from common.utils import get_custom_objects

# check tf version to be compatible with TF 2.x
if tf.__version__.startswith('2'):
    import tensorflow.compat.v1 as tf
    tf.disable_eager_execution()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def get_flops(model):
    run_meta = tf.RunMetadata()
    graph = tf.get_default_graph()

    # We use the Keras session graph in the call to the profiler.
    opts = tf.profiler.ProfileOptionBuilder.float_operation()
    flops = tf.profiler.profile(graph=graph, run_meta=run_meta, cmd='op', options=opts)

    opts = tf.profiler.ProfileOptionBuilder.trainable_variables_parameter()
    params = tf.profiler.profile(graph=graph, run_meta=run_meta, cmd='op', options=opts)

    flops_value = flops.total_float_ops
    param_value = params.total_parameters

    # get friendly FLOPs value string
    if flops_value > 0 and flops_value <= 1e3:
        flops_result_string = '{}'.format(flops_value)
    elif flops_value > 1e3 and flops_value <= 1e6:
        flops_result_string = '%.4fK'%(flops_value/1e3)
    elif flops_value > 1e6 and flops_value <= 1e9:
        flops_result_string = '%.4fM'%(flops_value/1e6)
    elif flops_value > 1e9 and flops_value <= 1e12:
        flops_result_string = '%.4fG'%(flops_value/1e9)
    elif flops_value > 1e12:
        flops_result_string = '%.4fT'%(flops_value/1e12)

    # get friendly PARAMs value string
    if param_value > 0 and param_value <= 1e3:
        param_result_string = '{}'.format(param_value)
    elif param_value > 1e3 and param_value <= 1e6:
        param_result_string = '%.4fK'%(param_value/1e3)
    elif param_value > 1e6 and param_value <= 1e9:
        param_result_string = '%.4fM'%(param_value/1e6)
    elif param_value > 1e9 and param_value <= 1e12:
        param_result_string = '%.4fG'%(param_value/1e9)
    elif param_value > 1e12:
        param_result_string = '%.4fT'%(param_value/1e12)

    print('Total FLOPs: {} float_ops'.format(flops_result_string))
    print('Total PARAMs: {}'.format(param_result_string))


def main():
    parser = argparse.ArgumentParser(description='tf.keras model FLOPs & PARAMs checking tool')
    parser.add_argument('--model_path', type=str, required=True, help='model file to evaluate')
    parser.add_argument('--model_image_size', type=str, required=False, default=None, help='model image input size as <height>x<width>, optional')
    args = parser.parse_args()

    custom_object_dict = get_custom_objects()
    model = load_model(args.model_path, compile=False, custom_objects=custom_object_dict)

    batch, height, width, channel = model.input.shape.as_list()

    if args.model_image_size:
        height, width = args.model_image_size.split('x')
        height, width = int(height), int(width)
        assert (height%32 == 0 and width%32 == 0), 'model_image_size should be multiples of 32'

    # to calculate FLOPs we need to use fixed input shape & batch size
    assert height and width and channel, 'input shape should be specified'

    if not batch:
        # if dynamic batch, rebuild model with batch_size=1
        input_tensor = Input(shape=(height, width, channel), batch_size=1)
        output_tensor = model(input_tensor)
        model = Model(input_tensor, output_tensor)

    K.set_learning_phase(0)
    get_flops(model)


if __name__ == '__main__':
    main()
