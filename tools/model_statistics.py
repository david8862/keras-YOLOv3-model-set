#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to calculate FLOPs & PARAMs of a tf.keras model.
Compatible with TF 1.x and TF 2.x
"""
import os, sys, argparse
import tensorflow as tf
from tensorflow.keras.models import load_model
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
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

    print('Total FLOPs: {}m float_ops'.format(flops.total_float_ops/1e6))
    print('Total PARAMs: {}m'.format(params.total_parameters/1e6))


def main():
    parser = argparse.ArgumentParser(description='tf.keras model FLOPs & PARAMs checking tool')
    parser.add_argument('--model_path', help='model file to evaluate', type=str, required=True)
    args = parser.parse_args()

    custom_object_dict = get_custom_objects()
    model = load_model(args.model_path, compile=False, custom_objects=custom_object_dict)

    get_flops(model)


if __name__ == '__main__':
    main()
