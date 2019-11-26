#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to calculate FLOPs of a tf.keras model.
Only valid for TF 1.x
"""
import argparse
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K


def get_flops(model):
    run_meta = tf.RunMetadata()
    opts = tf.profiler.ProfileOptionBuilder.float_operation()

    # We use the Keras session graph in the call to the profiler.
    flops = tf.profiler.profile(graph=K.get_session().graph, run_meta=run_meta, cmd='op', options=opts)
    print('Total FLOPs: {}m float_ops'.format(flops.total_float_ops/1e6))


def main():
    parser = argparse.ArgumentParser(description='tf.keras model FLOPs checking tool')
    parser.add_argument('--model_path', help='model file to evaluate', type=str, required=True)
    args = parser.parse_args()

    model = load_model(args.model_path, compile=False)
    get_flops(model)


if __name__ == '__main__':
    main()
