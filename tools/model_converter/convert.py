#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reads Darknet config and weights and creates Keras model with TF backend.

Support YOLOv2/v3/v4 families and Yolo-Fastest from following links:

https://pjreddie.com/darknet/yolo/
https://github.com/AlexeyAB/darknet
https://github.com/dog-qiuqiu/Yolo-Fastest

Refer README.md for usage details.

"""
import argparse
import configparser
import io
import os, sys
from collections import defaultdict

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (Conv2D, DepthwiseConv2D, Input, ZeroPadding2D, Add, Multiply, Lambda, Dropout,Reshape,
                          UpSampling2D, MaxPooling2D, AveragePooling2D, Dense, Flatten, Concatenate, Activation, GlobalAveragePooling2D)
from tensorflow.keras.layers import LeakyReLU, ReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import plot_model as plot

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))
from yolo4.models.layers import mish

parser = argparse.ArgumentParser(description='Darknet To Keras Converter.')
parser.add_argument('config_path', help='Path to Darknet cfg file.')
parser.add_argument('weights_path', help='Path to Darknet weights file.')
parser.add_argument('output_path', help='Path to output Keras model file.')
parser.add_argument(
    '-p',
    '--plot_model',
    help='Plot generated Keras model and save as image.',
    action='store_true')
parser.add_argument(
    '-w',
    '--weights_only',
    help='Save as Keras weights file instead of model file.',
    action='store_true')
parser.add_argument(
    '-f',
    '--fixed_input_shape',
    help='Use fixed input shape specified in cfg.',
    action='store_true')
parser.add_argument(
    '-r',
    '--yolo4_reorder',
    help='Reorder output tensors for YOLOv4 cfg and weights file.',
    action='store_true')


def unique_config_sections(config_file):
    """Convert all config sections to have unique names.

    Adds unique suffixes to config sections for compability with configparser.
    """
    section_counters = defaultdict(int)
    output_stream = io.StringIO()
    with open(config_file) as fin:
        for line in fin:
            if line.startswith('['):
                section = line.strip().strip('[]')
                _section = section + '_' + str(section_counters[section])
                section_counters[section] += 1
                line = line.replace(section, _section)
            output_stream.write(line)
    output_stream.seek(0)
    return output_stream


def main(args):
    config_path = os.path.expanduser(args.config_path)
    weights_path = os.path.expanduser(args.weights_path)
    assert config_path.endswith('.cfg'), '{} is not a .cfg file'.format(
        config_path)
    assert weights_path.endswith(
        '.weights'), '{} is not a .weights file'.format(weights_path)

    output_path = os.path.expanduser(args.output_path)
    #assert output_path.endswith(
        #'.h5'), 'output path {} is not a .h5 file'.format(output_path)
    output_root = os.path.splitext(output_path)[0]

    # Load weights and config.
    print('Loading weights.')
    weights_file = open(weights_path, 'rb')
    major, minor, revision = np.ndarray(
        shape=(3, ), dtype='int32', buffer=weights_file.read(12))
    if (major*10+minor)>=2 and major<1000 and minor<1000:
        seen = np.ndarray(shape=(1,), dtype='int64', buffer=weights_file.read(8))
    else:
        seen = np.ndarray(shape=(1,), dtype='int32', buffer=weights_file.read(4))
    print('Weights Header: ', major, minor, revision, seen)

    print('Parsing Darknet config.')
    unique_config_file = unique_config_sections(config_path)
    cfg_parser = configparser.ConfigParser()
    cfg_parser.read_file(unique_config_file)

    weight_decay = float(cfg_parser['net_0']['decay']) if 'net_0' in cfg_parser.sections() else 5e-4

    # Parase model input width, height
    width = int(cfg_parser['net_0']['width']) if 'net_0' in cfg_parser.sections() else None
    height = int(cfg_parser['net_0']['height']) if 'net_0' in cfg_parser.sections() else None

    print('Creating Keras model.')
    if width and height and args.fixed_input_shape:
        input_layer = Input(shape=(height, width, 3), name='image_input')
    else:
        input_layer = Input(shape=(None, None, 3), name='image_input')
    prev_layer = input_layer
    all_layers = []

    count = 0
    fc_flag = False
    out_index = []
    anchors = None
    for section in cfg_parser.sections():
        print('Parsing section {}'.format(section))
        if section.startswith('convolutional'):
            filters = int(cfg_parser[section]['filters'])
            size = int(cfg_parser[section]['size'])
            stride = int(cfg_parser[section]['stride'])
            pad = int(cfg_parser[section]['pad'])
            activation = cfg_parser[section]['activation'] if 'activation' in cfg_parser[section] else 'linear'
            batch_normalize = 'batch_normalize' in cfg_parser[section]

            padding = 'same' if pad == 1 and stride == 1 else 'valid'

            # support DepthwiseConv2D with "groups"
            # option in conv section
            if 'groups' in cfg_parser[section]:
                groups = int(cfg_parser[section]['groups'])
                # Now only support DepthwiseConv2D with "depth_multiplier=1",
                # which means conv groups should be same as filters
                assert groups == filters, 'Only support groups is same as filters.'
                depthwise = True
                depth_multiplier = 1
            else:
                depthwise = False

            # Setting weights.
            # Darknet serializes convolutional weights as:
            # [bias/beta, [gamma, mean, variance], conv_weights]
            prev_layer_shape = K.int_shape(prev_layer)

            if depthwise:
                # DepthwiseConv2D weights shape in TF:
                # (kernel_size, kernel_size, in_channels, depth_multiplier).
                weights_shape = (size, size, prev_layer_shape[-1], depth_multiplier)
                darknet_w_shape = (depth_multiplier, weights_shape[2], size, size)
                weights_size = np.product(weights_shape)
                print('depthwiseconv2d', 'bn'
                      if batch_normalize else '  ', activation, weights_shape)
            else:
                weights_shape = (size, size, prev_layer_shape[-1], filters)
                darknet_w_shape = (filters, weights_shape[2], size, size)
                weights_size = np.product(weights_shape)
                print('conv2d', 'bn'
                      if batch_normalize else '  ', activation, weights_shape)

            conv_bias = np.ndarray(
                shape=(filters, ),
                dtype='float32',
                buffer=weights_file.read(filters * 4))
            count += filters

            if batch_normalize:
                bn_weights = np.ndarray(
                    shape=(3, filters),
                    dtype='float32',
                    buffer=weights_file.read(filters * 12))
                count += 3 * filters

                bn_weight_list = [
                    bn_weights[0],  # scale gamma
                    conv_bias,  # shift beta
                    bn_weights[1],  # running mean
                    bn_weights[2]  # running var
                ]

            conv_weights = np.ndarray(
                shape=darknet_w_shape,
                dtype='float32',
                buffer=weights_file.read(weights_size * 4))
            count += weights_size

            # DarkNet conv_weights are serialized Caffe-style:
            # (out_dim, in_dim, height, width)
            # We would like to set these to Tensorflow order:
            # (height, width, in_dim, out_dim)
            conv_weights = np.transpose(conv_weights, [2, 3, 1, 0])
            conv_weights = [conv_weights] if batch_normalize else [
                conv_weights, conv_bias
            ]

            # Handle activation.
            act_fn = None
            if activation == 'leaky':
                pass  # Add advanced activation later.
            elif activation == 'relu':
                pass  # Add advanced activation later.
            elif activation == 'mish':
                pass  # Add advanced activation later.
            elif activation == 'logistic':
                pass  # Add advanced activation later.
            elif activation != 'linear':
                raise ValueError(
                    'Unknown activation function `{}` in section {}'.format(
                        activation, section))

            # Create Conv2D layer
            if stride>1:
                # Darknet uses left and top padding instead of 'same' mode
                prev_layer = ZeroPadding2D(((1,0),(1,0)))(prev_layer)

            if depthwise:
                conv_layer = (DepthwiseConv2D(
                    (size, size),
                    strides=(stride, stride),
                    depth_multiplier=depth_multiplier,
                    kernel_regularizer=l2(weight_decay),
                    use_bias=not batch_normalize,
                    weights=conv_weights,
                    activation=act_fn,
                    padding=padding))(prev_layer)
            else:
                conv_layer = (Conv2D(
                    filters, (size, size),
                    strides=(stride, stride),
                    kernel_regularizer=l2(weight_decay),
                    use_bias=not batch_normalize,
                    weights=conv_weights,
                    activation=act_fn,
                    padding=padding))(prev_layer)

            if batch_normalize:
                conv_layer = (BatchNormalization(
                    weights=bn_weight_list))(conv_layer)
            prev_layer = conv_layer

            if activation == 'linear':
                all_layers.append(prev_layer)
            elif activation == 'mish':
                act_layer = Activation(mish)(prev_layer)
                prev_layer = act_layer
                all_layers.append(act_layer)
            elif activation == 'leaky':
                act_layer = LeakyReLU(alpha=0.1)(prev_layer)
                prev_layer = act_layer
                all_layers.append(act_layer)
            elif activation == 'relu':
                act_layer = ReLU()(prev_layer)
                prev_layer = act_layer
                all_layers.append(act_layer)
            elif activation == 'logistic':
                act_layer = Activation('sigmoid')(prev_layer)
                prev_layer = act_layer
                all_layers.append(act_layer)

        elif section.startswith('connected'):
            assert args.fixed_input_shape, 'Model with full-connected layer need to specify fixed input shape'
            output_size = int(cfg_parser[section]['output'])
            activation = cfg_parser[section]['activation']

            prev_layer_shape = K.int_shape(prev_layer)

            # TODO: This assumes channel last dim_ordering.
            weights_shape = (np.prod(prev_layer_shape[1:]), output_size)
            darknet_w_shape = (output_size, weights_shape[0])
            weights_size = np.product(weights_shape)

            print('full-connected', activation, weights_shape)

            fc_bias = np.ndarray(
                shape=(output_size,),
                dtype='float32',
                buffer=weights_file.read(output_size * 4))
            count += output_size

            fc_weights = np.ndarray(
                shape=darknet_w_shape,
                dtype='float32',
                buffer=weights_file.read(weights_size * 4))
            count += weights_size

            # DarkNet fc_weights are serialized Caffe-style:
            # (out_dim, in_dim)
            # We would like to set these to Tensorflow order:
            # (in_dim, out_dim)
            # TODO: Add check for Theano dim ordering.
            fc_weights = np.transpose(fc_weights, [1, 0])
            fc_weights = [fc_weights, fc_bias]

            # Handle activation.
            act_fn = None
            if activation == 'leaky':
                pass  # Add advanced activation later.
            elif activation == 'relu':
                pass  # Add advanced activation later.
            elif activation == 'mish':
                pass  # Add advanced activation later.
            elif activation == 'logistic':
                pass  # Add advanced activation later.
            elif activation != 'linear':
                raise ValueError(
                    'Unknown activation function `{}` in section {}'.format(
                        activation, section))

            if not fc_flag:
                prev_layer = Flatten()(prev_layer)
                fc_flag = True

            # Create Full-Connect layer
            fc_layer = Dense(
                output_size,
                kernel_regularizer=l2(weight_decay),
                weights=fc_weights,
                activation=act_fn,
                name=format(section))(prev_layer)

            prev_layer = fc_layer

            if activation == 'linear':
                all_layers.append(prev_layer)
            elif activation == 'mish':
                act_layer = Activation(mish)(prev_layer)
                prev_layer = act_layer
                all_layers.append(act_layer)
            elif activation == 'leaky':
                act_layer = LeakyReLU(alpha=0.1)(prev_layer)
                prev_layer = act_layer
                all_layers.append(act_layer)
            elif activation == 'relu':
                act_layer = ReLU()(prev_layer)
                prev_layer = act_layer
                all_layers.append(act_layer)
            elif activation == 'logistic':
                act_layer = Activation('sigmoid')(prev_layer)
                prev_layer = act_layer
                all_layers.append(act_layer)

        elif section.startswith('route'):
            ids = [int(i) for i in cfg_parser[section]['layers'].split(',')]
            layers = [all_layers[i] for i in ids]

            if ('groups' in cfg_parser[section]):
                # support route with groups, which is for splitting input tensor into group
                # Reference comment (from AlexeyAB):
                #
                # https://github.com/lutzroeder/netron/issues/531
                #
                assert 'group_id' in cfg_parser[section], 'route with groups should have group_id.'
                assert len(layers) == 1, 'route with groups should have 1 input layer.'

                groups = int(cfg_parser[section]['groups'])
                group_id = int(cfg_parser[section]['group_id'])
                route_layer = layers[0]  # group route only have 1 input layer
                print('Split {} to {} groups and pick id {}'.format(route_layer, groups, group_id))

                all_layers.append(
                    Lambda(
                        # tf.split implementation for groups route
                        lambda x: tf.split(x, num_or_size_splits=groups, axis=-1)[group_id],
                        name='group_route_'+str(len(all_layers)))(route_layer))
                prev_layer = all_layers[-1]
            else:
                if len(layers) > 1:
                    print('Concatenating route layers:', layers)
                    concatenate_layer = Concatenate()(layers)
                    all_layers.append(concatenate_layer)
                    prev_layer = concatenate_layer
                else:
                    skip_layer = layers[0]  # only one layer to route
                    all_layers.append(skip_layer)
                    prev_layer = skip_layer

        elif section.startswith('maxpool'):
            size = int(cfg_parser[section]['size'])
            stride = int(cfg_parser[section]['stride'])
            all_layers.append(
                MaxPooling2D(
                    pool_size=(size, size),
                    strides=(stride, stride),
                    padding='same')(prev_layer))
            prev_layer = all_layers[-1]

        elif section.startswith('avgpool'):
            all_layers.append(
                AveragePooling2D()(prev_layer))
            prev_layer = all_layers[-1]

        # support GAP and Reshape layer for se block
        # Reference:
        #
        # https://github.com/gitE0Z9/keras-YOLOv3-model-set/commit/67ed826f2ce00f28296fcf88c03c86f1d86b8b9e
        #
        elif section.startswith('gap'):
            all_layers.append(
                GlobalAveragePooling2D()(prev_layer))
            prev_layer = all_layers[-1]

        elif section.startswith('reshape'):
            all_layers.append(
                Reshape((1, 1, K.int_shape(prev_layer)[-1]))(prev_layer))
            prev_layer = all_layers[-1]

        elif section.startswith('shortcut'):
            index = int(cfg_parser[section]['from'])
            activation = cfg_parser[section]['activation']
            assert activation == 'linear', 'Only linear activation supported.'
            all_layers.append(Add()([all_layers[index], prev_layer]))
            prev_layer = all_layers[-1]

        elif section.startswith('sam'):
            # support SAM (Modified Spatial Attention Module in YOLOv4) layer
            # Reference comment:
            #
            # https://github.com/AlexeyAB/darknet/issues/3708
            #
            index = int(cfg_parser[section]['from'])
            all_layers.append(Multiply()([all_layers[index], prev_layer]))
            prev_layer = all_layers[-1]

        elif section.startswith('dropout'):
            rate = float(cfg_parser[section]['probability'])
            assert rate >= 0 and rate <= 1, 'Dropout rate should be between 0 and 1, got {}.'.format(rate)
            all_layers.append(Dropout(rate=rate)(prev_layer))
            prev_layer = all_layers[-1]

        elif section.startswith('upsample'):
            stride = int(cfg_parser[section]['stride'])
            assert stride%2 == 0, 'upsample stride should be multiples of 2'
            all_layers.append(UpSampling2D(stride)(prev_layer))
            prev_layer = all_layers[-1]

        elif section.startswith('detection'):
            classes = int(cfg_parser[section]['classes'])
            coords = int(cfg_parser[section]['coords'])
            rescore = int(cfg_parser[section]['rescore'])
            side = int(cfg_parser[section]['side'])
            num = int(cfg_parser[section]['num'])
            reshape_layer = Reshape(
                (side, side, classes + num * (coords + rescore))
            )(prev_layer)
            prev_layer = reshape_layer
            all_layers.append(prev_layer)

        elif section.startswith('reorg'):
            block_size = int(cfg_parser[section]['stride'])
            assert block_size == 2, 'Only reorg with stride 2 supported.'
            all_layers.append(
                Lambda(
                    #space_to_depth_x2,
                    #output_shape=space_to_depth_x2_output_shape,
                    lambda x: tf.nn.space_to_depth(x, block_size=2),
                    name='space_to_depth_x2')(prev_layer))
            prev_layer = all_layers[-1]

        elif section.startswith('region'):
            # YOLOv2 anchors parse, here we convert origin
            # grid-reference value to pixel size value
            anchors_line = cfg_parser[section]['anchors']
            anchors_list = list(map(float, anchors_line.split(',')))
            anchors_line = [str(anchor * 32) for anchor in anchors_list]
            anchors = ', '.join(anchors_line)

        elif section.startswith('yolo'):
            out_index.append(len(all_layers)-1)
            all_layers.append(None)
            prev_layer = all_layers[-1]
            # YOLOv3/v4 anchors parse
            anchors = cfg_parser[section]['anchors']

        elif (section.startswith('net') or section.startswith('cost') or
              section.startswith('softmax')):
            pass  # Configs not currently handled during models definition.

        else:
            raise ValueError(
                'Unsupported section header type: {}'.format(section))

    # Create and save model.
    if len(out_index)==0:
        out_index.append(len(all_layers)-1)

    if args.yolo4_reorder:
        # reverse the output tensor index for YOLOv4 cfg & weights,
        # since it use a different yolo outout order
        out_index.reverse()

    model = Model(inputs=input_layer, outputs=[all_layers[i] for i in out_index])
    model.summary()

    if args.weights_only:
        model.save_weights('{}'.format(output_path))
        print('Saved Keras weights to {}'.format(output_path))
    else:
        model.save('{}'.format(output_path))
        print('Saved Keras model to {}'.format(output_path))

    # Check to see if all weights have been read.
    remaining_weights = len(weights_file.read()) / 4
    weights_file.close()
    print('Read {} of {} from Darknet weights.'.format(count, count +
                                                       remaining_weights))
    if remaining_weights > 0:
        print('Warning: {} unused weights'.format(remaining_weights))

    if anchors:
        with open('{}_anchors.txt'.format(output_root), 'w') as f:
            print(anchors, file=f)
        print('Saved anchors to {}_anchors.txt'.format(output_root))

    if args.plot_model:
        plot(model, to_file='{}.png'.format(output_root), show_shapes=True)
        print('Saved model plot to {}.png'.format(output_root))


if __name__ == '__main__':
    main(parser.parse_args())
