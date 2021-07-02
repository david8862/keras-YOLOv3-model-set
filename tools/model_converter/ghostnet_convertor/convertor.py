#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
convert ghostnet pytorch pretrained weights to corresponding tf.keras model weights.
tf.keras model definition should be prepared and the layer/op name should be matched
with pytorch state dict keys.
"""
import os, sys, argparse
import numpy as np
import torch
from ghostnet_pytorch import ghostnet

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', '..'))
from common.backbones.ghostnet import GhostNet


def ghostnet_pytorch_to_keras(output_file):
    # load pretrained pytorch model
    pytorch_model = ghostnet(pretrained=True, weights_path=None)
    pytorch_model_dict = {}

    # parse valid pytorch model weights and store in dict
    for k, v in pytorch_model.state_dict().items():
        if len(v.shape) > 0:
            pytorch_model_dict[k] = v
            #print(k, v.shape)

    # create empty keras model
    model = GhostNet(include_top=True, input_shape=(224, 224, 3), width=1.0, weights=None)

    # walk through keras model layers and get corresponding pytorch weights
    for layer in model.layers:
        lname = layer.name
        weights = np.asarray(model.get_layer(lname).get_weights())
        if len(weights) > 0:
            print('Parsing layer {} with shape {}'.format(lname, weights.shape))

            # adjust layer name to match pytorch weights key
            pytorch_lname = lname.replace('_', '.')
            pytorch_lname = pytorch_lname.replace('conv.stem', 'conv_stem')
            pytorch_lname = pytorch_lname.replace('primary.conv', 'primary_conv')
            pytorch_lname = pytorch_lname.replace('cheap.operation', 'cheap_operation')
            pytorch_lname = pytorch_lname.replace('conv.dw', 'conv_dw')
            pytorch_lname = pytorch_lname.replace('bn.dw', 'bn_dw')
            pytorch_lname = pytorch_lname.replace('conv.reduce', 'conv_reduce')
            pytorch_lname = pytorch_lname.replace('conv.expand', 'conv_expand')
            pytorch_lname = pytorch_lname.replace('conv.head', 'conv_head')

            # convert normal conv without bias
            if pytorch_lname.endswith(('conv', 'conv_stem', 'primary_conv.0', 'shortcut.2')):
                weight = pytorch_model_dict[pytorch_lname + '.weight'].cpu().numpy()
                weight = weight.transpose((2, 3, 1, 0))
                model.get_layer(lname).set_weights([weight])

            # convert depthwise conv without bias
            elif pytorch_lname.endswith(('cheap_operation.0', 'conv_dw', 'shortcut.0')):
                weight = pytorch_model_dict[pytorch_lname + '.weight'].cpu().numpy()
                weight = weight.transpose((2, 3, 0, 1))
                model.get_layer(lname).set_weights([weight])

            # convert normal conv with bias
            elif pytorch_lname.endswith(('conv_reduce', 'conv_expand', 'conv_head')):
                weight = pytorch_model_dict[pytorch_lname + '.weight'].cpu().numpy()
                weight = weight.transpose((2, 3, 1, 0))
                bias = pytorch_model_dict[pytorch_lname + '.bias'].cpu().numpy()
                model.get_layer(lname).set_weights([weight, bias])

            # convert batchnorm
            elif pytorch_lname.endswith(('bn1', 'primary_conv.1', 'cheap_operation.1', 'bn_dw', 'shortcut.1', 'shortcut.3')):
                weight = pytorch_model_dict[pytorch_lname + '.weight'].cpu().numpy()
                bias = pytorch_model_dict[pytorch_lname + '.bias'].cpu().numpy()
                running_mean = pytorch_model_dict[pytorch_lname + '.running_mean'].cpu().numpy()
                running_var = pytorch_model_dict[pytorch_lname + '.running_var'].cpu().numpy()
                model.get_layer(lname).set_weights([weight, bias, running_mean, running_var])

            # convert batchnorm
            elif pytorch_lname.endswith('classifier'):
                weight = pytorch_model_dict[pytorch_lname + '.weight'].cpu().numpy()
                bias = pytorch_model_dict[pytorch_lname + '.bias'].cpu().numpy()
                weight = weight.transpose((1, 0))
                model.get_layer(lname).set_weights([weight, bias])

            # unsupported layer
            else:
                raise ValueError('Unsupported layer name', pytorch_lname)

    model.summary()
    model.save_weights(output_file)
    print('Done')
    return



def main():
    parser = argparse.ArgumentParser(description='convert ghostnet pytorch pretrained weights to tf.keras weights')
    parser.add_argument('--output_file', required=True, type=str, help='output tf.keras weights file')
    args = parser.parse_args()

    ghostnet_pytorch_to_keras(args.output_file)


if __name__ == "__main__":
    main()
