#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
generate activation pattern image for convolution kernel of specified layers
with trained YOLO model

Reference:
    https://blog.csdn.net/qq_37781464/article/details/122946523
'''
import os, sys, argparse
import numpy as np

from tensorflow.keras.models import Model, load_model
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, SeparableConv2D
import tensorflow.keras.backend as K

# compatible with TF 2.x
if tf.__version__.startswith('2'):
    import tensorflow.compat.v1 as tf
    from tensorflow.compat.v1.keras import backend as K
    tf.disable_eager_execution()

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))
from common.utils import get_custom_objects, optimize_tf_gpu

optimize_tf_gpu(tf, K)


# convert tensor to visible image
def deprocess_image(x):
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1
    x += 0.5
    x = np.clip(x, 0, 1)
    # x *= 255
    # x = np.clip(x, 0, 255)
    # x/=255.
    return x


def get_layer_type(layer):
    if isinstance(layer, Conv2D):
        return 'Conv2D'
    elif isinstance(layer, DepthwiseConv2D):
        return 'DepthwiseConv2D'
    elif isinstance(layer, SeparableConv2D):
        return 'SeparableConv2D'
    else:
        return 'Others'


# generate visualize image of conv kernel by maximize activation
# of conv kernel on random input image, with sgd on a loss function
def generate_pattern(model, layer_name, kernel_index, model_input_shape):

    height, width = model_input_shape
    # get conv kernel output of layer, and
    # use mean value as loss
    layer_output = model.get_layer(layer_name).output
    loss = K.mean(layer_output[:, :, :, kernel_index])
    # get gradients of loss for model input
    grads = K.gradients(loss, model.input)[0]
    # normalize the gradients
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

    # stochastic gradient descent to maximize the loss
    iterate = K.function([model.input], [loss, grads])

    # random input image
    input_img_data = np.random.random((1, height, width, 3)) * 20 + 128.

    # run iteration_num loops to collect pattern
    step = 1.
    iteration_num = 40
    for i in range(iteration_num):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step
    img = input_img_data[0]

    return deprocess_image(img)


def convkernel_check(model_path, model_input_shape, layer_names, kernel_num, output_path):
    # load model
    custom_object_dict = get_custom_objects()
    model = load_model(model_path, compile=False, custom_objects=custom_object_dict)
    K.set_learning_phase(0)
    model.summary()

    height, width = model_input_shape

    os.makedirs(output_path, exist_ok=True)
    for layer_name in layer_names:
        layer = model.get_layer(layer_name)
        if get_layer_type(layer) == 'Others':
            print('layer {} is not convolution type, bypass it'.format(layer_name))
            continue

        layer_channel = layer.output.shape[-1]
        if layer_channel < kernel_num:
            print('layer {} does not have {} conv kernel. only pick {}'.format(layer_name, kernel_num, layer_channel))
            layer_kernel_num = layer_channel
        else:
            layer_kernel_num = kernel_num

        print('Creating pattern for layer {}...'.format(layer_name))

        # prepare featuremap display grid array
        column_num = 16  # 16 kernel pattern column per row
        margin = 5

        row_num = layer_kernel_num // column_num
        display_grid = np.zeros((height*row_num + margin*(row_num-1), width*column_num + margin*(column_num-1), 3))

        # fill one grid with a kernel pattern
        for row in range(row_num):
            for col in range(column_num):
                pattern_img = generate_pattern(model, layer_name, row*column_num+col, model_input_shape)
                horizontal_start = col*width + col*margin
                horizontal_end = horizontal_start + width
                vertical_start = row*height + row*margin
                vertical_end = vertical_start + height
                display_grid[vertical_start:vertical_end, horizontal_start:horizontal_end, :] = pattern_img

        # adjust display size for grid image
        row_scale = 1. / height
        col_scale = 1. / width
        plt.figure(figsize=(col_scale * display_grid.shape[1],
                            row_scale * display_grid.shape[0] + 1.8))

        plt.title('Conv kernel of layer '+layer_name)
        plt.imshow(display_grid)

        # save kernel pattern image & show it
        pattern_file = os.path.join(output_path, layer_name+'.jpg')
        plt.savefig(pattern_file, dpi=75)
        #plt.show()
        print('Kernel pattern of layer {} has been saved'.format(layer_name))



def main():
    parser = argparse.ArgumentParser(description='check kernel pattern of specified conv layer for trained YOLO model')
    parser.add_argument('--model_path', type=str, required=True, help='model file to predict')
    parser.add_argument('--model_input_shape', help='model image input shape as <height>x<width>, default=%(default)s', type=str, default='416x416')
    parser.add_argument('--layer_names', type=str, required=True, help='layer names to check conv kernel, separate with comma if more than one')
    parser.add_argument('--kernel_num', help='conv kernel number to check, default=%(default)s', type=int, default=64)
    parser.add_argument('--output_path', type=str, required=True, help='output featuremap file directory')

    args = parser.parse_args()

    height, width = args.model_input_shape.split('x')
    model_input_shape = (int(height), int(width))
    assert (model_input_shape[0]%32 == 0 and model_input_shape[1]%32 == 0), 'model_input_shape should be multiples of 32'

    layer_names = args.layer_names.split(',')

    convkernel_check(args.model_path, model_input_shape, layer_names, args.kernel_num, args.output_path)


if __name__ == "__main__":
    main()
