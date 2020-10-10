#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""YOLO_v3 VGG16 Model Defined in Keras."""

from tensorflow.keras.layers import Conv2D, UpSampling2D, Concatenate, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16

from common.backbones.layers import YoloConv2D
from yolo3.models.layers import compose, DarknetConv2D, DarknetConv2D_BN_Leaky, make_last_layers


def yolo3_vgg16_body(inputs, num_anchors, num_classes):
    """Create YOLO_V3 model CNN body in Keras."""
    '''
    Layer Name input_1 Output: Tensor("input_1:0", shape=(?, 416, 416, 3), dtype=float32)
    Layer Name block1_conv1 Output: Tensor("block1_conv1/Relu:0", shape=(?, 416, 416, 64), dtype=float32)
    Layer Name block1_conv2 Output: Tensor("block1_conv2/Relu:0", shape=(?, 416, 416, 64), dtype=float32)
    Layer Name block1_pool Output: Tensor("block1_pool/MaxPool:0", shape=(?, 208, 208, 64), dtype=float32)
    Layer Name block2_conv1 Output: Tensor("block2_conv1/Relu:0", shape=(?, 208, 208, 128), dtype=float32)
    Layer Name block2_conv2 Output: Tensor("block2_conv2/Relu:0", shape=(?, 208, 208, 128), dtype=float32)
    Layer Name block2_pool Output: Tensor("block2_pool/MaxPool:0", shape=(?, 104, 104, 128), dtype=float32)
    Layer Name block3_conv1 Output: Tensor("block3_conv1/Relu:0", shape=(?, 104, 104, 256), dtype=float32)
    Layer Name block3_conv2 Output: Tensor("block3_conv2/Relu:0", shape=(?, 104, 104, 256), dtype=float32)
    Layer Name block3_conv3 Output: Tensor("block3_conv3/Relu:0", shape=(?, 104, 104, 256), dtype=float32)
    Layer Name block3_pool Output: Tensor("block3_pool/MaxPool:0", shape=(?, 52, 52, 256), dtype=float32)
    Layer Name block4_conv1 Output: Tensor("block4_conv1/Relu:0", shape=(?, 52, 52, 512), dtype=float32)
    Layer Name block4_conv2 Output: Tensor("block4_conv2/Relu:0", shape=(?, 52, 52, 512), dtype=float32)
    Layer Name block4_conv3 Output: Tensor("block4_conv3/Relu:0", shape=(?, 52, 52, 512), dtype=float32)
    Layer Name block4_pool Output: Tensor("block4_pool/MaxPool:0", shape=(?, 26, 26, 512), dtype=float32)
    Layer Name block5_conv1 Output: Tensor("block5_conv1/Relu:0", shape=(?, 26, 26, 512), dtype=float32)
    Layer Name block5_conv2 Output: Tensor("block5_conv2/Relu:0", shape=(?, 26, 26, 512), dtype=float32)
    Layer Name block5_conv3 Output: Tensor("block5_conv3/Relu:0", shape=(?, 26, 26, 512), dtype=float32)
    Layer Name block5_pool Output: Tensor("block5_pool/MaxPool:0", shape=(?, 13, 13, 512), dtype=float32)
    '''

    #net, endpoint = inception_v2.inception_v2(inputs)
    vgg16 = VGG16(input_tensor=inputs,weights='imagenet',include_top=False)
    x = vgg16.get_layer('block5_pool').output
    x = YoloConv2D(512, (3, 3), activation='relu', padding='same', name='block6_conv1')(x)
    x = YoloConv2D(512, (3, 3), activation='relu', padding='same', name='block6_conv2')(x)
    x = YoloConv2D(512, (3, 3), activation='relu', padding='same', name='block6_conv3')(x)
    x = YoloConv2D(512, (3, 3), activation='relu', padding='same', name='block6_conv4')(x)

    # input: 416 x 416 x 3
    # block6_conv3 :13 x 13 x 512
    # block5_conv3 :26 x 26 x 512
    # block4_conv3 : 52 x 52 x 512


    # f1 :13 x 13 x 1024 13 x 13 x 512
    x, y1 = make_last_layers(x, 512, num_anchors * (num_classes + 5), predict_id='1')

    x = compose(
            DarknetConv2D_BN_Leaky(256, (1,1)),
            UpSampling2D(2))(x)

    f2 = vgg16.get_layer('block5_conv3').output
    # f2: 26 x 26 x 512
    x = Concatenate()([x,f2])

    x, y2 = make_last_layers(x, 256, num_anchors*(num_classes+5), predict_id='2')

    x = compose(
            DarknetConv2D_BN_Leaky(128, (1,1)),
            UpSampling2D(2))(x)

    f3 = vgg16.get_layer('block4_conv3').output
    # f3 : 52 x 52 x 256
    x = Concatenate()([x, f3])
    x, y3 = make_last_layers(x, 128, num_anchors*(num_classes+5), predict_id='3')

    return Model(inputs = inputs, outputs=[y1,y2,y3])

def tiny_yolo3_vgg16_body(inputs, num_anchors, num_classes):
    '''Create Tiny YOLO_v3 VGG16 model CNN body in keras.'''
    vgg16 = VGG16(input_tensor=inputs,weights='imagenet',include_top=False)
    x = vgg16.get_layer('block5_pool').output
    x = YoloConv2D(512, (3, 3), activation='relu', padding='same', name='block6_conv1')(x)
    x = YoloConv2D(512, (3, 3), activation='relu', padding='same', name='block6_conv2')(x)
    x = YoloConv2D(512, (3, 3), activation='relu', padding='same', name='block6_conv3')(x)
    #x = YoloConv2D(512, (3, 3), activation='relu', padding='same', name='block6_conv4')(x)

    # input: 416 x 416 x 3
    # block6_conv3 :13 x 13 x 512
    # block5_conv3 :26 x 26 x 512
    # block4_conv3 : 52 x 52 x 512

    x1 = vgg16.get_layer('block5_conv3').output
    x2 = x

    x2 = DarknetConv2D_BN_Leaky(512, (1,1))(x2)

    y1 = compose(
            DarknetConv2D_BN_Leaky(1024, (3,3)),
            #Depthwise_Separable_Conv2D_BN_Leaky(filters=1024, kernel_size=(3, 3), block_id_str='14'),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1)))(x2)

    x2 = compose(
            DarknetConv2D_BN_Leaky(256, (1,1)),
            UpSampling2D(2))(x2)
    y2 = compose(
            Concatenate(),
            DarknetConv2D_BN_Leaky(512, (3,3)),
            #Depthwise_Separable_Conv2D_BN_Leaky(filters=512, kernel_size=(3, 3), block_id_str='15'),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1)))([x2,x1])

    return Model(inputs, [y1,y2])

