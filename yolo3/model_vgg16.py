"""YOLO_v3 Model Defined in Keras."""

from functools import wraps

#import numpy as np
#import tensorflow as tf
#from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D, Add, ZeroPadding2D, UpSampling2D, Concatenate, MaxPooling2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.regularizers import l2

from yolo3.utils import compose


@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    """Wrapper to set Darknet parameters for Convolution2D."""
    darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4)}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides')==(2,2) else 'same'
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)


#def leakyRelu(x, leak=0.2, name="LeakyRelu"):
    #with tf.variable_scope(name):
        #f1 = 0.5 * (1 + leak)
        #f2 = 0.5 * (1 - leak)
        #return f1 * x + f2 * tf.abs(x)

def DarknetConv2D_BN_Leaky(*args, **kwargs):
    """Darknet Convolution2D followed by BatchNormalization and LeakyReLU."""
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1))

def resblock_body(x, num_filters, num_blocks):
    '''A series of resblocks starting with a downsampling Convolution2D'''
    # Darknet uses left and top padding instead of 'same' mode
    x = ZeroPadding2D(((1,0),(1,0)))(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (3,3), strides=(2,2))(x)
    for i in range(num_blocks):
        y = compose(
                DarknetConv2D_BN_Leaky(num_filters//2, (1,1)),
                DarknetConv2D_BN_Leaky(num_filters, (3,3)))(x)
        x = Add()([x,y])
    return x

def darknet_body(x):
    '''Darknent body having 52 Convolution2D layers'''
    # 416 x 416 x 3
    x = DarknetConv2D_BN_Leaky(32, (3,3))(x)

    # 208 x 208 x 32
    x = resblock_body(x, 64, 1)

    # 208 x 208 x 64
    x = resblock_body(x, 128, 2)

    # 104 x 104 x 128
    x = resblock_body(x, 256, 8)

    # 52 x 52 x 256
    x = resblock_body(x, 512, 8)

    # 26 x 26 x 512
    x = resblock_body(x, 1024, 4)

    # 13 x 13 x 1024
    return x

def make_last_layers(x, num_filters, out_filters):
    '''6 Conv2D_BN_Leaky layers followed by a Conv2D_linear layer'''
    x = compose(
            DarknetConv2D_BN_Leaky(num_filters, (1,1)),
            DarknetConv2D_BN_Leaky(num_filters*2, (3,3)),
            DarknetConv2D_BN_Leaky(num_filters, (1,1)),
            DarknetConv2D_BN_Leaky(num_filters*2, (3,3)),
            DarknetConv2D_BN_Leaky(num_filters, (1,1)))(x)
    y = compose(
            DarknetConv2D_BN_Leaky(num_filters*2, (3,3)),
            DarknetConv2D(out_filters, (1,1)))(x)
    return x, y


def yolo_body(inputs, num_anchors, num_classes):
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
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block6_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block6_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block6_conv3')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block6_conv4')(x)

    # input: 416 x 416 x 3
    # block6_conv3 :13 x 13 x 512
    # block5_conv3 :26 x 26 x 512
    # block4_conv3 : 52 x 52 x 512


    # f1 :13 x 13 x 1024 13 x 13 x 512
    x, y1 = make_last_layers(x, 512, num_anchors * (num_classes + 5))

    x = compose(
            DarknetConv2D_BN_Leaky(256, (1,1)),
            UpSampling2D(2))(x)

    f2 = vgg16.get_layer('block5_conv3').output
    # f2: 26 x 26 x 512
    x = Concatenate()([x,f2])

    x, y2 = make_last_layers(x, 256, num_anchors*(num_classes+5))

    x = compose(
            DarknetConv2D_BN_Leaky(128, (1,1)),
            UpSampling2D(2))(x)

    f3 = vgg16.get_layer('block4_conv3').output
    # f3 : 52 x 52 x 256
    x = Concatenate()([x, f3])
    x, y3 = make_last_layers(x, 128, num_anchors*(num_classes+5))

    return Model(inputs = inputs, outputs=[y1,y2,y3])

def tiny_yolo_body(inputs, num_anchors, num_classes):
    '''Create Tiny YOLO_v3 model CNN body in keras.'''
    x1 = compose(
            DarknetConv2D_BN_Leaky(16, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            DarknetConv2D_BN_Leaky(32, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            DarknetConv2D_BN_Leaky(64, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            DarknetConv2D_BN_Leaky(128, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            DarknetConv2D_BN_Leaky(256, (3,3)))(inputs)
    x2 = compose(
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            DarknetConv2D_BN_Leaky(512, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(1,1), padding='same'),
            DarknetConv2D_BN_Leaky(1024, (3,3)),
            DarknetConv2D_BN_Leaky(256, (1,1)))(x1)
    y1 = compose(
            DarknetConv2D_BN_Leaky(512, (3,3)),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1)))(x2)

    x2 = compose(
            DarknetConv2D_BN_Leaky(128, (1,1)),
            UpSampling2D(2))(x2)
    y2 = compose(
            Concatenate(),
            DarknetConv2D_BN_Leaky(256, (3,3)),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1)))([x2,x1])

    return Model(inputs, [y1,y2])

