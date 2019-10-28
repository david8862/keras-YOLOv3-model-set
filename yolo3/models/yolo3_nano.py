"""YOLO_v3 Model Defined in Keras."""

from tensorflow.keras.layers import UpSampling2D, Concatenate, Dense, Multiply, Add, Lambda
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, Concatenate, BatchNormalization, ReLU, ZeroPadding2D, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

from yolo3.models.layers import compose, DarknetConv2D


def correct_pad(backend, inputs, kernel_size):
    """Returns a tuple for zero-padding for 2D convolution with downsampling.

    # Arguments
        input_size: An integer or tuple/list of 2 integers.
        kernel_size: An integer or tuple/list of 2 integers.

    # Returns
        A tuple.
    """
    img_dim = 2 if backend.image_data_format() == 'channels_first' else 1
    input_size = backend.int_shape(inputs)[img_dim:(img_dim + 2)]

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)

    if input_size[0] is None:
        adjust = (1, 1)
    else:
        adjust = (1 - input_size[0] % 2, 1 - input_size[1] % 2)

    correct = (kernel_size[0] // 2, kernel_size[1] // 2)

    return ((correct[0] - adjust[0], correct[0]),
            (correct[1] - adjust[1], correct[1]))


def NanoConv2D_BN_Relu6(*args, **kwargs):
    """Darknet Convolution2D followed by BatchNormalization and ReLU6."""
    nano_name = kwargs.get('name')
    if nano_name:
        name_kwargs = {'name': nano_name + '_conv2d'}
        name_kwargs.update(kwargs)
        bn_name = nano_name + '_BN'
        relu_name = nano_name + '_relu'
    else:
        name_kwargs = {}
        name_kwargs.update(kwargs)
        bn_name = None
        relu_name = None

    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(name_kwargs)
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(name=bn_name),
        ReLU(6., name=relu_name))


def _ep_block(inputs, filters, stride, expansion, block_id):
    #in_channels = backend.int_shape(inputs)[-1]
    in_channels = inputs.shape.as_list()[-1]

    pointwise_conv_filters = int(filters)
    x = inputs
    prefix = 'ep_block_{}_'.format(block_id)

    # Expand
    x = Conv2D(int(expansion * in_channels), kernel_size=1, padding='same', use_bias=False, activation=None, name=prefix + 'expand')(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999, name=prefix + 'expand_BN')(x)
    x = ReLU(6., name=prefix + 'expand_relu')(x)

    # Depthwise
    if stride == 2:
        x = ZeroPadding2D(padding=correct_pad(K, x, 3), name=prefix + 'pad')(x)

    x = DepthwiseConv2D(kernel_size=3, strides=stride, activation=None, use_bias=False, padding='same' if stride == 1 else 'valid', name=prefix + 'depthwise')(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999, name=prefix + 'depthwise_BN')(x)
    x = ReLU(6., name=prefix + 'depthwise_relu')(x)

    # Project
    x = Conv2D(pointwise_conv_filters, kernel_size=1, padding='same', use_bias=False, activation=None, name=prefix + 'project')(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999, name=prefix + 'project_BN')(x)

    if in_channels == pointwise_conv_filters and stride == 1:
        return Add(name=prefix + 'add')([inputs, x])
    return x


def _pep_block(inputs, proj_filters, filters, stride, expansion, block_id):
    #in_channels = backend.int_shape(inputs)[-1]
    in_channels = inputs.shape.as_list()[-1]

    pointwise_conv_filters = int(filters)
    x = inputs
    prefix = 'pep_block_{}_'.format(block_id)


    # Pre-project
    x = Conv2D(proj_filters, kernel_size=1, padding='same', use_bias=False, activation=None, name=prefix + 'preproject')(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999, name=prefix + 'preproject_BN')(x)
    x = ReLU(6., name=prefix + 'preproject_relu')(x)

    # Expand
    #x = Conv2D(int(expansion * in_channels), kernel_size=1, padding='same', use_bias=False, activation=None, name=prefix + 'expand')(x)
    x = Conv2D(int(expansion * proj_filters), kernel_size=1, padding='same', use_bias=False, activation=None, name=prefix + 'expand')(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999, name=prefix + 'expand_BN')(x)
    x = ReLU(6., name=prefix + 'expand_relu')(x)

    # Depthwise
    if stride == 2:
        x = ZeroPadding2D(padding=correct_pad(K, x, 3), name=prefix + 'pad')(x)

    x = DepthwiseConv2D(kernel_size=3, strides=stride, activation=None, use_bias=False, padding='same' if stride == 1 else 'valid', name=prefix + 'depthwise')(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999, name=prefix + 'depthwise_BN')(x)
    x = ReLU(6., name=prefix + 'depthwise_relu')(x)

    # Project
    x = Conv2D(pointwise_conv_filters, kernel_size=1, padding='same', use_bias=False, activation=None, name=prefix + 'project')(x)
    x = BatchNormalization( epsilon=1e-3, momentum=0.999, name=prefix + 'project_BN')(x)

    if in_channels == pointwise_conv_filters and stride == 1:
        return Add(name=prefix + 'add')([inputs, x])
    return x


def expand_dims2d(x):
    x = K.expand_dims(x, axis=1)
    x = K.expand_dims(x, axis=2)
    return x

def _fca_block(inputs, reduct_ratio, block_id):
    in_channels = inputs.shape.as_list()[-1]
    in_shapes = inputs.shape.as_list()[1:3]
    reduct_channels = int(in_channels // reduct_ratio)
    prefix = 'fca_block_{}_'.format(block_id)

    x = GlobalAveragePooling2D(name=prefix + 'average_pooling')(inputs)
    x = Dense(reduct_channels, activation='relu', name=prefix + 'fc1')(x)
    x = Dense(in_channels, activation='sigmoid', name=prefix + 'fc2')(x)
    x = Lambda(expand_dims2d, name=prefix + 'expand_dims2d')(x)
    x = UpSampling2D(in_shapes, name=prefix + 'upsample')(x)
    x = Multiply(name=prefix + 'multiply')([x, inputs])
    return x


def nano_net_body(x):
    '''YOLO Nano backbone network body'''
    x = NanoConv2D_BN_Relu6(12, (3,3), name='Conv_1')(x)
    x = NanoConv2D_BN_Relu6(24, (3,3), strides=2, name='Conv_2')(x)
    x = _pep_block(x, proj_filters=7, filters=24, stride=1, expansion=2, block_id=1)
    x = _ep_block(x, filters=70, stride=2, expansion=2, block_id=1)
    x = _pep_block(x, proj_filters=25, filters=70, stride=1, expansion=2, block_id=2)
    x = _pep_block(x, proj_filters=24, filters=70, stride=1, expansion=2, block_id=3)
    x = _ep_block(x, filters=150, stride=2, expansion=2, block_id=2)
    x = _pep_block(x, proj_filters=56, filters=150, stride=1, expansion=2, block_id=4)
    x = NanoConv2D_BN_Relu6(150, (1,1), name='Conv_pw_1')(x)
    x = _fca_block(x, reduct_ratio=8, block_id=1)
    x = _pep_block(x, proj_filters=73, filters=150, stride=1, expansion=2, block_id=5)
    x = _pep_block(x, proj_filters=71, filters=150, stride=1, expansion=2, block_id=6)
    x = _pep_block(x, proj_filters=75, filters=150, stride=1, expansion=2, block_id=7)
    x = _ep_block(x, filters=325, stride=2, expansion=2, block_id=3)
    x = _pep_block(x, proj_filters=132, filters=325, stride=1, expansion=2, block_id=8)
    x = _pep_block(x, proj_filters=124, filters=325, stride=1, expansion=2, block_id=9)
    x = _pep_block(x, proj_filters=141, filters=325, stride=1, expansion=2, block_id=10)
    x = _pep_block(x, proj_filters=140, filters=325, stride=1, expansion=2, block_id=11)
    x = _pep_block(x, proj_filters=137, filters=325, stride=1, expansion=2, block_id=12)
    x = _pep_block(x, proj_filters=135, filters=325, stride=1, expansion=2, block_id=13)
    x = _pep_block(x, proj_filters=133, filters=325, stride=1, expansion=2, block_id=14)
    x = _pep_block(x, proj_filters=140, filters=325, stride=1, expansion=2, block_id=15)
    x = _ep_block(x, filters=545, stride=2, expansion=2, block_id=4)
    x = _pep_block(x, proj_filters=276, filters=545, stride=1, expansion=2, block_id=16)
    x = NanoConv2D_BN_Relu6(230, (1,1), name='Conv_pw_2')(x)
    x = _ep_block(x, filters=489, stride=1, expansion=2, block_id=5)
    x = _pep_block(x, proj_filters=213, filters=469, stride=1, expansion=2, block_id=17)
    x = NanoConv2D_BN_Relu6(189, (1,1), name='Conv_pw_3')(x)

    return x


def yolo_nano_body(inputs, num_anchors, num_classes, weights_path=None):
    """
    Create YOLO_V3 Nano model CNN body in Keras.

    Reference Paper:
        "YOLO Nano: a Highly Compact You Only Look Once Convolutional Neural Network for Object Detection"
        https://arxiv.org/abs/1910.01271
    """
    nano_net = Model(inputs, nano_net_body(inputs))
    if weights_path is not None:
        nano_net.load_weights(weights_path, by_name=True)
        print('Load weights {}.'.format(weights_path))

    # input: 416 x 416 x 3
    # Conv_pw_3_relu: 13 x 13 x 189
    # pep_block_15_add: 26 x 26 x 325
    # pep_block_7_add: 52 x 52 x 150

    f1 = nano_net.get_layer('Conv_pw_3').output
    # f1 :13 x 13 x 189
    y1 = _ep_block(f1, filters=462, stride=1, expansion=2, block_id=6)
    y1 = DarknetConv2D(num_anchors * (num_classes + 5), (1,1))(y1)
    x = compose(
            NanoConv2D_BN_Relu6(105, (1,1)),
            UpSampling2D(2))(f1)


    f2 = nano_net.get_layer('pep_block_15_add').output
    # f2: 26 x 26 x 325
    x = Concatenate()([x,f2])

    x = _pep_block(x, proj_filters=113, filters=325, stride=1, expansion=2, block_id=18)
    x = _pep_block(x, proj_filters=99, filters=207, stride=1, expansion=2, block_id=19)
    x = DarknetConv2D(98, (1,1))(x)

    y2 = _ep_block(x, filters=183, stride=1, expansion=2, block_id=7)
    y2 = DarknetConv2D(num_anchors * (num_classes + 5), (1,1))(y2)

    x = compose(
            NanoConv2D_BN_Relu6(47, (1,1)),
            UpSampling2D(2))(x)


    f3 = nano_net.get_layer('pep_block_7_add').output
    # f3 : 52 x 52 x 150
    x = Concatenate()([x, f3])

    x = _pep_block(x, proj_filters=58, filters=122, stride=1, expansion=2, block_id=20)
    x = _pep_block(x, proj_filters=52, filters=87, stride=1, expansion=2, block_id=21)
    x = _pep_block(x, proj_filters=47, filters=93, stride=1, expansion=2, block_id=22)
    y3 = DarknetConv2D(num_anchors * (num_classes + 5), (1,1))(x)


    return Model(inputs = inputs, outputs=[y1,y2,y3])


