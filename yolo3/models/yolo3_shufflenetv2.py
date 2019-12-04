"""YOLO_v3 Model Defined in Keras."""

from tensorflow.keras.layers import UpSampling2D, Concatenate
from tensorflow.keras.models import Model
from common.backbones.shufflenet_v2 import ShuffleNetV2

from yolo3.models.layers import compose, DarknetConv2D, DarknetConv2D_BN_Leaky, Depthwise_Separable_Conv2D_BN_Leaky, make_last_layers, make_depthwise_separable_last_layers, make_spp_depthwise_separable_last_layers


def yolo3_shufflenetv2_body(inputs, num_anchors, num_classes):
    """Create YOLO_V3 ShuffleNetV2 model CNN body in Keras."""
    shufflenetv2 = ShuffleNetV2(input_tensor=inputs, weights=None, include_top=False)

    # input: 416 x 416 x 3
    # 1x1conv5_out: 13 x 13 x 1024
    # stage4/block1/relu_1x1conv_1: 26 x 26 x 464
    # stage3/block1/relu_1x1conv_1: 52 x 52 x 232

    f1 = shufflenetv2.get_layer('1x1conv5_out').output
    # f1 :13 x 13 x 1024
    x, y1 = make_last_layers(f1, 464, num_anchors * (num_classes + 5))

    x = compose(
            DarknetConv2D_BN_Leaky(232, (1,1)),
            UpSampling2D(2))(x)

    f2 = shufflenetv2.get_layer('stage4/block1/relu_1x1conv_1').output
    # f2: 26 x 26 x 464
    x = Concatenate()([x,f2])

    x, y2 = make_last_layers(x, 232, num_anchors*(num_classes+5))

    x = compose(
            DarknetConv2D_BN_Leaky(116, (1,1)),
            UpSampling2D(2))(x)

    f3 = shufflenetv2.get_layer('stage3/block1/relu_1x1conv_1').output
    # f3 : 52 x 52 x 232
    x = Concatenate()([x, f3])
    x, y3 = make_last_layers(x, 116, num_anchors*(num_classes+5))

    return Model(inputs = inputs, outputs=[y1,y2,y3])


def yolo3lite_shufflenetv2_body(inputs, num_anchors, num_classes):
    '''Create YOLO_v3 Lite ShuffleNetV2 model CNN body in keras.'''
    shufflenetv2 = ShuffleNetV2(input_tensor=inputs, weights=None, include_top=False)

    # input: 416 x 416 x 3
    # 1x1conv5_out: 13 x 13 x 1024
    # stage4/block1/relu_1x1conv_1: 26 x 26 x 464
    # stage3/block1/relu_1x1conv_1: 52 x 52 x 232

    f1 = shufflenetv2.get_layer('1x1conv5_out').output
    # f1 :13 x 13 x 1024
    x, y1 = make_depthwise_separable_last_layers(f1, 464, num_anchors * (num_classes + 5), block_id_str='17')

    x = compose(
            DarknetConv2D_BN_Leaky(232, (1,1)),
            UpSampling2D(2))(x)

    f2 = shufflenetv2.get_layer('stage4/block1/relu_1x1conv_1').output
    # f2: 26 x 26 x 464
    x = Concatenate()([x,f2])

    x, y2 = make_depthwise_separable_last_layers(x, 232, num_anchors * (num_classes + 5), block_id_str='18')

    x = compose(
            DarknetConv2D_BN_Leaky(116, (1,1)),
            UpSampling2D(2))(x)

    f3 = shufflenetv2.get_layer('stage3/block1/relu_1x1conv_1').output
    # f3 : 52 x 52 x 232
    x = Concatenate()([x, f3])
    x, y3 = make_depthwise_separable_last_layers(x, 116, num_anchors * (num_classes + 5), block_id_str='19')

    return Model(inputs = inputs, outputs=[y1,y2,y3])


def yolo3lite_spp_shufflenetv2_body(inputs, num_anchors, num_classes):
    '''Create YOLO_v3 Lite SPP ShuffleNetV2 model CNN body in keras.'''
    shufflenetv2 = ShuffleNetV2(input_tensor=inputs, weights=None, include_top=False)

    # input: 416 x 416 x 3
    # 1x1conv5_out: 13 x 13 x 1024
    # stage4/block1/relu_1x1conv_1: 26 x 26 x 464
    # stage3/block1/relu_1x1conv_1: 52 x 52 x 232

    f1 = shufflenetv2.get_layer('1x1conv5_out').output
    # f1 :13 x 13 x 1024
    #x, y1 = make_depthwise_separable_last_layers(f1, 464, num_anchors * (num_classes + 5), block_id_str='14')
    x, y1 = make_spp_depthwise_separable_last_layers(f1, 464, num_anchors * (num_classes + 5), block_id_str='17')

    x = compose(
            DarknetConv2D_BN_Leaky(232, (1,1)),
            UpSampling2D(2))(x)

    f2 = shufflenetv2.get_layer('stage4/block1/relu_1x1conv_1').output
    # f2: 26 x 26 x 464
    x = Concatenate()([x,f2])

    x, y2 = make_depthwise_separable_last_layers(x, 232, num_anchors * (num_classes + 5), block_id_str='18')

    x = compose(
            DarknetConv2D_BN_Leaky(116, (1,1)),
            UpSampling2D(2))(x)

    f3 = shufflenetv2.get_layer('stage3/block1/relu_1x1conv_1').output
    # f3 : 52 x 52 x 232
    x = Concatenate()([x, f3])
    x, y3 = make_depthwise_separable_last_layers(x, 116, num_anchors * (num_classes + 5), block_id_str='19')

    return Model(inputs = inputs, outputs=[y1,y2,y3])


def tiny_yolo3_shufflenetv2_body(inputs, num_anchors, num_classes):
    '''Create Tiny YOLO_v3 ShuffleNetV2 model CNN body in keras.'''
    shufflenetv2 = ShuffleNetV2(input_tensor=inputs, weights=None, include_top=False)

    # input: 416 x 416 x 3
    # 1x1conv5_out: 13 x 13 x 1024
    # stage4/block1/relu_1x1conv_1: 26 x 26 x 464
    # stage3/block1/relu_1x1conv_1: 52 x 52 x 232

    x1 = shufflenetv2.get_layer('stage4/block1/relu_1x1conv_1').output

    x2 = shufflenetv2.get_layer('1x1conv5_out').output
    x2 = DarknetConv2D_BN_Leaky(464, (1,1))(x2)

    y1 = compose(
            DarknetConv2D_BN_Leaky(1024, (3,3)),
            #Depthwise_Separable_Conv2D_BN_Leaky(filters=1024, kernel_size=(3, 3), block_id_str='17'),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1)))(x2)

    x2 = compose(
            DarknetConv2D_BN_Leaky(232, (1,1)),
            UpSampling2D(2))(x2)
    y2 = compose(
            Concatenate(),
            DarknetConv2D_BN_Leaky(464, (3,3)),
            #Depthwise_Separable_Conv2D_BN_Leaky(filters=464, kernel_size=(3, 3), block_id_str='18'),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1)))([x2,x1])

    return Model(inputs, [y1,y2])


def tiny_yolo3lite_shufflenetv2_body(inputs, num_anchors, num_classes):
    '''Create Tiny YOLO_v3 Lite ShuffleNetV2 model CNN body in keras.'''
    shufflenetv2 = ShuffleNetV2(input_tensor=inputs, weights=None, include_top=False)

    # input: 416 x 416 x 3
    # 1x1conv5_out: 13 x 13 x 1024
    # stage4/block1/relu_1x1conv_1: 26 x 26 x 464
    # stage3/block1/relu_1x1conv_1: 52 x 52 x 232

    x1 = shufflenetv2.get_layer('stage4/block1/relu_1x1conv_1').output

    x2 = shufflenetv2.get_layer('1x1conv5_out').output
    x2 = DarknetConv2D_BN_Leaky(464, (1,1))(x2)

    y1 = compose(
            #DarknetConv2D_BN_Leaky(1024, (3,3)),
            Depthwise_Separable_Conv2D_BN_Leaky(filters=1024, kernel_size=(3, 3), block_id_str='17'),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1)))(x2)

    x2 = compose(
            DarknetConv2D_BN_Leaky(232, (1,1)),
            UpSampling2D(2))(x2)
    y2 = compose(
            Concatenate(),
            #DarknetConv2D_BN_Leaky(464, (3,3)),
            Depthwise_Separable_Conv2D_BN_Leaky(filters=464, kernel_size=(3, 3), block_id_str='18'),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1)))([x2,x1])

    return Model(inputs, [y1,y2])

