#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
create YOLOv3 models with different backbone & head
"""
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from yolo3.models.yolo3_darknet import yolo_body, custom_tiny_yolo_body, yololite_body, tiny_yololite_body, custom_yolo_spp_body
from yolo3.models.yolo3_mobilenet import yolo_mobilenet_body, tiny_yolo_mobilenet_body, yololite_mobilenet_body, tiny_yololite_mobilenet_body
from yolo3.models.yolo3_vgg16 import yolo_vgg16_body, tiny_yolo_vgg16_body
from yolo3.models.yolo3_xception import yolo_xception_body, yololite_xception_body, tiny_yolo_xception_body, tiny_yololite_xception_body
from yolo3.loss import yolo_loss
from yolo3.postprocess import yolo3_postprocess


# A map of model type to construction info list for YOLOv3
#
# info list format:
#   [model_function, backbone_length, pretrain_weight_path]
#
yolo3_model_map = {
    'mobilenet': [yolo_mobilenet_body, 87, None],
    'mobilenet_lite': [yololite_mobilenet_body, 87, None],
    'darknet': [yolo_body, 185, 'weights/darknet53.h5'],
    'darknet_spp': [custom_yolo_spp_body, 185, 'weights/yolov3-spp.h5'],
    #Doesn't have pretrained weights, so no need to return backbone length
    'darknet_lite': [yololite_body, 0, None],
    'vgg16': [yolo_vgg16_body, 19, None],
    'xception': [yolo_xception_body, 132, None],
    'xception_lite': [yololite_xception_body, 132, None],
}


# A map of model type to construction info list for Tiny YOLOv3
#
# info list format:
#   [model_function, backbone_length, pretrain_weight_file]
#
yolo3_tiny_model_map = {
    'mobilenet': [tiny_yolo_mobilenet_body, 87, None],
    'mobilenet_lite': [tiny_yololite_mobilenet_body, 87, None],
    'darknet': [custom_tiny_yolo_body, 20, 'weights/yolov3-tiny.h5'],
    #Doesn't have pretrained weights, so no need to return backbone length
    'darknet_lite': [tiny_yololite_body, 0, None],
    'vgg16': [tiny_yolo_vgg16_body, 19, None],
    'xception': [tiny_yolo_xception_body, 132, None],
    'xception_lite': [tiny_yololite_xception_body, 132, None],
}


def get_yolo3_model(model_type, num_feature_layers, num_anchors, num_classes, input_tensor=None, input_shape=None):
    #prepare input tensor
    if input_shape:
        input_tensor = Input(shape=input_shape, name='image_input')

    if input_tensor == None:
        input_tensor = Input(shape=(None, None, 3), name='image_input')

    #Tiny YOLOv3 model has 6 anchors and 2 feature layers
    if num_feature_layers == 2:
        if model_type in yolo3_tiny_model_map:
            model_function = yolo3_tiny_model_map[model_type][0]
            backbone_len = yolo3_tiny_model_map[model_type][1]
            weights_path = yolo3_tiny_model_map[model_type][2]

            if weights_path:
                model_body = model_function(input_tensor, num_anchors//2, num_classes, weights_path=weights_path)
            else:
                model_body = model_function(input_tensor, num_anchors//2, num_classes)
        else:
            raise ValueError('This model type is not supported now')

    #YOLOv3 model has 9 anchors and 3 feature layers
    elif num_feature_layers == 3:
        if model_type in yolo3_model_map:
            model_function = yolo3_model_map[model_type][0]
            backbone_len = yolo3_model_map[model_type][1]
            weights_path = yolo3_model_map[model_type][2]

            if weights_path:
                model_body = model_function(input_tensor, num_anchors//3, num_classes, weights_path=weights_path)
            else:
                model_body = model_function(input_tensor, num_anchors//3, num_classes)
        else:
            raise ValueError('This model type is not supported now')
    else:
        raise ValueError('Unsupported model type')
    return model_body, backbone_len



def add_metrics(model, loss_dict):
    '''
    add loss scalar into model, which could be tracked in training
    log and tensorboard callback
    '''
    for (name, loss) in loss_dict.items():
        # seems add_metric() is newly added in tf.keras. So if you
        # want to customize metrics on raw keras model, just use
        # "metrics_names" and "metrics_tensors" as follow:
        #
        #model.metrics_names.append(name)
        #model.metrics_tensors.append(loss)
        model.add_metric(loss, name=name, aggregation='mean')


def get_yolo3_train_model(model_type, anchors, num_classes, weights_path=None, freeze_level=1, learning_rate=1e-3):
    '''create the training model, for YOLOv3'''
    K.clear_session() # get a new session
    num_anchors = len(anchors)
    #YOLOv3 model has 9 anchors and 3 feature layers but
    #Tiny YOLOv3 model has 6 anchors and 2 feature layers,
    #so we can calculate feature layers number to get model type
    num_feature_layers = num_anchors//3

    #feature map target value, so its shape should be like:
    # [
    #  (image_height/32, image_width/32, 3, num_classes+5),
    #  (image_height/16, image_width/16, 3, num_classes+5),
    #  (image_height/8, image_width/8, 3, num_classes+5)
    # ]
    y_true = [Input(shape=(None, None, 3, num_classes+5)) for l in range(num_feature_layers)]

    model_body, backbone_len = get_yolo3_model(model_type, num_feature_layers, num_anchors, num_classes)
    print('Create {} YOLOv3 {} model with {} anchors and {} classes.'.format('Tiny' if num_feature_layers==2 else '', model_type, num_anchors, num_classes))

    if weights_path:
        model_body.load_weights(weights_path, by_name=True)#, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))

    if freeze_level in [1, 2]:
        # Freeze the backbone part or freeze all but final feature map & input layers.
        num = (backbone_len, len(model_body.layers)-3)[freeze_level-1]
        for i in range(num): model_body.layers[i].trainable = False
        print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))
    elif freeze_level == 0:
        # Unfreeze all layers.
        for i in range(len(model_body.layers)):
            model_body.layers[i].trainable= True
        print('Unfreeze all of the layers.')

    model_loss, location_loss, confidence_loss, class_loss = Lambda(yolo_loss, name='yolo_loss',
            arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5, 'use_focal_loss': False, 'use_softmax_loss': False, 'use_giou_loss': False})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    model.compile(optimizer=Adam(lr=learning_rate), loss={
        # use custom yolo_loss Lambda layer.
        'yolo_loss': lambda y_true, y_pred: y_pred})

    loss_dict = {'location_loss':location_loss, 'confidence_loss':confidence_loss, 'class_loss':class_loss}
    add_metrics(model, loss_dict)

    return model


# TODO(david8862): Unfinished work, need to implement
# a batch-wise postprocess layer/model for inference
def get_yolo3_inference_model(model_type, anchors, num_classes, weights_path=None, confidence=0.1):
    '''create the inference model, for YOLOv3'''
    K.clear_session() # get a new session
    num_anchors = len(anchors)
    #YOLOv3 model has 9 anchors and 3 feature layers but
    #Tiny YOLOv3 model has 6 anchors and 2 feature layers,
    #so we can calculate feature layers number to get model type
    num_feature_layers = num_anchors//3

    image_shape = Input(shape=(2,))

    model_body, _ = get_yolo3_model(model_type, num_feature_layers, num_anchors, num_classes)
    print('Create {} YOLOv3 {} model with {} anchors and {} classes.'.format('Tiny' if num_feature_layers==2 else '', model_type, num_anchors, num_classes))

    if weights_path:
        model_body.load_weights(weights_path, by_name=True)#, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))

    boxes, scores, classes = Lambda(yolo3_postprocess, name='yolo3_postprocess',
            arguments={'anchors': anchors, 'num_classes': num_classes, 'confidence': confidence})(
        [*model_body.output, image_shape])
    model = Model([model_body.input, image_shape], [boxes, scores, classes])

    return model

