#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
create YOLOv5 models with different backbone & head
"""
import warnings
from functools import partial

import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from scaled_yolo4.models.scaled_yolo4_csp_darknet import scaled_yolo4_csp_body
from yolo5.models.yolo5_darknet import yolo5_body
from yolo5.models.yolo5_mobilenet import yolo5_mobilenet_body, yolo5lite_mobilenet_body
from yolo5.models.yolo5_mobilenetv2 import yolo5_mobilenetv2_body, yolo5lite_mobilenetv2_body

from yolo5.loss import yolo5_loss
from yolo5.postprocess import batched_yolo5_postprocess

from common.model_utils import add_metrics, get_pruning_model


# A map of model type to construction info list for YOLOv5
#
# info list format:
#   [model_function, backbone_length, pretrain_weight_path]
#
yolo5_model_map = {
    'yolo5_small_darknet': [partial(yolo5_body, depth_multiple=0.33, width_multiple=0.50), 0, None],
    'yolo5_medium_darknet': [partial(yolo5_body, depth_multiple=0.67, width_multiple=0.75), 0, None],
    'yolo5_large_darknet': [partial(yolo5_body, depth_multiple=1.0, width_multiple=1.0), 0, None],
    'yolo5_xlarge_darknet': [partial(yolo5_body, depth_multiple=1.33, width_multiple=1.25), 0, None],

    'yolo5_mobilenet': [yolo5_mobilenet_body, 87, None],
    'yolo5_mobilenet_lite': [yolo5lite_mobilenet_body, 87, None],
    'yolo5_mobilenetv2': [yolo5_mobilenetv2_body, 155, None],
    'yolo5_mobilenetv2_lite': [yolo5lite_mobilenetv2_body, 155, None],

    'scaled_yolo4_csp_darknet': [scaled_yolo4_csp_body, 0, None],
}

# A map of model type to construction info list for Tiny YOLOv5
#
# info list format:
#   [model_function, backbone_length, pretrain_weight_file]
#
yolo5_tiny_model_map = {
}


def get_yolo5_model(model_type, num_feature_layers, num_anchors, num_classes, input_tensor=None, input_shape=None, model_pruning=False, pruning_end_step=10000):
    #prepare input tensor
    if input_shape:
        input_tensor = Input(shape=input_shape, batch_size=None, name='image_input')

    if input_tensor is None:
        input_tensor = Input(shape=(None, None, 3), batch_size=None, name='image_input')

    #Tiny YOLOv5 model has 6 anchors and 2 feature layers
    if num_feature_layers == 2:
        if model_type in yolo5_tiny_model_map:
            model_function = yolo5_tiny_model_map[model_type][0]
            backbone_len = yolo5_tiny_model_map[model_type][1]
            weights_path = yolo5_tiny_model_map[model_type][2]

            if weights_path:
                model_body = model_function(input_tensor, num_anchors//2, num_classes, weights_path=weights_path)
            else:
                model_body = model_function(input_tensor, num_anchors//2, num_classes)
        else:
            raise ValueError('This model type is not supported now')

    #YOLOv5 model has 9 anchors and 3 feature layers
    elif num_feature_layers == 3:
        if model_type in yolo5_model_map:
            model_function = yolo5_model_map[model_type][0]
            backbone_len = yolo5_model_map[model_type][1]
            weights_path = yolo5_model_map[model_type][2]

            if weights_path:
                model_body = model_function(input_tensor, num_anchors//3, num_classes, weights_path=weights_path)
            else:
                model_body = model_function(input_tensor, num_anchors//3, num_classes)
        else:
            raise ValueError('This model type is not supported now')
    else:
        raise ValueError('model type mismatch anchors')

    if model_pruning:
        model_body = get_pruning_model(model_body, begin_step=0, end_step=pruning_end_step)

    return model_body, backbone_len


def get_yolo5_train_model(model_type, anchors, num_classes, weights_path=None, freeze_level=1, optimizer=Adam(lr=1e-3, decay=0), label_smoothing=0, elim_grid_sense=True, model_pruning=False, pruning_end_step=10000):
    '''create the training model, for YOLOv5'''
    #K.clear_session() # get a new session
    num_anchors = len(anchors)
    #YOLOv5 model has 9 anchors and 3 feature layers but
    #Tiny YOLOv5 model has 6 anchors and 2 feature layers,
    #so we can calculate feature layers number to get model type
    num_feature_layers = num_anchors//3

    #feature map target value, so its shape should be like:
    # [
    #  (image_height/32, image_width/32, 3, num_classes+5),
    #  (image_height/16, image_width/16, 3, num_classes+5),
    #  (image_height/8, image_width/8, 3, num_classes+5)
    # ]
    y_true = [Input(shape=(None, None, 3, num_classes+5), name='y_true_{}'.format(l)) for l in range(num_feature_layers)]

    model_body, backbone_len = get_yolo5_model(model_type, num_feature_layers, num_anchors, num_classes, model_pruning=model_pruning, pruning_end_step=pruning_end_step)
    print('Create {} {} model with {} anchors and {} classes.'.format('Tiny' if num_feature_layers==2 else '', model_type, num_anchors, num_classes))
    print('model layer number:', len(model_body.layers))

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

    model_loss, location_loss, confidence_loss, class_loss = Lambda(yolo5_loss, name='yolo_loss',
            arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5, 'label_smoothing': label_smoothing, 'elim_grid_sense': elim_grid_sense})(
        [*model_body.output, *y_true])

    model = Model([model_body.input, *y_true], model_loss)

    loss_dict = {'location_loss':location_loss, 'confidence_loss':confidence_loss, 'class_loss':class_loss}
    add_metrics(model, loss_dict)

    model.compile(optimizer=optimizer, loss={'yolo_loss': lambda y_true, y_pred: y_pred}) # use custom yolo_loss Lambda layer

    return model


def get_yolo5_inference_model(model_type, anchors, num_classes, weights_path=None, input_shape=None, confidence=0.1, iou_threshold=0.4, elim_grid_sense=True):
    '''create the inference model, for YOLOv5'''
    #K.clear_session() # get a new session
    num_anchors = len(anchors)
    #YOLOv5 model has 9 anchors and 3 feature layers but
    #Tiny YOLOv5 model has 6 anchors and 2 feature layers,
    #so we can calculate feature layers number to get model type
    num_feature_layers = num_anchors//3

    image_shape = Input(shape=(2,), dtype='int64', name='image_shape')

    model_body, _ = get_yolo5_model(model_type, num_feature_layers, num_anchors, num_classes, input_shape=input_shape)
    print('Create {} YOLOv5 {} model with {} anchors and {} classes.'.format('Tiny' if num_feature_layers==2 else '', model_type, num_anchors, num_classes))

    if weights_path:
        model_body.load_weights(weights_path, by_name=False)#, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))

    boxes, scores, classes = Lambda(batched_yolo5_postprocess, name='yolo5_postprocess',
            arguments={'anchors': anchors, 'num_classes': num_classes, 'confidence': confidence, 'iou_threshold': iou_threshold, 'elim_grid_sense': elim_grid_sense})(
        [*model_body.output, image_shape])
    model = Model([model_body.input, image_shape], [boxes, scores, classes])

    return model

