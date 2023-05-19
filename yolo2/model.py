#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
create YOLOv2 models with different backbone & head
"""
import os
import warnings
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from yolo2.models.yolo2_darknet import yolo2_body, tiny_yolo2_body
from yolo2.models.yolo2_mobilenet import yolo2_mobilenet_body, yolo2lite_mobilenet_body, tiny_yolo2_mobilenet_body, tiny_yolo2lite_mobilenet_body
from yolo2.models.yolo2_mobilenetv2 import yolo2_mobilenetv2_body, yolo2lite_mobilenetv2_body, tiny_yolo2_mobilenetv2_body, tiny_yolo2lite_mobilenetv2_body
from yolo2.models.yolo2_xception import yolo2_xception_body, yolo2lite_xception_body
from yolo2.models.yolo2_efficientnet import yolo2_efficientnet_body, yolo2lite_efficientnet_body, tiny_yolo2_efficientnet_body, tiny_yolo2lite_efficientnet_body
from yolo2.models.yolo2_mobilenetv3_large import yolo2_mobilenetv3large_body, yolo2lite_mobilenetv3large_body, tiny_yolo2_mobilenetv3large_body, tiny_yolo2lite_mobilenetv3large_body
from yolo2.models.yolo2_mobilenetv3_small import yolo2_mobilenetv3small_body, yolo2lite_mobilenetv3small_body, tiny_yolo2_mobilenetv3small_body, tiny_yolo2lite_mobilenetv3small_body
from yolo2.loss import yolo2_loss
from yolo2.postprocess import batched_yolo2_postprocess

from common.model_utils import add_metrics, get_pruning_model

ROOT_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')

# A map of model type to construction info list for YOLOv2
#
# info list format:
#   [model_function, backbone_length, pretrain_weight_path]
#
yolo2_model_map = {
    'yolo2_darknet': [yolo2_body, 60, os.path.join(ROOT_PATH, 'weights', 'darknet19.h5')],
    'yolo2_mobilenet': [yolo2_mobilenet_body, 87, None],
    'yolo2_mobilenet_lite': [yolo2lite_mobilenet_body, 87, None],
    'yolo2_mobilenetv2': [yolo2_mobilenetv2_body, 155, None],
    'yolo2_mobilenetv2_lite': [yolo2lite_mobilenetv2_body, 155, None],

    'yolo2_mobilenetv3large': [yolo2_mobilenetv3large_body, 195, None],
    'yolo2_mobilenetv3large_lite': [yolo2lite_mobilenetv3large_body, 195, None],
    'yolo2_mobilenetv3small': [yolo2_mobilenetv3small_body, 166, None],
    'yolo2_mobilenetv3small_lite': [yolo2lite_mobilenetv3small_body, 166, None],

    # NOTE: backbone_length is for EfficientNetB0
    # if change to other efficientnet level, you need to modify it
    'yolo2_efficientnet': [yolo2_efficientnet_body, 235, None],
    'yolo2_efficientnet_lite': [yolo2lite_efficientnet_body, 235, None],

    'yolo2_xception': [yolo2_xception_body, 132, None],
    'yolo2_xception_lite': [yolo2lite_xception_body, 132, None],

    'tiny_yolo2_darknet': [tiny_yolo2_body, 0, None],
    'tiny_yolo2_mobilenet': [tiny_yolo2_mobilenet_body, 87, None],
    'tiny_yolo2_mobilenet_lite': [tiny_yolo2lite_mobilenet_body, 87, None],
    'tiny_yolo2_mobilenetv2': [tiny_yolo2_mobilenetv2_body, 155, None],
    'tiny_yolo2_mobilenetv2_lite': [tiny_yolo2lite_mobilenetv2_body, 155, None],

    'tiny_yolo2_mobilenetv3large': [tiny_yolo2_mobilenetv3large_body, 195, None],
    'tiny_yolo2_mobilenetv3large_lite': [tiny_yolo2lite_mobilenetv3large_body, 195, None],
    'tiny_yolo2_mobilenetv3small': [tiny_yolo2_mobilenetv3small_body, 166, None],
    'tiny_yolo2_mobilenetv3small_lite': [tiny_yolo2lite_mobilenetv3small_body, 166, None],

    # NOTE: backbone_length is for EfficientNetB0
    # if change to other efficientnet level, you need to modify it
    'tiny_yolo2_efficientnet': [tiny_yolo2_efficientnet_body, 235, None],
    'tiny_yolo2_efficientnet_lite': [tiny_yolo2lite_efficientnet_body, 235, None],
}


def get_yolo2_model(model_type, num_anchors, num_classes, input_tensor=None, input_shape=None, model_pruning=False, pruning_end_step=10000):
    #prepare input tensor
    if input_shape:
        input_tensor = Input(shape=input_shape, batch_size=None, name='image_input')

    if input_tensor is None:
        input_tensor = Input(shape=(None, None, 3), batch_size=None, name='image_input')

    #YOLOv2 model has 5 anchors
    if model_type in yolo2_model_map:
        model_function = yolo2_model_map[model_type][0]
        backbone_len = yolo2_model_map[model_type][1]
        weights_path = yolo2_model_map[model_type][2]

        if weights_path:
            model_body = model_function(input_tensor, num_anchors, num_classes, weights_path=weights_path)
        else:
            model_body = model_function(input_tensor, num_anchors, num_classes)
    else:
        raise ValueError('model type mismatch anchors')

    if model_pruning:
        model_body = get_pruning_model(model_body, begin_step=0, end_step=pruning_end_step)

    return model_body, backbone_len



def get_yolo2_train_model(model_type, anchors, num_classes, weights_path=None, freeze_level=1, optimizer=Adam(lr=1e-3, decay=1e-6), label_smoothing=0, elim_grid_sense=False, model_pruning=False, pruning_end_step=10000):
    '''create the training model, for YOLOv2'''
    #K.clear_session() # get a new session
    num_anchors = len(anchors)

    # y_true in form of relative x, y, w, h, objectness, class
    y_true_input = Input(shape=(None, None, num_anchors, 6))

    model_body, backbone_len = get_yolo2_model(model_type, num_anchors, num_classes, model_pruning=model_pruning, pruning_end_step=pruning_end_step)
    print('Create YOLOv2 {} model with {} anchors and {} classes.'.format(model_type, num_anchors, num_classes))
    print('model layer number:', len(model_body.layers))

    if weights_path:
        model_body.load_weights(weights_path, by_name=True)#, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))

    if freeze_level in [1, 2]:
        # Freeze the backbone part or freeze all but final feature map & input layers.
        num = (backbone_len, len(model_body.layers)-2)[freeze_level-1]
        for i in range(num): model_body.layers[i].trainable = False
        print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))
    elif freeze_level == 0:
        # Unfreeze all layers.
        for i in range(len(model_body.layers)):
            model_body.layers[i].trainable= True
        print('Unfreeze all of the layers.')

    model_loss, location_loss, confidence_loss, class_loss = Lambda(yolo2_loss, name='yolo_loss',
            arguments={'anchors': anchors, 'num_classes': num_classes, 'label_smoothing': label_smoothing, 'elim_grid_sense': elim_grid_sense})(
            [model_body.output, y_true_input])

    model = Model([model_body.input, y_true_input], model_loss)

    loss_dict = {'location_loss':location_loss, 'confidence_loss':confidence_loss, 'class_loss':class_loss}
    add_metrics(model, loss_dict)

    model.compile(optimizer=optimizer, loss={'yolo_loss': lambda y_true, y_pred: y_pred}) # use custom yolo_loss Lambda layer

    return model


def get_yolo2_inference_model(model_type, anchors, num_classes, weights_path=None, input_shape=None, confidence=0.1, iou_threshold=0.4, elim_grid_sense=False):
    '''create the inference model, for YOLOv2'''
    #K.clear_session() # get a new session
    num_anchors = len(anchors)

    image_shape = Input(shape=(2,), dtype='int64', name='image_shape')

    model_body, _ = get_yolo2_model(model_type, num_anchors, num_classes, input_shape=input_shape)
    print('Create YOLOv2 {} model with {} anchors and {} classes.'.format(model_type, num_anchors, num_classes))

    if weights_path:
        model_body.load_weights(weights_path, by_name=False)#, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))

    boxes, scores, classes = Lambda(batched_yolo2_postprocess, name='yolo2_postprocess',
            arguments={'anchors': anchors, 'num_classes': num_classes, 'confidence': confidence, 'iou_threshold': iou_threshold, 'elim_grid_sense': elim_grid_sense})(
        [model_body.output, image_shape])

    model = Model([model_body.input, image_shape], [boxes, scores, classes])

    return model

