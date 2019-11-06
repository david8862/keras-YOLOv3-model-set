#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
create YOLOv2 models with different backbone & head
"""
import warnings
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Lambda, Conv2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow_model_optimization.sparsity import keras as sparsity

from yolo2.models.yolo2_darknet import yolo2_body
from yolo2.models.yolo2_mobilenet import yolo2_mobilenet_body, yolo2lite_mobilenet_body
from yolo2.loss import yolo2_loss
from yolo2.postprocess import batched_yolo2_postprocess


# A map of model type to construction info list for YOLOv2
#
# info list format:
#   [model_function, backbone_length, pretrain_weight_path]
#
yolo2_model_map = {
    'yolo2_darknet': [yolo2_body, 60, 'weights/darknet19.h5'],
    'yolo2_mobilenet': [yolo2_mobilenet_body, 87, None],
    'yolo2_mobilenet_lite': [yolo2lite_mobilenet_body, 87, None],

}


def get_yolo2_model(model_type, num_anchors, num_classes, input_tensor=None, input_shape=None, model_pruning=False, pruning_end_step=10000):
    #prepare input tensor
    if input_shape:
        input_tensor = Input(shape=input_shape, name='image_input')

    if input_tensor is None:
        input_tensor = Input(shape=(None, None, 3), name='image_input')

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
        raise ValueError('This model type is not supported now')

    if model_pruning:
        model_body = get_pruning_model(model_body, begin_step=0, end_step=pruning_end_step)

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


def get_pruning_model(model, begin_step, end_step):
    import tensorflow as tf
    if tf.__version__.startswith('2'):
        # model pruning API is not supported in TF 2.0 yet
        raise Exception('model pruning is not fully supported in TF 2.x, Please switch env to TF 1.x for this feature')

    pruning_params = {
      'pruning_schedule': sparsity.PolynomialDecay(initial_sparsity=0.0,
                                                   final_sparsity=0.7,
                                                   begin_step=begin_step,
                                                   end_step=end_step,
                                                   frequency=100)
    }

    pruning_model = sparsity.prune_low_magnitude(model, **pruning_params)
    return pruning_model



def get_yolo2_train_model(model_type, anchors, num_classes, weights_path=None, freeze_level=1, optimizer=Adam(lr=1e-3, decay=1e-6), label_smoothing=0, model_pruning=False, pruning_end_step=10000):
    '''create the training model, for YOLOv2'''
    #K.clear_session() # get a new session
    num_anchors = len(anchors)

    # input boxes in form of relative x, y, w, h, class
    boxes_input = Input(shape=(None, 5))
    detectors_mask_input = Input(shape=(None, None, num_anchors, 1))
    matching_boxes_input = Input(shape=(None, None, num_anchors, 5))


    model_body, backbone_len = get_yolo2_model(model_type, num_anchors, num_classes, model_pruning=model_pruning, pruning_end_step=pruning_end_step)
    print('Create YOLOv2 {} model with {} anchors and {} classes.'.format(model_type, num_anchors, num_classes))
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

    model_loss, location_loss, confidence_loss, class_loss = Lambda(yolo2_loss, name='yolo_loss',
            arguments={'anchors': anchors, 'num_classes': num_classes})(
            [model_body.output, boxes_input, detectors_mask_input, matching_boxes_input])

    model = Model([model_body.input, boxes_input, detectors_mask_input, matching_boxes_input], model_loss)

    model.compile(optimizer=optimizer, loss={
        # use custom yolo_loss Lambda layer.
        'yolo_loss': lambda y_true, y_pred: y_pred})

    loss_dict = {'location_loss':location_loss, 'confidence_loss':confidence_loss, 'class_loss':class_loss}
    add_metrics(model, loss_dict)

    return model


def get_yolo2_inference_model(model_type, anchors, num_classes, weights_path=None, input_shape=None, confidence=0.1):
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
            arguments={'anchors': anchors, 'num_classes': num_classes, 'confidence': confidence})(
        [model_body.output, image_shape])

    model = Model([model_body.input, image_shape], [boxes, scores, classes])

    return model

