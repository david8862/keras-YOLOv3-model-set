#!/usr/bin/python3
# -*- coding=utf-8 -*-
"""Model utility functions."""

from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow_model_optimization.sparsity import keras as sparsity


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


def get_optimizer(optim_type, learning_rate, decay=0):
    optim_type = optim_type.lower()

    if optim_type == 'adam':
        optimizer = Adam(lr=learning_rate, decay=decay)
    elif optim_type == 'rmsprop':
        optimizer = RMSprop(lr=learning_rate, decay=decay)
    elif optim_type == 'sgd':
        optimizer = SGD(lr=learning_rate, decay=decay)
    else:
        raise ValueError('Unsupported optimizer type')

    return optimizer
