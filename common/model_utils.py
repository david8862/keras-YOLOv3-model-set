#!/usr/bin/python3
# -*- coding=utf-8 -*-
"""Model utility functions."""

import os, sys
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.optimizers.schedules import ExponentialDecay, PolynomialDecay
from tensorflow.keras.experimental import CosineDecay
from tensorflow_model_optimization.sparsity import keras as sparsity
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import Callback

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
from eval import eval_AP


def add_metrics(model, metric_dict):
    '''
    add metric scalar tensor into model, which could be tracked in training
    log and tensorboard callback
    '''
    for (name, metric) in metric_dict.items():
        # seems add_metric() is newly added in tf.keras. So if you
        # want to customize metrics on raw keras model, just use
        # "metrics_names" and "metrics_tensors" as follow:
        #
        #model.metrics_names.append(name)
        #model.metrics_tensors.append(loss)
        model.add_metric(metric, name=name, aggregation='mean')


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


# some global value for lr scheduler
# need to update to CLI option in main()
#lr_base = 1e-3
#total_epochs = 250

#def learning_rate_scheduler(epoch, curr_lr, mode='cosine_decay'):
    #lr_power = 0.9
    #lr = curr_lr

    ## adam default lr
    #if mode is 'adam':
        #lr = 0.001

    ## original lr scheduler
    #if mode is 'power_decay':
        #lr = lr_base * ((1 - float(epoch) / total_epochs) ** lr_power)

    ## exponential decay policy
    #if mode is 'exp_decay':
        #lr = (float(lr_base) ** float(lr_power)) ** float(epoch + 1)

    ## cosine decay policy, including warmup and hold stage
    #if mode is 'cosine_decay':
        ##warmup & hold hyperparams, adjust for your training
        #warmup_epochs = 0
        #hold_base_rate_epochs = 0
        #warmup_lr = lr_base * 0.01
        #lr = 0.5 * lr_base * (1 + np.cos(
             #np.pi * float(epoch - warmup_epochs - hold_base_rate_epochs) /
             #float(total_epochs - warmup_epochs - hold_base_rate_epochs)))

        #if hold_base_rate_epochs > 0 and epoch < warmup_epochs + hold_base_rate_epochs:
            #lr = lr_base

        #if warmup_epochs > 0 and epoch < warmup_epochs:
            #if lr_base < warmup_lr:
                #raise ValueError('learning_rate_base must be larger or equal to '
                                 #'warmup_learning_rate.')
            #slope = (lr_base - warmup_lr) / float(warmup_epochs)
            #warmup_rate = slope * float(epoch) + warmup_lr
            #lr = warmup_rate

    #if mode is 'progressive_drops':
        ## drops as progression proceeds, good for sgd
        #if epoch > 0.9 * total_epochs:
            #lr = 0.0001
        #elif epoch > 0.75 * total_epochs:
            #lr = 0.001
        #elif epoch > 0.5 * total_epochs:
            #lr = 0.01
        #else:
            #lr = 0.1

    #print('learning_rate change to: {}'.format(lr))
    #return lr


def get_lr_scheduler(learning_rate, decay_type, decay_steps):
    if decay_type:
        decay_type = decay_type.lower()

    if decay_type == None:
        lr_scheduler = learning_rate
    elif decay_type == 'cosine':
        lr_scheduler = CosineDecay(initial_learning_rate=learning_rate, decay_steps=decay_steps)
    elif decay_type == 'exponential':
        lr_scheduler = ExponentialDecay(initial_learning_rate=learning_rate, decay_steps=decay_steps, decay_rate=0.9)
    elif decay_type == 'polynomial':
        lr_scheduler = PolynomialDecay(initial_learning_rate=learning_rate, decay_steps=decay_steps, end_learning_rate=learning_rate/100)
    else:
        raise ValueError('Unsupported lr decay type')

    return lr_scheduler


def get_optimizer(optim_type, learning_rate, decay_type='cosine', decay_steps=100000):
    optim_type = optim_type.lower()

    lr_scheduler = get_lr_scheduler(learning_rate, decay_type, decay_steps)

    if optim_type == 'adam':
        optimizer = Adam(learning_rate=lr_scheduler)
    elif optim_type == 'rmsprop':
        optimizer = RMSprop(learning_rate=lr_scheduler)
    elif optim_type == 'sgd':
        optimizer = SGD(learning_rate=lr_scheduler)
    else:
        raise ValueError('Unsupported optimizer type')

    return optimizer


class EvalCallBack(Callback):
    def __init__(self, annotation_lines, anchors, class_names, model_image_size, log_dir, eval_epoch_interval=10, save_eval_checkpoint=False):
        self.annotation_lines = annotation_lines
        self.anchors = anchors
        self.class_names = class_names
        self.model_image_size = model_image_size
        self.log_dir = log_dir
        self.eval_epoch_interval = eval_epoch_interval
        self.save_eval_checkpoint = save_eval_checkpoint
        self.best_mAP = 0.0

    def get_eval_model(self, model):
        # We strip the extra layers in training model to get eval model
        num_anchors = len(self.anchors)

        if num_anchors == 9:
            # YOLOv3 use 9 anchors and 3 prediction layers.
            # Has 7 extra layers (including metrics) in training model
            y1 = model.layers[-10].output
            y2 = model.layers[-9].output
            y3 = model.layers[-8].output

            eval_model = Model(inputs=model.input[0], outputs=[y1,y2,y3])
        elif num_anchors == 6:
            # Tiny YOLOv3 use 6 anchors and 2 prediction layers.
            # Has 6 extra layers in training model
            y1 = model.layers[-8].output
            y2 = model.layers[-7].output

            eval_model = Model(inputs=model.input[0], outputs=[y1,y2])
        elif num_anchors == 5:
            # YOLOv2 use 5 anchors and 1 prediction layer.
            # Has 6 extra layers in training model
            eval_model = Model(inputs=model.input[0], outputs=model.layers[-7].output)
        else:
            raise ValueError('Invalid anchor set')

        return eval_model


    def on_epoch_end(self, epoch, logs=None):
        if (epoch+1) % self.eval_epoch_interval == 0:
            # Do eval every eval_epoch_interval epochs
            eval_model = self.get_eval_model(self.model)
            mAP = eval_AP(eval_model, 'H5', self.annotation_lines, self.anchors, self.class_names, self.model_image_size, eval_type='VOC', iou_threshold=0.5, conf_threshold=0.001, save_result=False)
            if self.save_eval_checkpoint and mAP > self.best_mAP:
                # Save best mAP value and model checkpoint
                self.best_mAP = mAP
                self.model.save(self.log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}-mAP{mAP:.3f}.h5'.format(epoch=(epoch+1), loss=logs.get('loss'), val_loss=logs.get('val_loss'), mAP=mAP))
