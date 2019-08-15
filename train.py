#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Retrain the YOLO model for your own dataset.
"""

import os, argparse
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TerminateOnNaN, LambdaCallback
#from tensorflow_model_optimization.sparsity import keras as sparsity
from yolo3.model import get_yolo3_model
from yolo3.data import data_generator_wrapper
from yolo3.utils import resize_anchors, get_classes, get_anchors

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth=True   #dynamic alloc GPU resource
config.gpu_options.per_process_gpu_memory_fraction = 0.9  #GPU memory threshold 0.3
session = tf.Session(config=config)

# set session
K.set_session(session)

def _main(args):
    annotation_path = 'trainval.txt'
    log_dir = 'logs/000/'
    classes_path = 'model_data/voc_classes.txt'
    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    if args.tiny_version:
        anchors_path = 'model_data/tiny_yolo_anchors.txt'
    else:
        anchors_path = 'model_data/yolo_anchors.txt'
    base_anchors = get_anchors(anchors_path)
    if args.weights_path:
        freeze_level = 0
    else:
        freeze_level = 1

    input_shape = (416,416) # multiple of 32, hw

    # resize base anchors acoording to input shape
    anchors = resize_anchors(base_anchors, input_shape)

    logging = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=False, write_grads=False, write_images=False, update_freq='batch')
    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        monitor='val_loss',
        verbose=1,
        save_weights_only=False,
        save_best_only=True,
        period=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=30, verbose=1)
    terminate_on_nan = TerminateOnNaN()
    sess_saver_callback = LambdaCallback(on_epoch_end=lambda epoch, logs: tf.train.Saver().save(sess, log_dir+'model-epoch%03d-loss%.3f-val_loss%.3f' % (epoch, logs['loss'], logs['val_loss'])))

    val_split = 0.1
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val

    model = get_yolo3_model(args.model_type, input_shape, anchors, num_classes, weights_path=args.weights_path, freeze_level=freeze_level, learning_rate=args.learning_rate)
    model.summary()

    #pruning_callbacks = [sparsity.UpdatePruningStep(), sparsity.PruningSummaries(log_dir=log_dir, profile_batch=0)]

    # Train with frozen layers first, to get a stable loss.
    # Adjust num epochs to your dataset. This step is enough to obtain a not bad model.
    if True:
        batch_size = args.batch_size
        epochs = 50
        #end_step = np.ceil(1.0 * num_train / batch_size).astype(np.int32) * epochs
        #model = add_pruning(model, begin_step=0, end_step=end_step)
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
                steps_per_epoch=max(1, num_train//batch_size),
                validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes),
                validation_steps=max(1, num_val//batch_size),
                epochs=epochs,
                initial_epoch=0,
                callbacks=[logging, checkpoint, terminate_on_nan])
        #model = sparsity.strip_pruning(model)
        model.save(log_dir + 'trained_stage_1.h5')

    # Unfreeze and continue training, to fine-tune.
    # Train longer if the result is not good.
    if True:
        print("Unfreeze and continue training, to fine-tune.")
        for i in range(len(model.layers)):
            model.layers[i].trainable = True
        model.compile(optimizer=Adam(lr=args.learning_rate/10), loss={'yolo_loss': lambda y_true, y_pred: y_pred}) # recompile to apply the change
        batch_size = args.batch_size # note that more GPU memory is required after unfreezing the body
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
            steps_per_epoch=max(1, num_train//batch_size),
            validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes),
            validation_steps=max(1, num_val//batch_size),
            epochs=150,
            initial_epoch=50,
            callbacks=[logging, checkpoint, reduce_lr, early_stopping, terminate_on_nan])

    # Lower down batch size for another 50 epochs
    if True:
        print("Lower batch_size to fine-tune.")
        batch_size = batch_size//2 # lower down batch_size for more random gradient search
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
            steps_per_epoch=max(1, num_train//batch_size),
            validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes),
            validation_steps=max(1, num_val//batch_size),
            epochs=200,
            initial_epoch=150,
            callbacks=[logging, checkpoint, reduce_lr, early_stopping, terminate_on_nan])

    # Keep lower down batch size for another 50 epochs
    if True:
        print("Keep lower batch_size to fine-tune.")
        batch_size = batch_size//2 # lower down batch_size for more random gradient search
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
            steps_per_epoch=max(1, num_train//batch_size),
            validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes),
            validation_steps=max(1, num_val//batch_size),
            epochs=250,
            initial_epoch=200,
            callbacks=[logging, checkpoint, reduce_lr, early_stopping, terminate_on_nan])

    # Finally store model
    model.save(log_dir + 'trained_final.h5')


def add_pruning(model, begin_step, end_step):
    print(type(model))
    new_pruning_params = {
      'pruning_schedule': sparsity.PolynomialDecay(initial_sparsity=0.50,
                                                   final_sparsity=0.90,
                                                   begin_step=begin_step,
                                                   end_step=end_step,
                                                   frequency=100)
    }
    pruned_model = sparsity.prune_low_magnitude(model, **new_pruning_params)
    pruned_model.summary()
    pruned_model.compile(optimizer=Adam(lr=1e-3), loss={
        # use custom yolo_loss Lambda layer.
        'prune_low_magnitude_yolo_loss': lambda y_true, y_pred: y_pred})

    return pruned_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', required=False,
            help='YOLO model type: mobilenet_lite/mobilenet/darknet/vgg16/xception/xception_lite, default=mobilenet_lite', type=str, default='mobilenet_lite')
    parser.add_argument('--tiny_version', default=False, action="store_true",
            help='Whether to use a tiny YOLO version')
    parser.add_argument('--weights_path', type=str,required=False, default=None,
        help = "Pretrained model/weights file for fine tune")
    parser.add_argument('--learning_rate', type=float,required=False, default=1e-3,
        help = "Initial learning rate, default=0.001")
    parser.add_argument('--batch_size', type=int,required=False, default=16,
        help = "Initial batch size for train, default=16")

    args = parser.parse_args()
    _main(args)
