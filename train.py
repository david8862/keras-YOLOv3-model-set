#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Retrain the YOLO model for your own dataset.
"""

import os, random, argparse
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TerminateOnNaN, LambdaCallback
from yolo3.model import get_yolo3_model
from yolo3.data import data_generator_wrapper
from yolo3.utils import get_classes, get_anchors

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth=True   #dynamic alloc GPU resource
config.gpu_options.per_process_gpu_memory_fraction = 0.9  #GPU memory threshold 0.3
session = tf.Session(config=config)

# set session
K.set_session(session)


def get_multiscale_param(model_type, tiny_version):
    # get input_shape & batch_size list for multiscale training
    if (model_type == 'darknet' or model_type == 'xception') and not tiny_version:
        # due to GPU memory limit, we could only use small input_shape and batch_size
        # for full YOLOv3 and Xception models
        input_shape_list = [(320,320), (416,416), (480,480)]
        batch_size_list = [4, 8]
    else:
        input_shape_list = [(320,320), (416,416), (512,512), (608,608)]
        batch_size_list = [4, 8, 16]

    return input_shape_list, batch_size_list


def _main(args):
    annotation_file = args.annotation_file
    log_dir = 'logs/000/'
    classes_path = args.classes_path
    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    if args.tiny_version:
        anchors_path = 'model_data/tiny_yolo_anchors.txt'
    else:
        anchors_path = 'model_data/yolo_anchors.txt'
    anchors = get_anchors(anchors_path)

    # get freeze level according to CLI option
    if args.weights_path:
        freeze_level = 0
    else:
        freeze_level = 1

    if args.freeze_level:
        freeze_level = args.freeze_level


    logging = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=False, write_grads=False, write_images=False, update_freq='batch')
    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        monitor='val_loss',
        verbose=1,
        save_weights_only=False,
        save_best_only=True,
        period=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=30, verbose=1)
    terminate_on_nan = TerminateOnNaN()
    sess_saver_callback = LambdaCallback(on_epoch_end=lambda epoch, logs: tf.train.Saver().save(sess, log_dir+'model-epoch%03d-loss%.3f-val_loss%.3f' % (epoch, logs['loss'], logs['val_loss'])))

    # get train&val dataset
    val_split = args.val_split
    with open(annotation_file) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val

    if args.multiscale:
        input_shape_list, batch_size_list = get_multiscale_param(args.model_type, args.tiny_version)
    else:
        input_shape_list = [args.model_image_size]
        batch_size_list = [args.batch_size]

    model = get_yolo3_model(args.model_type, anchors, num_classes, weights_path=args.weights_path, freeze_level=freeze_level, learning_rate=args.learning_rate)
    model.summary()

    # Train some initial epochs with frozen layers first if needed, to get a stable loss.
    input_shape = args.model_image_size
    assert (input_shape[0]%32 == 0 and input_shape[1]%32 == 0), 'Multiples of 32 required'
    batch_size = args.batch_size
    initial_epoch = 0
    epochs = args.init_epoch
    print('Train on {} samples, val on {} samples, with batch size {}, input_shape {}.'.format(num_train, num_val, batch_size, input_shape))
    model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
            steps_per_epoch=max(1, num_train//batch_size),
            validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes),
            validation_steps=max(1, num_val//batch_size),
            epochs=epochs,
            initial_epoch=initial_epoch,
            callbacks=[logging, checkpoint, terminate_on_nan])

    # free the whole network for further training
    print("Unfreeze and continue training, to fine-tune.")
    for i in range(len(model.layers)):
        model.layers[i].trainable = True
    model.compile(optimizer=Adam(lr=args.learning_rate/2), loss={'yolo_loss': lambda y_true, y_pred: y_pred}) # recompile to apply the change

    # Do multi-scale training on different input shape
    # change every "rescale_interval" epochs
    interval = args.rescale_interval
    for epoch_step in range(epochs+interval, args.total_epoch, interval):
        input_shape = input_shape_list[random.randint(0,len(input_shape_list)-1)]
        batch_size = batch_size_list[random.randint(0,len(batch_size_list)-1)]
        initial_epoch = epochs
        epochs = epoch_step
        print('Train on {} samples, val on {} samples, with batch size {}, input_shape {}.'.format(num_train, num_val, batch_size, input_shape))
        model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
            steps_per_epoch=max(1, num_train//batch_size),
            validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes),
            validation_steps=max(1, num_val//batch_size),
            epochs=epochs,
            initial_epoch=initial_epoch,
            callbacks=[logging, checkpoint, reduce_lr, early_stopping, terminate_on_nan])

    # Finally store model
    model.save(log_dir + 'trained_final.h5')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Model definition options
    parser.add_argument('--model_type', type=str, required=False, default='mobilenet_lite',
        help='YOLO model type: mobilenet_lite/mobilenet/darknet/vgg16/xception/xception_lite, default=mobilenet_lite')
    parser.add_argument('--tiny_version', default=False, action="store_true",
        help='Whether to use a tiny YOLO version')
    parser.add_argument('--model_image_size', type=str,required=False, default='416x416',
        help = "Initial model image input size as <num>x<num>, default 416x416")
    parser.add_argument('--weights_path', type=str,required=False, default=None,
        help = "Pretrained model/weights file for fine tune")
    parser.add_argument('--freeze_level', type=int,required=False, default=None,
        help = "Freeze level of the training model. 0:NA/1:backbone/2:only open feature_map")

    # Data options
    parser.add_argument('--annotation_file', type=str, required=False, default='trainval.txt',
        help='train&val annotation txt file, default=trainval.txt')
    parser.add_argument('--val_split', type=float,required=False, default=0.1,
        help = "validation data persentage in dataset, default=0.1")
    parser.add_argument('--classes_path', type=str, required=False, default='model_data/voc_classes.txt',
        help='path to class definitions, default=model_data/voc_classes.txt')

    # Training options
    parser.add_argument('--learning_rate', type=float,required=False, default=1e-3,
        help = "Initial learning rate, default=0.001")
    parser.add_argument('--batch_size', type=int,required=False, default=16,
        help = "Initial batch size for train, default=16")
    parser.add_argument('--init_epoch', type=int,required=False, default=40,
        help = "Initial stage training epochs, default=40")
    parser.add_argument('--total_epoch', type=int,required=False, default=300,
        help = "Total training epochs, default=300")
    parser.add_argument('--multiscale', default=False, action="store_true",
        help='Whether to use multiscale training')
    parser.add_argument('--rescale_interval', type=int,required=False, default=10,
        help = "Number of epochs to rescale input image in, default=10")

    args = parser.parse_args()
    height, width = args.model_image_size.split('x')
    args.model_image_size = (int(height), int(width))

    _main(args)
