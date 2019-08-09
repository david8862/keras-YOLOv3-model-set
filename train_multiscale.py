"""
Retrain the YOLO model for your own dataset.
"""

import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import os, random, time, argparse
from yolo3.model import get_yolo3_model
from yolo3.data import data_generator_wrapper
from yolo3.utils import get_classes, get_anchors
from  multiprocessing import Process, Queue


#import tensorflow as tf
#config = tf.ConfigProto()
#config.gpu_options.allow_growth=True   #dynamic alloc GPU resource
#config.gpu_options.per_process_gpu_memory_fraction = 0.9  #GPU memory threshold 0.3
#session = tf.Session(config=config)

## set session
#K.set_session(session)


def train_on_scale(model_type, input_shape, lines, val_split, anchors, class_names,
        callbacks, log_dir, epochs, initial_epoch,
        batch_size=8,
        weights_path=None,
        load_pretrained=True,
        freeze_level=0):

    import tensorflow as tf
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True   #dynamic alloc GPU resource
    config.gpu_options.per_process_gpu_memory_fraction = 0.9  #GPU memory threshold 0.3
    session = tf.Session(config=config)

    # set session
    K.set_session(session)

    num_classes = len(class_names)
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val

    is_tiny_version = len(anchors)==6 # default setting
    model = get_yolo3_model(model_type, input_shape, anchors, num_classes, load_pretrained=load_pretrained,
                        weights_path=weights_path, transfer_learn=True, freeze_level=freeze_level) # make sure you know what you freeze
    model.summary()

    batch_size = batch_size
    print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
    model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
        steps_per_epoch=max(1, num_train//batch_size),
        validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes),
        validation_steps=max(1, num_val//batch_size),
        epochs=epochs,
        initial_epoch=initial_epoch,
        callbacks=callbacks)

    weights_path = log_dir + 'trained_epoch{}_shape{}.h5'.format(epochs, input_shape[0])
    model.save(weights_path)
    #return model


def _main(model_type):
    annotation_path = 'trainval.txt'
    log_dir = 'logs/000/'
    classes_path = 'model_data/voc_classes.txt'
    anchors_path = 'model_data/yolo_anchors.txt'
    #model_type = 'mobilenet'
    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    anchors = get_anchors(anchors_path)

    val_split = 0.1
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val

    # Callbacks config
    logging = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=False, write_grads=False, write_images=False, update_freq='batch')
    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        monitor='val_loss',
        verbose=1,
        save_weights_only=False,
        save_best_only=True,
        period=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=30, verbose=1)

    callbacks = [logging, checkpoint, reduce_lr, early_stopping]

    # get initial base weights and input_shape/batch_size list
    # input_shape/batch_size list will be used in multi scale training
    is_tiny_version = len(anchors)==6 # default setting
    if is_tiny_version:
        input_shape_list = [(320,320), (416,416), (512,512), (608,608)]
        batch_size_list = [2, 4, 8, 16]
    else:
        # due to GPU memory limit, we could only use small
        # input_shape and batch_size for full YOLOv3 model
        input_shape_list = [(320,320), (416,416), (480,480)]
        batch_size_list = [2, 4, 8]


    # Train 40 epochs with frozen layers first, to get a stable loss.
    input_shape = (416,416) # multiple of 32, hw
    batch_size = 8
    initial_epoch = 0
    epochs = 40
    weights_path = None

    p = Process(target=train_on_scale, args=(model_type, input_shape, lines, val_split, anchors, class_names, callbacks, log_dir, epochs, initial_epoch, batch_size, weights_path, False, 1))
    p.start()
    p.join()

    weights_path = log_dir + 'trained_epoch{}_shape{}.h5'.format(epochs, input_shape[0])

    # wait 3s for GPU free
    time.sleep(3)

    # Do multi-scale training on different input shape
    # change every 20 epochs
    for epoch_step in range(60, 300, 20):
        input_shape = input_shape_list[random.randint(0,len(input_shape_list)-1)]
        batch_size = batch_size_list[random.randint(0,len(batch_size_list)-1)]
        initial_epoch = epochs
        epochs = epoch_step

        p = Process(target=train_on_scale, args=(model_type, input_shape, lines, val_split, anchors, class_names, callbacks, log_dir, epochs, initial_epoch, batch_size, weights_path, True, 0))
        p.start()
        p.join()
        # save the trained model and load in next round for different input shape
        weights_path = log_dir + 'trained_epoch{}_shape{}.h5'.format(epochs, input_shape[0])

        # wait 3s for GPU free
        time.sleep(3)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', required=False,
            help='YOLO model type: mobilenet_lite/mobilenet/darknet/vgg16, default=mobilenet_lite', type=str, default='mobilenet_lite')

    args = parser.parse_args()
    _main(args.model_type)

