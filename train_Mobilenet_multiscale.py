"""
Retrain the YOLO model for your own dataset.
"""

import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import os, random
from yolo3.model_Mobilenet import preprocess_true_boxes, yolo_mobilenet_body, tiny_yolo_mobilenet_body, custom_yolo_mobilenet_body, yolo_loss
from yolo3.utils import get_random_data

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth=True   #dynamic alloc GPU resource
config.gpu_options.per_process_gpu_memory_fraction = 0.9  #GPU memory threshold 0.3
session = tf.Session(config=config)

# set session
K.set_session(session)


def train_on_scale(input_shape, lines, val_split, anchors, class_names,
        callbacks, epochs, initial_epoch,
        batch_size=8,
        weights_path=None,
        load_pretrained=True,
        freeze=False):

    num_classes = len(class_names)
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val

    is_tiny_version = len(anchors)==6 # default setting
    if is_tiny_version:
        model = create_tiny_model(input_shape, anchors, num_classes, load_pretrained=load_pretrained,
                             weights_path=weights_path,
            freeze_body=1, transfer_learn=True)
    else:
        model = create_model(input_shape, anchors, num_classes, load_pretrained=load_pretrained,
                             weights_path=weights_path,
            freeze_body=1, transfer_learn=True) # make sure you know what you freeze


    print("Train with input_shape {}".format(input_shape))
    if freeze == False:
        print('Unfreeze all of the layers.')
        for i in range(len(model.layers)):
            model.layers[i].trainable= True
    model.compile(optimizer=Adam(lr=1e-4), loss={'yolo_loss': lambda y_true, y_pred: y_pred}) # recompile to apply the change
    batch_size = batch_size
    print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
    model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
        steps_per_epoch=max(1, num_train//batch_size),
        validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes),
        validation_steps=max(1, num_val//batch_size),
        epochs=epochs,
        initial_epoch=initial_epoch,
        callbacks=callbacks)

    return model



def _main():
    train_path = 'roborock_2007_trainval.txt'
    val_path = 'val.txt'
    log_dir = 'logs/000/'
    classes_path = 'model_data/roborock_classes.txt'
    anchors_path = 'model_data/tiny_yolo_anchors.txt'
    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    anchors = get_anchors(anchors_path)

    val_split = 0.1
    with open(train_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val

    # Callbacks config
    logging = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        monitor='val_loss',
        verbose=1,
        save_weights_only=False,
        save_best_only=True,
        period=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=30, verbose=1)

    callbacks = [logging, checkpoint, reduce_lr, early_stopping]

    # get initial base weights
    is_tiny_version = len(anchors)==6 # default setting
    if is_tiny_version:
        weights_path = 'model_data/tiny_yolo_weights.h5'
    else:
        weights_path = 'model_data/yolo_weights.h5'

    # Train 40 epochs with frozen layers first, to get a stable loss.
    input_shape = (416,416) # multiple of 32, hw
    batch_size = 8
    initial_epoch = 0
    epochs = 40

    model = train_on_scale(input_shape, lines, val_split, anchors, class_names, callbacks, epochs=epochs, initial_epoch=initial_epoch, batch_size=batch_size, weights_path=weights_path, load_pretrained=False,freeze=True)
    weights_path = log_dir + 'trained_epoch{}_shape{}.h5'.format(epochs, input_shape[0])
    model.save(weights_path)


    # input shape and batch size list to
    # choose from in multi scale training
    input_shape_list = [(320,320), (416,416), (608,608)]
    batch_size_list = [16, 8, 2]

    # Do multi-scale training on different input shape
    # change every 20 epochs
    for epoch_step in range(60, 200, 20):
        input_shape = input_shape_list[random.randint(0,len(input_shape_list)-1)]
        batch_size = batch_size_list[random.randint(0,len(batch_size_list)-1)]
        initial_epoch = epochs
        epochs = epoch_step

        model = train_on_scale(input_shape, lines, val_split, anchors, class_names, callbacks, epochs=epochs, initial_epoch=initial_epoch, batch_size=batch_size, weights_path=weights_path, load_pretrained=True,freeze=False)
        # save the trained model and load in next round for different input shape
        weights_path = log_dir + 'trained_epoch{}_shape{}.h5'.format(epochs, input_shape[0])
        model.save(weights_path)




def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)


def create_model(input_shape, anchors, num_classes, freeze_body=1, load_pretrained=False,
            weights_path='model_data/yolo_weights.h5', transfer_learn=True):
    '''create the training model'''
    K.clear_session() # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [Input(shape=(h//{0:32, 1:16, 2:8}[l], w//{0:32, 1:16, 2:8}[l], \
        num_anchors//3, num_classes+5)) for l in range(3)]

    #model_body = yolo_mobilenet_body(image_input, num_anchors//3, num_classes)
    model_body = custom_yolo_mobilenet_body(image_input, num_anchors//3, num_classes)
    print('Create YOLOv3 MobileNet model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True)#, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))

    if transfer_learn:
        if freeze_body in [1, 2]:
            # Freeze the mobilenet body or freeze all but final feature map & input layers.
            num = (87, len(model_body.layers)-3)[freeze_body-1]
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    return model

def create_tiny_model(input_shape, anchors, num_classes, freeze_body=1, load_pretrained=False,
            weights_path='model_data/tiny_yolo_weights.h5', transfer_learn=True):
    '''create the training model, for Tiny YOLOv3 MobileNet'''
    K.clear_session() # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [Input(shape=(h//{0:32, 1:16}[l], w//{0:32, 1:16}[l], \
        num_anchors//2, num_classes+5)) for l in range(2)]

    model_body = tiny_yolo_mobilenet_body(image_input, num_anchors//2, num_classes)
    print('Create Tiny YOLOv3 MobileNet model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True)#, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))

    if transfer_learn:
        if freeze_body in [1, 2]:
            # Freeze the mobilenet body or freeze all but final feature map & input layers.
            num = (87, len(model_body.layers)-2)[freeze_body-1]
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.7})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    return model

def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes):
    '''data generator for fit_generator'''
    n = len(annotation_lines)
    i = 0
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            if i==0:
                np.random.shuffle(annotation_lines)
            image, box = get_random_data(annotation_lines[i], input_shape, random=True)
            image_data.append(image)
            box_data.append(box)
            i = (i+1) % n
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        yield [image_data, *y_true], np.zeros(batch_size)

def data_generator_wrapper(annotation_lines, batch_size, input_shape, anchors, num_classes):
    n = len(annotation_lines)
    if n==0 or batch_size<=0: return None
    return data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes)

if __name__ == '__main__':
    _main()


