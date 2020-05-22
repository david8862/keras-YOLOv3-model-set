#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# train backbone network with imagenet dataset
#

import os, sys, argparse
import numpy as np
from multiprocessing import cpu_count

import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler, TerminateOnNaN
from tensorflow.keras.utils import multi_gpu_model

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
from shufflenet import ShuffleNet
from shufflenet_v2 import ShuffleNetV2

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', '..'))
from yolo3.models.yolo3_nano import NanoNet
from yolo3.models.yolo3_darknet import DarkNet53
from yolo4.models.yolo4_darknet import CSPDarkNet53

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#import tensorflow as tf
#config = tf.ConfigProto()
#config.gpu_options.allow_growth=True   #dynamic alloc GPU resource
#config.gpu_options.per_process_gpu_memory_fraction = 0.9  #GPU memory threshold 0.3
#session = tf.Session(config=config)

## set session
#K.set_session(session)


def preprocess(x):
    x = np.expand_dims(x, axis=0)

    """
    "mode" option description in preprocess_input
    mode: One of "caffe", "tf" or "torch".
        - caffe: will convert the images from RGB to BGR,
            then will zero-center each color channel with
            respect to the ImageNet dataset,
            without scaling.
        - tf: will scale pixels between -1 and 1,
            sample-wise.
        - torch: will scale pixels between 0 and 1 and then
            will normalize each channel with respect to the
            ImageNet dataset.
    """
    #x = preprocess_input(x, mode='tf')
    x /= 255.0
    x -= 0.5
    x *= 2.0
    return x


def get_model(model_type, include_top=True):
    if model_type == 'shufflenet':
        input_shape = (224, 224, 3)
        model = ShuffleNet(groups=3, weights=None, include_top=include_top)
    elif model_type == 'shufflenet_v2':
        input_shape = (224, 224, 3)
        model = ShuffleNetV2(bottleneck_ratio=1, weights=None, include_top=include_top)
    elif model_type == 'nanonet':
        input_shape = (224, 224, 3)
        model = NanoNet(weights=None, include_top=include_top)
    elif model_type == 'darknet53':
        input_shape = (224, 224, 3)
        model = DarkNet53(weights=None, include_top=include_top)
    elif model_type == 'cspdarknet53':
        input_shape = (224, 224, 3)
        model = CSPDarkNet53(weights=None, include_top=include_top)
    else:
        raise ValueError('Unsupported model type')
    return model, input_shape[:2]


def get_optimizer(optim_type, learning_rate):
    if optim_type == 'sgd':
        optimizer = SGD(lr=learning_rate, decay=5e-4, momentum=0.9)
    elif optim_type == 'rmsprop':
        optimizer = RMSprop(lr=learning_rate)
    elif optim_type == 'adam':
        optimizer = Adam(lr=learning_rate, decay=5e-4)
    else:
        raise ValueError('Unsupported optimizer type')
    return optimizer


def train(args, model, input_shape):
    log_dir = 'logs'

    # callbacks for training process
    checkpoint = ModelCheckpoint(os.path.join(log_dir, 'ep{epoch:03d}-val_loss{val_loss:.3f}-val_acc{val_acc:.3f}-val_top_k_categorical_accuracy{val_top_k_categorical_accuracy:.3f}.h5'),
        monitor='val_acc',
        mode='max',
        verbose=1,
        save_weights_only=False,
        save_best_only=True,
        period=1)
    logging = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=False, write_grads=False, write_images=False, update_freq='batch')
    terminate_on_nan = TerminateOnNaN()
    learn_rates = [0.05, 0.01, 0.005, 0.001, 0.0005]
    lr_scheduler = LearningRateScheduler(lambda epoch: learn_rates[epoch // 30])

    # data generator
    train_datagen = ImageDataGenerator(preprocessing_function=preprocess,
                                       zoom_range=0.25,
                                       width_shift_range=0.05,
                                       height_shift_range=0.05,
                                       brightness_range=0.05,
                                       #rotation_range=30,
                                       #shear_range=0.2,
                                       #channel_shift_range=0.1,
                                       #rescale=1./255,
                                       #vertical_flip=True,
                                       horizontal_flip=True)

    test_datagen = ImageDataGenerator(preprocessing_function=preprocess)

    train_generator = train_datagen.flow_from_directory(
            args.train_data_path,
            target_size=input_shape,
            batch_size=args.batch_size)

    test_generator = test_datagen.flow_from_directory(
            args.val_data_path,
            target_size=input_shape,
            batch_size=args.batch_size)

    # get optimizer
    optimizer = get_optimizer(args.optim_type, args.learning_rate)

    # start training
    model.compile(
              optimizer=optimizer,
              metrics=['accuracy', 'top_k_categorical_accuracy'],
              loss='categorical_crossentropy')

    print('Train on {} samples, val on {} samples, with batch size {}.'.format(train_generator.samples, test_generator.samples, args.batch_size))
    model.fit_generator(
            train_generator,
            steps_per_epoch=train_generator.samples // args.batch_size,
            epochs=args.total_epoch,
            workers=cpu_count()-1,  #Try to parallized feeding image data but leave one cpu core idle
            initial_epoch=args.init_epoch,
            use_multiprocessing=True,
            max_queue_size=10,
            validation_data=test_generator,
            validation_steps=test_generator.samples // args.batch_size,
            callbacks=[logging, checkpoint, lr_scheduler, terminate_on_nan])

    # Finally store model
    model.save(os.path.join(log_dir, 'trained_final.h5'))



def verify_with_image(model, input_shape):
    from tensorflow.keras.applications.resnet50 import decode_predictions
    from PIL import Image
    while True:
        img_file = input('Input image filename:')
        try:
            img = Image.open(img_file)
            resized_img = img.resize(input_shape, Image.BICUBIC)
        except:
            print('Open Error! Try again!')
            continue
        else:
            img_array = np.asarray(resized_img).astype('float32')
            x = preprocess(img_array)
            preds = model.predict(x)
            print('Predict result:', decode_predictions(preds))
            img.show()


def main(args):
    include_top = True
    if args.dump_headless:
        include_top = False

    # prepare model
    model, input_shape = get_model(args.model_type, include_top=include_top)
    if args.weights_path:
        model.load_weights(args.weights_path, by_name=True)
    # support multi-gpu training
    if args.gpu_num >= 2:
        model = multi_gpu_model(model, gpus=args.gpu_num)
    model.summary()

    if args.verify_with_image:
        K.set_learning_phase(0)
        verify_with_image(model, input_shape)
    elif args.dump_headless:
        K.set_learning_phase(0)
        model.save(args.output_model_file)
        print('export headless model to %s' % str(args.output_model_file))
    else:
        train(args, model, input_shape)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, required=False, default='shufflenet_v2',
        help='backbone model type: shufflenet/shufflenet_v2/nanonet/darknet53/cspdarknet53, default=shufflenet_v2')
    parser.add_argument('--train_data_path', type=str,# required=True,
        help='path to Imagenet train data')
    parser.add_argument('--val_data_path', type=str,# required=True,
        help='path to Imagenet validation dataset')
    parser.add_argument('--weights_path', type=str,required=False, default=None,
        help = "Pretrained model/weights file for fine tune")
    parser.add_argument('--batch_size', type=int,required=False, default=128,
        help = "batch size for train, default=128")
    parser.add_argument('--optim_type', type=str, required=False, default='sgd',
        help='optimizer type: sgd/rmsprop/adam, default=sgd')
    parser.add_argument('--learning_rate', type=float,required=False, default=.05,
        help = "Initial learning rate, default=0.05")
    parser.add_argument('--init_epoch', type=int,required=False, default=0,
        help = "Initial training epochs for fine tune training, default=0")
    parser.add_argument('--total_epoch', type=int,required=False, default=200,
        help = "Total training epochs, default=200")
    parser.add_argument('--gpu_num', type=int, required=False, default=1,
        help='Number of GPU to use, default=1')
    parser.add_argument('--verify_with_image', default=False, action="store_true",
        help='Verify trained model with image')
    parser.add_argument('--dump_headless', default=False, action="store_true",
        help='Dump out classification model to headless backbone model')
    parser.add_argument('--output_model_file', type=str,
        help='output headless backbone model file')

    args = parser.parse_args()

    main(args)
