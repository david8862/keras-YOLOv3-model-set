#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# train backbone network with imagenet dataset
#

import os, sys, argparse
import numpy as np
import cv2
from multiprocessing import cpu_count

import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler, TerminateOnNaN
#from tensorflow.keras.utils import multi_gpu_model
import tensorflow as tf

from data_utils import normalize_image, random_grayscale, random_chroma, random_contrast, random_sharpness

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
from shufflenet import ShuffleNet
from shufflenet_v2 import ShuffleNetV2
from mobilevit import MobileViT_S, MobileViT_XS, MobileViT_XXS

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', '..'))
from yolo3.models.yolo3_nano import NanoNet
from yolo3.models.yolo3_darknet import DarkNet53
from yolo4.models.yolo4_darknet import CSPDarkNet53

from common.utils import optimize_tf_gpu
from common.model_utils import get_optimizer
from common.callbacks import CheckpointCleanCallBack


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#import tensorflow as tf
#config = tf.ConfigProto()
#config.gpu_options.allow_growth=True   #dynamic alloc GPU resource
#config.gpu_options.per_process_gpu_memory_fraction = 0.9  #GPU memory threshold 0.3
#session = tf.Session(config=config)

## set session
#K.set_session(session)

optimize_tf_gpu(tf, K)


def preprocess(image):
    # random adjust color level
    image = random_chroma(image)

    # random adjust contrast
    image = random_contrast(image)

    # random adjust sharpness
    image = random_sharpness(image)

    # random convert image to grayscale
    image = random_grayscale(image)

    # normalize image
    image = normalize_image(image)

    return image


def get_model(model_type, include_top=True):
    if model_type == 'shufflenet':
        input_shape = (224, 224, 3)
        model = ShuffleNet(input_shape=input_shape, groups=3, weights=None, include_top=include_top)
    elif model_type == 'shufflenet_v2':
        input_shape = (224, 224, 3)
        model = ShuffleNetV2(input_shape=input_shape, bottleneck_ratio=1, weights=None, include_top=include_top)
    elif model_type == 'nanonet':
        input_shape = (224, 224, 3)
        model = NanoNet(input_shape=input_shape, weights=None, include_top=include_top)
    elif model_type == 'darknet53':
        input_shape = (224, 224, 3)
        model = DarkNet53(input_shape=input_shape, weights=None, include_top=include_top)
    elif model_type == 'cspdarknet53':
        input_shape = (224, 224, 3)
        model = CSPDarkNet53(input_shape=input_shape, weights=None, include_top=include_top)
    elif model_type == 'mobilevit_s':
        input_shape = (256, 256, 3)
        model = MobileViT_S(input_shape=input_shape, weights=None, include_top=include_top)
    elif model_type == 'mobilevit_xs':
        input_shape = (256, 256, 3)
        model = MobileViT_XS(input_shape=input_shape, weights=None, include_top=include_top)
    elif model_type == 'mobilevit_xxs':
        input_shape = (256, 256, 3)
        model = MobileViT_XXS(input_shape=input_shape, weights=None, include_top=include_top)
    else:
        raise ValueError('Unsupported model type')
    return model, input_shape[:2]


def train(args, model, input_shape, strategy):
    log_dir = os.path.join('logs', '000')

    # callbacks for training process
    checkpoint = ModelCheckpoint(os.path.join(log_dir, 'ep{epoch:03d}-val_loss{val_loss:.3f}-val_accuracy{val_accuracy:.3f}-val_top_k_categorical_accuracy{val_top_k_categorical_accuracy:.3f}.h5'),
        monitor='val_accuracy',
        mode='max',
        verbose=1,
        save_weights_only=False,
        save_best_only=True,
        period=1)
    logging = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=False, write_grads=False, write_images=False, update_freq='batch')
    terminate_on_nan = TerminateOnNaN()
    learn_rates = [0.05, 0.01, 0.005, 0.001, 0.0005]
    lr_scheduler = LearningRateScheduler(lambda epoch: learn_rates[epoch // 30])
    checkpoint_clean = CheckpointCleanCallBack(log_dir, max_val_keep=3)

    callbacks=[logging, checkpoint, lr_scheduler, terminate_on_nan, checkpoint_clean]

    # data generator
    train_datagen = ImageDataGenerator(preprocessing_function=preprocess,
                                       #featurewise_center=False,
                                       #samplewise_center=False,
                                       #featurewise_std_normalization=False,
                                       #samplewise_std_normalization=False,
                                       #zca_whitening=False,
                                       #zca_epsilon=1e-06,
                                       zoom_range=0.25,
                                       brightness_range=[0.5,1.5],
                                       channel_shift_range=0.1,
                                       shear_range=0.2,
                                       rotation_range=30,
                                       width_shift_range=0.05,
                                       height_shift_range=0.05,
                                       vertical_flip=True,
                                       horizontal_flip=True,
                                       #rescale=1./255,
                                       #validation_split=0.1,
                                       fill_mode='constant',
                                       cval=0.,
                                       data_format=None,
                                       dtype=None)

    test_datagen = ImageDataGenerator(preprocessing_function=preprocess)

    train_generator = train_datagen.flow_from_directory(
            args.train_data_path,
            target_size=input_shape,
            batch_size=args.batch_size,
            color_mode='rgb',
            classes=None,
            class_mode='categorical',
            shuffle=True,
            #save_to_dir='check',
            #save_prefix='augmented_',
            #save_format='jpg',
            interpolation='nearest')

    test_generator = test_datagen.flow_from_directory(
            args.val_data_path,
            target_size=input_shape,
            batch_size=args.batch_size,
            color_mode='rgb',
            classes=None,
            class_mode='categorical',
            shuffle=True,
            #save_to_dir='check',
            #save_prefix='augmented_',
            #save_format='jpg',
            interpolation='nearest')

    # get optimizer
    if args.decay_type:
        callbacks.remove(lr_scheduler)
    steps_per_epoch = max(1, train_generator.samples//args.batch_size)
    decay_steps = steps_per_epoch * (args.total_epoch - args.init_epoch)
    optimizer = get_optimizer(args.optimizer, args.learning_rate, average_type=None, decay_type=args.decay_type, decay_steps=decay_steps)

    # get loss
    losses = CategoricalCrossentropy(label_smoothing=args.label_smoothing)

    # model compile
    if strategy:
        with strategy.scope():
            model.compile(
                      optimizer=optimizer,
                      metrics=['accuracy', 'top_k_categorical_accuracy'],
                      loss=losses)
    else:
        model.compile(
                  optimizer=optimizer,
                  metrics=['accuracy', 'top_k_categorical_accuracy'],
                  loss=losses)

    # start training
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
            callbacks=callbacks)

    # Finally store model
    model.save(os.path.join(log_dir, 'trained_final.h5'))



def evaluate_model(args, model, input_shape):
    # eval data generator
    eval_datagen = ImageDataGenerator(preprocessing_function=preprocess)
    eval_generator = eval_datagen.flow_from_directory(
            args.val_data_path,
            target_size=input_shape,
            batch_size=args.batch_size)

    # get optimizer
    optimizer = get_optimizer(args.optimizer, args.learning_rate, average_type=None, decay_type=None)

    # start evaluate
    model.compile(
              optimizer=optimizer,
              metrics=['accuracy', 'top_k_categorical_accuracy'],
              loss='categorical_crossentropy')

    print('Evaluate on {} samples, with batch size {}.'.format(eval_generator.samples, args.batch_size))
    scores = model.evaluate_generator(
            eval_generator,
            steps=eval_generator.samples // args.batch_size,
            max_queue_size=10,
            workers=1,
            use_multiprocessing=False,
            verbose=1)

    print('Evaluate loss:', scores[0])
    print('Top-1 accuracy:', scores[1])
    print('Top-k accuracy:', scores[2])


def verify_with_image(model, input_shape):
    from tensorflow.keras.applications.resnet50 import decode_predictions
    from PIL import Image
    while True:
        img_file = input('Input image filename:')
        try:
            img = Image.open(img_file).convert('RGB')
            resized_img = img.resize(input_shape, Image.BICUBIC)
        except:
            print('Open Error! Try again!')
            continue
        else:
            img_array = np.asarray(resized_img).astype('float32')
            x = normalize_image(img_array)
            preds = model.predict(np.expand_dims(x, 0))

            result = decode_predictions(preds)
            print('Predict result:', result)

            # show predict result on origin image
            img_array = np.asarray(img)
            cv2.putText(img_array, '{name}:{conf:.3f}'.format(name=result[0][0][1], conf=float(result[0][0][2])),
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1,
                        color=(255, 0, 0),
                        thickness=2,
                        lineType=cv2.LINE_AA)
            Image.fromarray(img_array).show()


def main(args):
    include_top = True
    if args.dump_headless:
        include_top = False

    # support multi-gpu training
    if args.gpu_num >= 2:
        # devices_list=["/gpu:0", "/gpu:1"]
        devices_list=["/gpu:{}".format(n) for n in range(args.gpu_num)]
        strategy = tf.distribute.MirroredStrategy(devices=devices_list)
        print ('Number of devices: {}'.format(strategy.num_replicas_in_sync))
        with strategy.scope():
            model, input_shape = get_model(args.model_type, include_top=include_top)
    else:
        # get normal train model
        model, input_shape = get_model(args.model_type, include_top=include_top)
        strategy = None

    if args.weights_path:
        model.load_weights(args.weights_path, by_name=True)
    model.summary()

    if args.evaluate:
        K.set_learning_phase(0)
        evaluate_model(args, model, input_shape)
    elif args.verify_with_image:
        K.set_learning_phase(0)
        verify_with_image(model, input_shape)
    elif args.dump_headless:
        K.set_learning_phase(0)
        model.save(args.output_model_file)
        print('export headless model to %s' % str(args.output_model_file))
    else:
        train(args, model, input_shape, strategy)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Model definition options
    parser.add_argument('--model_type', type=str, required=True, choices=['shufflenet', 'shufflenet_v2', 'nanonet', 'darknet53', 'cspdarknet53', 'mobilevit_s', 'mobilevit_xs', 'mobilevit_xxs'],
        help='backbone model type')
    parser.add_argument('--weights_path', type=str, required=False, default=None,
        help = "Pretrained model/weights file for fine tune")

    # Data options
    parser.add_argument('--train_data_path', type=str, #required=True,
        help='path to Imagenet train data')
    parser.add_argument('--val_data_path', type=str, #required=True,
        help='path to Imagenet validation dataset')

    # Training options
    parser.add_argument('--batch_size', type=int, required=False, default=128,
        help = "batch size for train, default=%(default)s")
    parser.add_argument('--optimizer', type=str, required=False, default='sgd', choices=['adam', 'rmsprop', 'sgd'],
        help = "optimizer for training (adam/rmsprop/sgd), default=%(default)s")
    parser.add_argument('--learning_rate', type=float,required=False, default=.01,
        help = "Initial learning rate, default=%(default)s")
    parser.add_argument('--decay_type', type=str, required=False, default=None, choices=[None, 'cosine', 'exponential', 'polynomial', 'piecewise_constant'],
        help = "Learning rate decay type, default=%(default)s")
    parser.add_argument('--label_smoothing', type=float, required=False, default=0,
        help = "Label smoothing factor (between 0 and 1) for classification loss, default=%(default)s")
    parser.add_argument('--init_epoch', type=int,required=False, default=0,
        help = "Initial training epochs for fine tune training, default=%(default)s")
    parser.add_argument('--total_epoch', type=int,required=False, default=200,
        help = "Total training epochs, default=%(default)s")
    parser.add_argument('--gpu_num', type=int, required=False, default=1,
        help='Number of GPU to use, default=%(default)s')

    # Evaluation options
    parser.add_argument('--evaluate', default=False, action="store_true",
        help='Evaluate a trained model with validation dataset')
    parser.add_argument('--verify_with_image', default=False, action="store_true",
        help='Verify trained model with image')
    parser.add_argument('--dump_headless', default=False, action="store_true",
        help='Dump out classification model to headless backbone model')
    parser.add_argument('--output_model_file', type=str,
        help='output headless backbone model file')

    args = parser.parse_args()

    main(args)
