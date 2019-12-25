#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run a YOLOv3/YOLOv2 style detection model on test images.
"""

import colorsys
import os, sys, argparse
import cv2
import time
from timeit import default_timer as timer
import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Lambda
from tensorflow_model_optimization.sparsity import keras as sparsity
from PIL import Image

from yolo3.model import get_yolo3_model, get_yolo3_inference_model#, get_yolo3_prenms_model
from yolo3.postprocess_np import yolo3_postprocess_np
from yolo2.model import get_yolo2_model, get_yolo2_inference_model
from yolo2.postprocess_np import yolo2_postprocess_np
from common.data_utils import preprocess_image
from common.utils import get_classes, get_anchors, get_colors, draw_boxes, touchdir

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from tensorflow.keras.utils import multi_gpu_model

#tf.enable_eager_execution()

default_config = {
        "model_type": 'tiny_yolo3_darknet',
        "model_path": 'weights/yolov3-tiny.h5',
        "pruning_model": False,
        "anchors_path": 'configs/tiny_yolo3_anchors.txt',
        "classes_path": 'configs/coco_classes.txt',
        "score" : 0.1,
        "iou" : 0.4,
        "model_image_size" : (416, 416),
        "gpu_num" : 1,
    }


class YOLO_np(object):
    _defaults = default_config

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        super(YOLO_np, self).__init__()
        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides
        self.class_names = get_classes(self.classes_path)
        self.anchors = get_anchors(self.anchors_path)
        self.colors = get_colors(self.class_names)
        K.set_learning_phase(0)
        self.yolo_model = self._generate_model()

    def _generate_model(self):
        '''to generate the bounding boxes'''
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        #YOLOv3 model has 9 anchors and 3 feature layers but
        #Tiny YOLOv3 model has 6 anchors and 2 feature layers,
        #so we can calculate feature layers number to get model type
        num_feature_layers = num_anchors//3

        try:
            if num_anchors == 5:
                # YOLOv2 use 5 anchors
                yolo_model, _ = get_yolo2_model(self.model_type, num_anchors, num_classes, input_shape=self.model_image_size + (3,), model_pruning=self.pruning_model)
            else:
                yolo_model, _ = get_yolo3_model(self.model_type, num_feature_layers, num_anchors, num_classes, input_shape=self.model_image_size + (3,), model_pruning=self.pruning_model)
            yolo_model.load_weights(model_path) # make sure model, anchors and classes match
            if self.pruning_model:
                yolo_model = sparsity.strip_pruning(yolo_model)
            yolo_model.summary()
        except Exception as e:
            print(repr(e))
            assert yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'
        print('{} model, anchors, and classes loaded.'.format(model_path))
        if self.gpu_num>=2:
            yolo_model = multi_gpu_model(yolo_model, gpus=self.gpu_num)

        return yolo_model


    def detect_image(self, image):
        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'

        image_data = preprocess_image(image, self.model_image_size)
        image_shape = image.size

        start = time.time()
        out_boxes, out_classes, out_scores = self.predict(image_data, image_shape)
        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))
        end = time.time()
        print("Inference time: {:.8f}s".format(end - start))

        #draw result on input image
        image_array = np.array(image, dtype='uint8')
        image_array = draw_boxes(image_array, out_boxes, out_classes, out_scores, self.class_names, self.colors)
        return Image.fromarray(image_array)


    def predict(self, image_data, image_shape):
        num_anchors = len(self.anchors)
        if num_anchors == 5:
            # YOLOv2 use 5 anchors
            out_boxes, out_classes, out_scores = yolo2_postprocess_np(self.yolo_model.predict(image_data), image_shape, self.anchors, len(self.class_names), self.model_image_size, max_boxes=100)
        else:
            out_boxes, out_classes, out_scores = yolo3_postprocess_np(self.yolo_model.predict(image_data), image_shape, self.anchors, len(self.class_names), self.model_image_size, max_boxes=100)
        return out_boxes, out_classes, out_scores


    def dump_model_file(self, output_model_file):
        self.yolo_model.save(output_model_file)



class YOLO(object):
    _defaults = default_config

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        super(YOLO, self).__init__()
        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides
        self.class_names = get_classes(self.classes_path)
        self.anchors = get_anchors(self.anchors_path)
        self.colors = get_colors(self.class_names)
        K.set_learning_phase(0)
        self.inference_model = self._generate_model()

    def _generate_model(self):
        '''to generate the bounding boxes'''
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        #YOLOv3 model has 9 anchors and 3 feature layers but
        #Tiny YOLOv3 model has 6 anchors and 2 feature layers,
        #so we can calculate feature layers number to get model type
        num_feature_layers = num_anchors//3

        if num_anchors == 5:
            # YOLOv2 use 5 anchors
            inference_model = get_yolo2_inference_model(self.model_type, self.anchors, num_classes, weights_path=model_path, input_shape=self.model_image_size + (3,), confidence=0.1)
        else:
            inference_model = get_yolo3_inference_model(self.model_type, self.anchors, num_classes, weights_path=model_path, input_shape=self.model_image_size + (3,), confidence=0.1)

        inference_model.summary()
        return inference_model

    def predict(self, image_data, image_shape):
        out_boxes, out_scores, out_classes = self.inference_model.predict([image_data, image_shape])

        out_boxes = out_boxes[0]
        out_scores = out_scores[0]
        out_classes = out_classes[0]

        out_boxes = out_boxes.astype(np.int32)
        out_classes = out_classes.astype(np.int32)
        return out_boxes, out_classes, out_scores

    def detect_image(self, image):
        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'

        image_data = preprocess_image(image, self.model_image_size)
        image_shape = np.array([image.size[0], image.size[1]])
        image_shape = np.expand_dims(image_shape, 0)

        start = time.time()
        out_boxes, out_classes, out_scores = self.predict(image_data, image_shape)
        end = time.time()
        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))
        print("Inference time: {:.8f}s".format(end - start))

        #draw result on input image
        image_array = np.array(image, dtype='uint8')
        image_array = draw_boxes(image_array, out_boxes, out_classes, out_scores, self.class_names, self.colors)
        return Image.fromarray(image_array)

    def dump_model_file(self, output_model_file):
        self.inference_model.save(output_model_file)

    def dump_saved_model(self, saved_model_path):
        model = self.inference_model
        touchdir(saved_model_path)

        tf.keras.experimental.export_saved_model(model, saved_model_path)
        print('export inference model to %s' % str(saved_model_path))


#class YOLO_prenms(object):
    #_defaults = default_config

    #@classmethod
    #def get_defaults(cls, n):
        #if n in cls._defaults:
            #return cls._defaults[n]
        #else:
            #return "Unrecognized attribute name '" + n + "'"

    #def __init__(self, **kwargs):
        #super(YOLO_prenms, self).__init__()
        #self.__dict__.update(self._defaults) # set up default values
        #self.__dict__.update(kwargs) # and update with user overrides
        #self.class_names = get_classes(self.classes_path)
        #self.anchors = get_anchors(self.anchors_path)
        #self.colors = get_colors(self.class_names)
        #K.set_learning_phase(0)
        #self.prenms_model = self._generate_model()

    #def _generate_model(self):
        #'''to generate the bounding boxes'''
        #model_path = os.path.expanduser(self.model_path)
        #assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        ## Load model, or construct model and load weights.
        #num_anchors = len(self.anchors)
        #num_classes = len(self.class_names)
        ##YOLOv3 model has 9 anchors and 3 feature layers but
        ##Tiny YOLOv3 model has 6 anchors and 2 feature layers,
        ##so we can calculate feature layers number to get model type
        #num_feature_layers = num_anchors//3

        #prenms_model = get_yolo3_prenms_model(self.model_type, self.anchors, num_classes, weights_path=model_path, input_shape=self.model_image_size + (3,))

        #return prenms_model


    #def dump_model_file(self, output_model_file):
        #self.prenms_model.save(output_model_file)

    #def dump_saved_model(self, saved_model_path):
        #model = self.prenms_model
        #touchdir(saved_model_path)

        #tf.keras.experimental.export_saved_model(model, saved_model_path)
        #print('export inference model to %s' % str(saved_model_path))


def detect_video(yolo, video_path, output_path=""):
    import cv2
    vid = cv2.VideoCapture(0 if video_path == '0' else video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_FourCC    = cv2.VideoWriter_fourcc(*'XVID') if video_path == '0' else int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps       = vid.get(cv2.CAP_PROP_FPS)
    video_size      = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    isOutput = True if output_path != "" else False
    if isOutput:
        print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
        out = cv2.VideoWriter(output_path, video_FourCC, (5. if video_path == '0' else video_fps), video_size)
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    while True:
        return_value, frame = vid.read()
        image = Image.fromarray(frame)
        image = yolo.detect_image(image)
        result = np.asarray(image)
        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0
        cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(255, 0, 0), thickness=2)
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.imshow("result", result)
        if isOutput:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # Release everything if job is finished
    vid.release()
    if isOutput:
        out.release()
    cv2.destroyAllWindows()


def detect_img(yolo):
    while True:
        img = input('Input image filename:')
        try:
            image = Image.open(img)
        except:
            print('Open Error! Try again!')
            continue
        else:
            r_image = yolo.detect_image(image)
            r_image.show()


if __name__ == '__main__':
    # class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS, description='demo or dump out YOLO h5 model')
    '''
    Command line options
    '''
    parser.add_argument(
        '--model_type', type=str,
        help='YOLO model type: yolo3_mobilenet_lite/tiny_yolo3_mobilenet/yolo3_darknet/..., default ' + YOLO.get_defaults("model_type")
    )

    parser.add_argument(
        '--weights_path', type=str,
        help='path to model weight file, default ' + YOLO.get_defaults("model_path")
    )

    parser.add_argument(
        '--pruning_model', default=False, action="store_true",
        help='Whether to be a pruning model/weights file')

    parser.add_argument(
        '--anchors_path', type=str,
        help='path to anchor definitions, default ' + YOLO.get_defaults("anchors_path")
    )

    parser.add_argument(
        '--classes_path', type=str,
        help='path to class definitions, default ' + YOLO.get_defaults("classes_path")
    )

    parser.add_argument(
        '--model_image_size', type=str,
        help='model image input size as <num>x<num>, default ' +
        str(YOLO.get_defaults("model_image_size")[0])+'x'+str(YOLO.get_defaults("model_image_size")[1]),
        default=str(YOLO.get_defaults("model_image_size")[0])+'x'+str(YOLO.get_defaults("model_image_size")[1])
    )

    parser.add_argument(
        '--gpu_num', type=int,
        help='Number of GPU to use, default ' + str(YOLO.get_defaults("gpu_num"))
    )
    parser.add_argument(
        '--image', default=False, action="store_true",
        help='Image detection mode, will ignore all positional arguments'
    )
    '''
    Command line positional arguments -- for video detection mode
    '''
    parser.add_argument(
        "--input", nargs='?', type=str,required=False,default='./path2your_video',
        help = "Video input path"
    )

    parser.add_argument(
        "--output", nargs='?', type=str, default="",
        help = "[Optional] Video output path"
    )
    '''
    Command line positional arguments -- for model dump
    '''
    parser.add_argument(
        '--dump_model', default=False, action="store_true",
        help='Dump out training model to inference model'
    )

    parser.add_argument(
        '--output_model_file', type=str,
        help='output inference model file'
    )

    args = parser.parse_args()
    # param parse
    if args.model_image_size:
        height, width = args.model_image_size.split('x')
        args.model_image_size = (int(height), int(width))

    # get wrapped inference object
    yolo = YOLO_np(**vars(args))

    if args.dump_model:
        """
        Dump out training model to inference model
        """
        if not args.output_model_file:
            raise ValueError('output model file is not specified')

        print('Dumping out training model to inference model')
        yolo.dump_model_file(args.output_model_file)
        sys.exit()

    if args.image:
        """
        Image detection mode, disregard any remaining command line arguments
        """
        print("Image detection mode")
        if "input" in args:
            print(" Ignoring remaining command line arguments: " + args.input + "," + args.output)
        detect_img(yolo)
    elif "input" in args:
        detect_video(yolo, args.input, args.output)
    else:
        print("Must specify at least video_input_path.  See usage with --help.")
