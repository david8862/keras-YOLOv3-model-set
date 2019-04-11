#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run a YOLO_v3 style detection model on test images.
"""

import colorsys
import os, argparse
import cv2
import time
from timeit import default_timer as timer
import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input
from PIL import Image, ImageFont, ImageDraw

from yolo3.model_Mobilenet import yolo_eval, yolo_mobilenet_body, tiny_yolo_mobilenet_body, custom_yolo_mobilenet_body
from yolo3.utils import letterbox_image
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from tensorflow.keras.utils import multi_gpu_model
gpu_num=1

class YOLO(object):
    _defaults = {
        "model_path": 'model_data/yolov3-tiny.h5',
        "anchors_path": 'model_data/tiny_yolo_anchors.txt',
        "classes_path": 'model_data/coco_classes.txt',
        "score" : 0.3,
        "iou" : 0.45,
        "model_image_size" : (416, 416),
        "gpu_num" : 1,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        '''to generate the bounding boxes'''
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            #self.yolo_model = tiny_yolo_mobilenet_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                #if is_tiny_version else yolo_mobilenet_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model = tiny_yolo_mobilenet_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                if is_tiny_version else custom_yolo_mobilenet_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
            self.yolo_model.summary()
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        if gpu_num>=2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        # default arg
        # self.yolo_model->'model_data/yolo.h5'
        # self.anchors->'model_data/yolo_anchors.txt'-> 9 scales for anchors
        return boxes, scores, classes

    def detect_image(self, image):
        start = timer()
        rects = []
        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        # print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        # tf.Session.run(fetches, feed_dict=None)
        # Runs the operations and evaluates the tensors in fetches.
        #
        # Args:
        # fetches: A single graph element, or a list of graph elements(described above).
        #
        # feed_dict: A dictionary that maps graph elements to values(described above).
        #
        # Returns:Either a single value if fetches is a single graph element, or a
        # list of values if fetches is a list(described above).
        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                    size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 500

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            y1, x1, y2, x2 = box
            y1 = max(0, np.floor(y1 + 0.5).astype('float32'))
            x1 = max(0, np.floor(x1 + 0.5).astype('float32'))
            y2 = min(image.size[1], np.floor(y2 + 0.5).astype('float32'))
            x2 = min(image.size[0], np.floor(x2 + 0.5).astype('float32'))
            # print(label, (x1, y1), (x2, y2))
            bbox = dict([("score",str(score)),("x1",str(x1)),("y1", str(y1)),("x2", str(x2)),("y2", str(y2))])
            rects.append(bbox)

            if y1 - label_size[1] >= 0:
                text_origin = np.array([x1, y1 - label_size[1]])
            else:
                text_origin = np.array([x1, y1 + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle(
                    [x1 + i, y1 + i, x2 - i, y2 - i],
                    outline=self.colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw

        end = timer()
        print(str(end - start))
        return image

    def detect_image2(self, image):
        start = time.time()

        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
        image_data = np.array(image, dtype='uint8')

        from predict import predict, draw_boxes

        out_boxes, out_classes, out_scores = predict(self.yolo_model, image_data, len(self.class_names), self.model_image_size)

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))
        image_data = draw_boxes(image_data, out_boxes, out_classes, out_scores, self.class_names, self.colors)

        end = time.time()
        print("Inference time: {:.2f}s".format(end - start))
        return Image.fromarray(image_data)

    def predict(self, image):
        start = time.time()

        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
        image_data = np.array(image, dtype='uint8')

        from predict import predict

        out_boxes, out_classes, out_scores = predict(self.yolo_model, image_data, len(self.class_names), self.model_image_size)

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))
        end = time.time()
        print("Inference time: {:.2f}s".format(end - start))
        return out_boxes, out_classes, out_scores

    def dump_model_file(self, output_model_file):
        self.yolo_model.save(output_model_file)

    def close_session(self):
        self.sess.close()


def detect_video(yolo, video_path, output_path=""):
    import cv2
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_FourCC    = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps       = vid.get(cv2.CAP_PROP_FPS)
    video_size      = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    isOutput = True if output_path != "" else False
    if isOutput:
        print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
        out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
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
    yolo.close_session()


def detect_img(yolo):
    while True:
        img = input('Input image filename:')
        try:
            image = Image.open(img)
        except:
            print('Open Error! Try again!')
            continue
        else:
            r_image = yolo.detect_image2(image)
            r_image.show()
    yolo.close_session()

def detect_test_draw(yolo,json_name,test_pic):
    import cv2
    import json

    data_dst = 'dataset/brainwash/'
    with open(json_name) as load_f:
        load_dict = json.load(load_f)
        for pic in load_dict:
            picname = pic['image_path']
            root,name = os.path.split(picname)
            print(name)
            image = Image.open(data_dst + picname)
            rects = yolo.detect_image(image)
            frame = cv2.imread(data_dst+picname)
            for rect in rects:
                score, x1, y1, x2, y2 = float(rect['score']),int(float(rect['x1'])),int(float(rect['y1'])),int(float(rect['x2'])),int(float(rect['y2']))
                cv2.rectangle(frame,(x1,y1),(x2,y2),(255,255,255),1)
            cv2.imwrite(test_pic+name,frame)
    yolo.close_session()

def detect_test(yolo,json_name,test_out_json = 'caltech_new_result_0.001.json',data_dst = '../caltech_ped/caltech-pedestrian-dataset-converter/'):
    import json
    import time

    #
    with open(json_name) as load_f:
        load_dict = json.load(load_f)
    count = 0
    json_images=[]
    with open(test_out_json,'w') as outfile:
        time_start = time.time()

        for pic in load_dict:
            # root, filename = os.path.split(pic['image_path'])
            # name = filename.split('.')[0]
            # set_id, v_id, frame_id = name.split('_')
            # frame_id = int(frame_id)
            #
            # if frame_id % 30 == 0 and frame_id != 0:
                picname = pic['image_path'][2:]
                count +=1
                print(picname)
                image = Image.open(data_dst + picname)
                rects = yolo.detect_image(image)
                json_image = dict([("image_path", picname), ("rects", rects)])
                json_images.append(json_image)

        time_end = time.time()
        duration = time_end - time_start
        print('totally cost', duration)
        print('{} pictures , average time {}'.format(count,duration/count))
        str = json.dumps(json_images,indent=4)
        outfile.write(str)
        outfile.close()
    yolo.close_session()

def car_detect(yolo,mainFolder = '/home/wenwen/Viewnyx/FrameImages/'):
    import json
    fold_list = range(1, 15)

    for i in fold_list:
        foldname = mainFolder+'video'+str(i)
        list = os.listdir(foldname)  # 列出文件夹下所有的目录与文件
        json_all = {}
        json_f = open('car/'+'annotation_{}_YOLOv3.json'.format('video'+str(i)),'w')
        for i in range(0, len(list)):
            name,ext = os.path.splitext(list[i])
            if ext=='.jpg':
                print(list[i])
                json_pic = {}
                annotation = []
                image = Image.open(foldname+'/'+list[i])
                rects = yolo.detect_image(image)
                for rect in rects:
                    score, x1, y1, x2, y2 = float(rect['score']), int(float(rect['x1'])), int(float(rect['y1'])), int(
                        float(rect['x2'])), int(float(rect['y2']))
                    bbox = {"category": "sideways",
                            "id": 0,
                            "shape": ["Box",1],
                            "label": "car",
                            "x":x1,
                            "y":y1,
                            "width":x2-x1,
                            "height":y2-y1,
                            "score":score}
                    annotation.append(bbox)
                json_pic["annotations"]=annotation
                json_pic["height"] = 480
                json_pic["name"] =  list[i]
                json_pic["width"] =  640
                json_all[list[i]] = json_pic
        json_f.write(json.dumps(json_all,indent=4))
        json_f.close()
    yolo.close_session()


if __name__ == '__main__':
    # class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument(
        '--model_path', type=str,
        help='path to model weight file, default ' + YOLO.get_defaults("model_path")
    )

    parser.add_argument(
        '--anchors_path', type=str,
        help='path to anchor definitions, default ' + YOLO.get_defaults("anchors_path")
    )

    parser.add_argument(
        '--classes_path', type=str,
        help='path to class definitions, default ' + YOLO.get_defaults("classes_path")
    )

    parser.add_argument(
        '--gpu_num', type=int,
        help='Number of GPU to use, default ' + str(YOLO.get_defaults("gpu_num"))
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

    FLAGS = parser.parse_args()

    if FLAGS.dump_model:
        """
        Dump out training model to inference model
        """
        if not FLAGS.output_model_file:
            raise ValueError('output model file is not specified')

        print('Dumping out training model to inference model')
        YOLO(**vars(FLAGS)).dump_model_file(FLAGS.output_model_file)
    else:
        detect_img(YOLO(**vars(FLAGS)))

