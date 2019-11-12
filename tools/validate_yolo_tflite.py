#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
from PIL import Image
import os, sys, argparse
import numpy as np
from tensorflow.lite.python import interpreter as interpreter_wrapper

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
from yolo3.postprocess_np import yolo3_head, yolo3_handle_predictions, yolo3_adjust_boxes
from yolo2.postprocess_np import yolo2_head
from common.data_utils import preprocess_image
from common.utils import get_classes, get_anchors, get_colors, draw_boxes


def validate_yolo_model_tflite(model_path, image_file, anchors, class_names, loop_count):
    interpreter = interpreter_wrapper.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    #print(input_details)
    #print(output_details)

    # check the type of the input tensor
    if input_details[0]['dtype'] == np.float32:
        floating_model = True

    img = Image.open(image_file)
    image = np.array(img, dtype='uint8')

    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    image_data = preprocess_image(img, (height, width))
    image_shape = img.size

    # predict once first to bypass the model building time
    interpreter.set_tensor(input_details[0]['index'], image_data)
    interpreter.invoke()

    start = time.time()
    for i in range(loop_count):
        interpreter.set_tensor(input_details[0]['index'], image_data)
        interpreter.invoke()
    end = time.time()
    print("Average Inference time: {:.8f}ms".format((end - start) * 1000 /loop_count))

    out_list = []
    for output_detail in output_details:
        output_data = interpreter.get_tensor(output_detail['index'])
        out_list.append(output_data)

    start = time.time()
    if len(anchors) == 5:
        # YOLOv2 use 5 anchors and have only 1 prediction
        assert len(out_list) == 1, 'invalid YOLOv2 prediction number.'
        predictions = yolo2_head(out_list[0], anchors, num_classes=len(class_names), input_dims=(height, width))
    else:
        predictions = yolo3_head(out_list, anchors, num_classes=len(class_names), input_dims=(height, width))

    boxes, classes, scores = yolo3_handle_predictions(predictions, confidence=0.1, iou_threshold=0.4)
    boxes = yolo3_adjust_boxes(boxes, image_shape, (height, width))
    end = time.time()
    print("PostProcess time: {:.8f}ms".format((end - start) * 1000))

    print('Found {} boxes for {}'.format(len(boxes), image_file))

    for box, cls, score in zip(boxes, classes, scores):
        print("Class: {}, Score: {}".format(class_names[cls], score))

    colors = get_colors(class_names)
    image = draw_boxes(image, boxes, classes, scores, class_names, colors)

    Image.fromarray(image).show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', help='model file to predict', type=str, required=True)
    parser.add_argument('--image_file', help='image file to predict', type=str, required=True)
    parser.add_argument('--anchors_path',help='path to anchor definitions', type=str, required=True)
    parser.add_argument('--classes_path', help='path to class definitions, default ../configs/voc_classes.txt', type=str, default='../configs/voc_classes.txt')
    parser.add_argument('--loop_count', help='loop inference for certain times', type=int, default=1)

    args = parser.parse_args()

    # param parse
    anchors = get_anchors(args.anchors_path)
    class_names = get_classes(args.classes_path)

    validate_yolo_model_tflite(args.model_path, args.image_file, anchors, class_names, args.loop_count)


if __name__ == '__main__':
    main()
