#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
from PIL import Image
import os, sys, argparse
import numpy as np
import MNN

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
from yolo3.postprocess_np import yolo3_head, yolo3_handle_predictions, yolo3_adjust_boxes
from yolo2.postprocess_np import yolo2_head
from yolo3.data import preprocess_image
from yolo3.utils import get_classes, get_anchors, get_colors, draw_boxes


def validate_yolo_model_mnn(model_path, image_file, anchors, class_names, loop_count):
    interpreter = MNN.Interpreter(model_path)
    session = interpreter.createSession()

    # TODO: currently MNN python API only support getting input/output tensor by default or
    # by name. so we need to hardcode the output tensor names here to get them from model
    if len(anchors) == 6:
        output_tensor_names = ['conv2d_1/Conv2D', 'conv2d_3/Conv2D']
    elif len(anchors) == 9:
        output_tensor_names = ['conv2d_3/Conv2D', 'conv2d_8/Conv2D', 'conv2d_13/Conv2D']
    elif len(anchors) == 5:
        # YOLOv2 use 5 anchors and have only 1 prediction
        output_tensor_names = ['predict_conv/Conv2D']
    else:
        raise ValueError('invalid anchor number')

    # assume only 1 input tensor for image
    input_tensor = interpreter.getSessionInput(session)
    # get input shape
    input_shape = input_tensor.getShape()
    if input_tensor.getDimensionType() == MNN.Tensor_DimensionType_Tensorflow:
        batch, height, width, channel = input_shape
    elif input_tensor.getDimensionType() == MNN.Tensor_DimensionType_Caffe:
        batch, channel, height, width = input_shape
    else:
        # should be MNN.Tensor_DimensionType_Caffe_C4, unsupported now
        raise ValueError('unsupported input tensor dimension type')

    # prepare input image
    img = Image.open(image_file)
    image = np.array(img, dtype='uint8')
    image_data = preprocess_image(img, (height, width))
    image_shape = img.size

    # use a temp tensor to copy data
    tmp_input = MNN.Tensor(input_shape, input_tensor.getDataType(),\
                    image_data, input_tensor.getDimensionType())

    # predict once first to bypass the model building time
    input_tensor.copyFrom(tmp_input)
    interpreter.runSession(session)

    start = time.time()
    for i in range(loop_count):
        input_tensor.copyFrom(tmp_input)
        interpreter.runSession(session)
    end = time.time()
    print("Average Inference time: {:.8f}ms".format((end - start) * 1000 /loop_count))

    out_list = []
    for output_tensor_name in output_tensor_names:
        output_tensor = interpreter.getSessionOutput(session, output_tensor_name)
        output_shape = output_tensor.getShape()

        assert output_tensor.getDataType() == MNN.Halide_Type_Float

        # copy output tensor to host, for further postprocess
        tmp_output = MNN.Tensor(output_shape, output_tensor.getDataType(),\
                    np.zeros(output_shape, dtype=float), output_tensor.getDimensionType())

        output_tensor.copyToHostTensor(tmp_output)
        #tmp_output.printTensorData()

        output_data = np.array(tmp_output.getData(), dtype=float).reshape(output_shape)
        # our postprocess code based on TF channel last format, so if the output format
        # doesn't match, we need to transpose
        if output_tensor.getDimensionType() == MNN.Tensor_DimensionType_Caffe:
            output_data = output_data.transpose((0,2,3,1))
        elif output_tensor.getDimensionType() == MNN.Tensor_DimensionType_Caffe_C4:
            raise ValueError('unsupported output tensor dimension type')

        out_list.append(output_data)

    start = time.time()
    if len(out_list) == 1:
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

    validate_yolo_model_mnn(args.model_path, args.image_file, anchors, class_names, args.loop_count)


if __name__ == '__main__':
    main()
