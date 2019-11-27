#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
from PIL import Image
import os, sys, argparse
import numpy as np
import MNN
from tensorflow.keras.models import load_model
from tensorflow.lite.python import interpreter as interpreter_wrapper

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
from yolo3.postprocess_np import yolo3_postprocess_np
from yolo2.postprocess_np import yolo2_postprocess_np
from common.data_utils import preprocess_image
from common.utils import get_classes, get_anchors, get_colors, draw_boxes


def validate_yolo_model(model_path, image_file, anchors, class_names, model_image_size, loop_count):
    model = load_model(model_path, compile=False)

    img = Image.open(image_file)
    image = np.array(img, dtype='uint8')
    image_data = preprocess_image(img, model_image_size)
    image_shape = img.size

    # predict once first to bypass the model building time
    model.predict([image_data])

    start = time.time()
    for i in range(loop_count):
        out_list = model.predict([image_data])
    end = time.time()
    print("Average Inference time: {:.8f}ms".format((end - start) * 1000 /loop_count))

    handle_prediction(out_list, image_file, image, image_shape, anchors, class_names, model_image_size)
    return


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
    model_image_size = (height, width)

    image_data = preprocess_image(img, model_image_size)
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

    handle_prediction(out_list, image_file, image, image_shape, anchors, class_names, model_image_size)
    return


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

    model_image_size = (height, width)

    # prepare input image
    img = Image.open(image_file)
    image = np.array(img, dtype='uint8')
    image_data = preprocess_image(img, model_image_size)
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

    prediction = []
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

        prediction.append(output_data)

    handle_prediction(prediction, image_file, image, image_shape, anchors, class_names, model_image_size)
    return


def handle_prediction(prediction, image_file, image, image_shape, anchors, class_names, model_image_size):
    start = time.time()
    if len(anchors) == 5:
        # YOLOv2 use 5 anchors and have only 1 prediction
        assert len(prediction) == 1, 'invalid YOLOv2 prediction number.'
        boxes, classes, scores = yolo2_postprocess_np(prediction[0], image_shape, anchors, len(class_names), model_image_size)
    else:
        boxes, classes, scores = yolo3_postprocess_np(prediction, image_shape, anchors, len(class_names), model_image_size)

    end = time.time()
    print("PostProcess time: {:.8f}ms".format((end - start) * 1000))

    print('Found {} boxes for {}'.format(len(boxes), image_file))
    for box, cls, score in zip(boxes, classes, scores):
        print("Class: {}, Score: {}".format(class_names[cls], score))

    colors = get_colors(class_names)
    image = draw_boxes(image, boxes, classes, scores, class_names, colors)

    Image.fromarray(image).show()
    return


def main():
    parser = argparse.ArgumentParser(description='validate YOLO model (h5/tflite/mnn) with image')
    parser.add_argument('--model_path', help='model file to predict', type=str, required=True)
    parser.add_argument('--image_file', help='image file to predict', type=str, required=True)
    parser.add_argument('--anchors_path',help='path to anchor definitions', type=str, required=True)
    parser.add_argument('--classes_path', help='path to class definitions, default ../configs/voc_classes.txt', type=str, default='../configs/voc_classes.txt')
    parser.add_argument('--model_image_size', help='model image input size as <num>x<num>, default 416x416', type=str, default='416x416')
    parser.add_argument('--loop_count', help='loop inference for certain times', type=int, default=1)

    args = parser.parse_args()

    # param parse
    anchors = get_anchors(args.anchors_path)
    class_names = get_classes(args.classes_path)
    height, width = args.model_image_size.split('x')
    model_image_size = (int(height), int(width))

    # support of tflite model
    if args.model_path.endswith('.tflite'):
        validate_yolo_model_tflite(args.model_path, args.image_file, anchors, class_names, args.loop_count)
    # support of MNN model
    elif args.model_path.endswith('.mnn'):
        validate_yolo_model_mnn(args.model_path, args.image_file, anchors, class_names, args.loop_count)
    # normal keras h5 model
    else:
        validate_yolo_model(args.model_path, args.image_file, anchors, class_names, model_image_size, args.loop_count)


if __name__ == '__main__':
    main()
