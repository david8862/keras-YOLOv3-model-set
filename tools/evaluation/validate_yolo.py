#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
from PIL import Image
import os, sys, argparse
import numpy as np
from operator import mul
from functools import reduce
import MNN
import onnxruntime
from tensorflow.keras.models import load_model
from tensorflow.lite.python import interpreter as interpreter_wrapper
import tensorflow as tf

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))
from yolo3.postprocess_np import yolo3_postprocess_np
from yolo2.postprocess_np import yolo2_postprocess_np
from common.data_utils import preprocess_image
from common.utils import get_classes, get_anchors, get_colors, draw_boxes, get_custom_objects

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def validate_yolo_model(model_path, image_file, anchors, class_names, model_image_size, elim_grid_sense, loop_count):

    custom_object_dict = get_custom_objects()
    model = load_model(model_path, compile=False, custom_objects=custom_object_dict)

    img = Image.open(image_file)
    image = np.array(img, dtype='uint8')
    image_data = preprocess_image(img, model_image_size)
    #origin image shape, in (height, width) format
    image_shape = tuple(reversed(img.size))

    # predict once first to bypass the model building time
    model.predict([image_data])

    start = time.time()
    for i in range(loop_count):
        prediction = model.predict([image_data])
    end = time.time()
    print("Average Inference time: {:.8f}ms".format((end - start) * 1000 /loop_count))
    if type(prediction) is not list:
        prediction = [prediction]

    prediction.sort(key=lambda x: len(x[0]))
    handle_prediction(prediction, image_file, image, image_shape, anchors, class_names, model_image_size, elim_grid_sense)
    return


def validate_yolo_model_tflite(model_path, image_file, anchors, class_names, elim_grid_sense, loop_count):
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
    #origin image shape, in (height, width) format
    image_shape = tuple(reversed(img.size))

    # predict once first to bypass the model building time
    interpreter.set_tensor(input_details[0]['index'], image_data)
    interpreter.invoke()

    start = time.time()
    for i in range(loop_count):
        interpreter.set_tensor(input_details[0]['index'], image_data)
        interpreter.invoke()
    end = time.time()
    print("Average Inference time: {:.8f}ms".format((end - start) * 1000 /loop_count))

    prediction = []
    for output_detail in output_details:
        output_data = interpreter.get_tensor(output_detail['index'])
        prediction.append(output_data)

    prediction.sort(key=lambda x: len(x[0]))
    handle_prediction(prediction, image_file, image, image_shape, anchors, class_names, model_image_size, elim_grid_sense)
    return


def validate_yolo_model_mnn(model_path, image_file, anchors, class_names, elim_grid_sense, loop_count):
    interpreter = MNN.Interpreter(model_path)
    session = interpreter.createSession()

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
    #origin image shape, in (height, width) format
    image_shape = tuple(reversed(img.size))

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

    def get_tensor_list(output_tensors):
        # transform the output tensor dict to ordered tensor list, for further postprocess
        #
        # output tensor list should be like (for YOLOv3):
        # [
        #  (name, tensor) for (13, 13, 3, num_classes+5),
        #  (name, tensor) for (26, 26, 3, num_classes+5),
        #  (name, tensor) for (52, 52, 3, num_classes+5)
        # ]
        output_list = []

        for (output_tensor_name, output_tensor) in output_tensors.items():
            tensor_shape = output_tensor.getShape()
            dim_type = output_tensor.getDimensionType()
            tensor_height, tensor_width = tensor_shape[2:4] if dim_type == MNN.Tensor_DimensionType_Caffe else tensor_shape[1:3]

            if len(anchors) == 6:
                # Tiny YOLOv3
                if tensor_height == height//32:
                    output_list.insert(0, (output_tensor_name, output_tensor))
                elif tensor_height == height//16:
                    output_list.insert(1, (output_tensor_name, output_tensor))
                else:
                    raise ValueError('invalid tensor shape')
            elif len(anchors) == 9:
                # YOLOv3
                if tensor_height == height//32:
                    output_list.insert(0, (output_tensor_name, output_tensor))
                elif tensor_height == height//16:
                    output_list.insert(1, (output_tensor_name, output_tensor))
                elif tensor_height == height//8:
                    output_list.insert(2, (output_tensor_name, output_tensor))
                else:
                    raise ValueError('invalid tensor shape')
            elif len(anchors) == 5:
                # YOLOv2 use 5 anchors and have only 1 prediction
                assert len(output_tensors) == 1, 'YOLOv2 model should have only 1 output tensor.'
                output_list.insert(0, (output_tensor_name, output_tensor))
            else:
                raise ValueError('invalid anchor number')

        return output_list


    output_tensors = interpreter.getSessionOutputAll(session)
    output_tensor_list = get_tensor_list(output_tensors)

    prediction = []
    for (output_tensor_name, output_tensor) in output_tensor_list:
        output_shape = output_tensor.getShape()
        output_elementsize = reduce(mul, output_shape)
        print('output tensor name: {}, shape: {}'.format(output_tensor_name, output_shape))

        assert output_tensor.getDataType() == MNN.Halide_Type_Float

        # copy output tensor to host, for further postprocess
        tmp_output = MNN.Tensor(output_shape, output_tensor.getDataType(),\
                    #np.zeros(output_shape, dtype=float), output_tensor.getDimensionType())
                    tuple(np.zeros(output_shape, dtype=float).reshape(output_elementsize, -1)), output_tensor.getDimensionType())

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

    prediction.sort(key=lambda x: len(x[0]))
    handle_prediction(prediction, image_file, image, image_shape, anchors, class_names, model_image_size, elim_grid_sense)
    return


def validate_yolo_model_pb(model_path, image_file, anchors, class_names, model_image_size, elim_grid_sense, loop_count):
    # check tf version to be compatible with TF 2.x
    global tf
    if tf.__version__.startswith('2'):
        import tensorflow.compat.v1 as tf
        tf.disable_eager_execution()

    # NOTE: TF 1.x frozen pb graph need to specify input/output tensor name
    # so we hardcode the input/output tensor names here to get them from model
    if len(anchors) == 6:
        output_tensor_names = ['graph/predict_conv_1/BiasAdd:0', 'graph/predict_conv_2/BiasAdd:0']
    elif len(anchors) == 9:
        output_tensor_names = ['graph/predict_conv_1/BiasAdd:0', 'graph/predict_conv_2/BiasAdd:0', 'graph/predict_conv_3/BiasAdd:0']
    elif len(anchors) == 5:
        # YOLOv2 use 5 anchors and have only 1 prediction
        output_tensor_names = ['graph/predict_conv/BiasAdd:0']
    else:
        raise ValueError('invalid anchor number')

    # assume only 1 input tensor for image
    input_tensor_name = 'graph/image_input:0'

    #load frozen pb graph
    def load_pb_graph(model_path):
        # We parse the graph_def file
        with tf.gfile.GFile(model_path, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        # We load the graph_def in the default graph
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(
                graph_def,
                input_map=None,
                return_elements=None,
                name="graph",
                op_dict=None,
                producer_op_list=None
            )
        return graph

    graph = load_pb_graph(model_path)

    # We can list operations, op.values() gives you a list of tensors it produces
    # op.name gives you the name. These op also include input & output node
    # print output like:
    # prefix/Placeholder/inputs_placeholder
    # ...
    # prefix/Accuracy/predictions
    #
    # NOTE: prefix/Placeholder/inputs_placeholder is only op's name.
    # tensor name should be like prefix/Placeholder/inputs_placeholder:0

    #for op in graph.get_operations():
        #print(op.name, op.values())

    image_input = graph.get_tensor_by_name(input_tensor_name)
    output_tensors = [graph.get_tensor_by_name(output_tensor_name) for output_tensor_name in output_tensor_names]

    batch, height, width, channel = image_input.shape
    model_image_size = (int(height), int(width))

    img = Image.open(image_file)
    image = np.array(img, dtype='uint8')
    image_data = preprocess_image(img, model_image_size)
    #origin image shape, in (height, width) format
    image_shape = tuple(reversed(img.size))

    # predict once first to bypass the model building time
    with tf.Session(graph=graph) as sess:
        prediction = sess.run(output_tensors, feed_dict={
            image_input: image_data
        })

    start = time.time()
    for i in range(loop_count):
            with tf.Session(graph=graph) as sess:
                prediction = sess.run(output_tensors, feed_dict={
                    image_input: image_data
                })
    end = time.time()
    print("Average Inference time: {:.8f}ms".format((end - start) * 1000 /loop_count))

    prediction.sort(key=lambda x: len(x[0]))
    handle_prediction(prediction, image_file, image, image_shape, anchors, class_names, model_image_size, elim_grid_sense)


def validate_yolo_model_onnx(model_path, image_file, anchors, class_names, elim_grid_sense, loop_count):
    sess = onnxruntime.InferenceSession(model_path)

    input_tensors = []
    for i, input_tensor in enumerate(sess.get_inputs()):
        input_tensors.append(input_tensor)

    # assume only 1 input tensor for image
    assert len(input_tensors) == 1, 'invalid input tensor number.'

    batch, height, width, channel = input_tensors[0].shape
    model_image_size = (height, width)

    # prepare input image
    img = Image.open(image_file)
    image = np.array(img, dtype='uint8')
    image_data = preprocess_image(img, model_image_size)
    #origin image shape, in (height, width) format
    image_shape = tuple(reversed(img.size))

    feed = {input_tensors[0].name: image_data}

    # predict once first to bypass the model building time
    prediction = sess.run(None, feed)

    start = time.time()
    for i in range(loop_count):
        prediction = sess.run(None, feed)

    end = time.time()
    print("Average Inference time: {:.8f}ms".format((end - start) * 1000 /loop_count))

    prediction.sort(key=lambda x: len(x[0]))
    handle_prediction(prediction, image_file, image, image_shape, anchors, class_names, model_image_size, elim_grid_sense)


def handle_prediction(prediction, image_file, image, image_shape, anchors, class_names, model_image_size, elim_grid_sense):
    start = time.time()
    if len(anchors) == 5:
        # YOLOv2 use 5 anchors and have only 1 prediction
        assert len(prediction) == 1, 'invalid YOLOv2 prediction number.'
        boxes, classes, scores = yolo2_postprocess_np(prediction[0], image_shape, anchors, len(class_names), model_image_size, elim_grid_sense=elim_grid_sense)
    else:
        boxes, classes, scores = yolo3_postprocess_np(prediction, image_shape, anchors, len(class_names), model_image_size, elim_grid_sense=elim_grid_sense)

    end = time.time()
    print("PostProcess time: {:.8f}ms".format((end - start) * 1000))

    print('Found {} boxes for {}'.format(len(boxes), image_file))
    for box, cls, score in zip(boxes, classes, scores):
        xmin, ymin, xmax, ymax = box
        print("Class: {}, Score: {}, Box: {},{}".format(class_names[cls], score, (xmin, ymin), (xmax, ymax)))

    colors = get_colors(class_names)
    image = draw_boxes(image, boxes, classes, scores, class_names, colors)

    Image.fromarray(image).show()
    return


def main():
    parser = argparse.ArgumentParser(description='validate YOLO model (h5/pb/onnx/tflite/mnn) with image')
    parser.add_argument('--model_path', help='model file to predict', type=str, required=True)
    parser.add_argument('--image_file', help='image file to predict', type=str, required=True)
    parser.add_argument('--anchors_path', help='path to anchor definitions', type=str, required=True)
    parser.add_argument('--classes_path', help='path to class definitions, default=%(default)s', type=str, default='../../configs/voc_classes.txt')
    parser.add_argument('--model_image_size', help='model image input size as <height>x<width>, default=%(default)s', type=str, default='416x416')
    parser.add_argument('--elim_grid_sense', help="Eliminate grid sensitivity", default=False, action="store_true")
    parser.add_argument('--loop_count', help='loop inference for certain times', type=int, default=1)

    args = parser.parse_args()

    # param parse
    anchors = get_anchors(args.anchors_path)
    class_names = get_classes(args.classes_path)
    height, width = args.model_image_size.split('x')
    model_image_size = (int(height), int(width))
    assert (model_image_size[0]%32 == 0 and model_image_size[1]%32 == 0), 'model_image_size should be multiples of 32'

    # support of tflite model
    if args.model_path.endswith('.tflite'):
        validate_yolo_model_tflite(args.model_path, args.image_file, anchors, class_names, args.elim_grid_sense, args.loop_count)
    # support of MNN model
    elif args.model_path.endswith('.mnn'):
        validate_yolo_model_mnn(args.model_path, args.image_file, anchors, class_names, args.elim_grid_sense, args.loop_count)
    # support of TF 1.x frozen pb model
    elif args.model_path.endswith('.pb'):
        validate_yolo_model_pb(args.model_path, args.image_file, anchors, class_names, model_image_size, args.elim_grid_sense, args.loop_count)
    # support of ONNX model
    elif args.model_path.endswith('.onnx'):
        validate_yolo_model_onnx(args.model_path, args.image_file, anchors, class_names, args.elim_grid_sense, args.loop_count)
    # normal keras h5 model
    elif args.model_path.endswith('.h5'):
        validate_yolo_model(args.model_path, args.image_file, anchors, class_names, model_image_size, args.elim_grid_sense, args.loop_count)
    else:
        raise ValueError('invalid model file')


if __name__ == '__main__':
    main()
