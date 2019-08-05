import time
from PIL import Image
import os, argparse
import numpy as np

from tensorflow.lite.python import interpreter as interpreter_wrapper
from yolo3.predict_np import yolo_head, handle_predictions, adjust_boxes
from yolo3.data import preprocess_image
from yolo3.utils import get_classes, get_anchors, get_colors, draw_boxes


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
    img = np.array(img, dtype='uint8')

    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    image, image_data = preprocess_image(img, (height, width))

    start = time.time()
    for i in range(loop_count):
        interpreter.set_tensor(input_details[0]['index'], image_data)
        interpreter.invoke()
    end = time.time()

    out_list = []
    for output_detail in output_details:
        output_data = interpreter.get_tensor(output_detail['index'])
        out_list.append(output_data)

    predictions = yolo_head(out_list, anchors, num_classes=len(class_names), input_dims=(height, width))

    boxes, classes, scores = handle_predictions(predictions, confidence=0.3, iou_threshold=0.4)
    boxes = adjust_boxes(boxes, image, (height, width))
    print('Found {} boxes for {}'.format(len(boxes), image_file))

    for box, cls, score in zip(boxes, classes, scores):
        print("Class: {}, Score: {}".format(class_names[cls], score))

    colors = get_colors(class_names)
    image = draw_boxes(image, boxes, classes, scores, class_names, colors)
    print("Average Inference time: {:.8f}s".format((end - start)/loop_count))

    Image.fromarray(image).show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', help='model file to predict', type=str)
    parser.add_argument('--image_file', help='image file to predict', type=str)
    parser.add_argument('--anchors_path',help='path to anchor definitions', type=str)
    parser.add_argument('--classes_path', help='path to class definitions, default model_data/voc_classes.txt', type=str, default='model_data/voc_classes.txt')
    parser.add_argument('--loop_count', help='loop inference for certain times', type=int, default=1)

    args = parser.parse_args()
    if not args.model_path:
        raise ValueError('model file is not specified')
    if not args.image_file:
        raise ValueError('image file is not specified')

    # param parse
    anchors = get_anchors(args.anchors_path)
    class_names = get_classes(args.classes_path)

    validate_yolo_model_tflite(args.model_path, args.image_file, anchors, class_names, args.loop_count)


if __name__ == '__main__':
    main()
