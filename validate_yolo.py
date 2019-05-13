import time
from PIL import Image
import os, argparse
import numpy as np

from predict import predict, get_classes, get_colors, get_anchors, draw_boxes
from tensorflow.keras.models import load_model


def validate_yolo_model(model, image_file, anchors, class_names, model_image_size, loop_count):
    image = Image.open(image_file)
    image = np.array(image, dtype='uint8')

    start = time.time()
    for i in range(loop_count):
        boxes, classes, scores = predict(model, image, anchors, len(class_names), model_image_size)
    end = time.time()

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
    parser.add_argument('--model_image_size', help='model image input size as <num>x<num>, default 416x416', type=str, default='416x416')
    parser.add_argument('--loop_count', help='loop inference for certain times', type=int, default=1)

    args = parser.parse_args()
    if not args.model_path:
        raise ValueError('model file is not specified')
    if not args.image_file:
        raise ValueError('image file is not specified')

    # param parse
    model = load_model(args.model_path, compile=False)
    anchors = get_anchors(args.anchors_path)
    class_names = get_classes(args.classes_path)
    height, width = args.model_image_size.split('x')
    model_image_size = (int(height), int(width))

    validate_yolo_model(model, args.image_file, anchors, class_names, model_image_size, args.loop_count)


if __name__ == '__main__':
    main()
