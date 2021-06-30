import argparse
import time

import tensorflow as tf
from tensorflow.keras import backend as K
assert float(tf.__version__[:3]) >= 2.3
import numpy as np
from PIL import Image
import os

from utils import *

def load_labels(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f.readlines()]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'Run integer-only-quantised TF-Lite Yolo-v3 Tiny inference.'
    )
    parser.add_argument(
        '-i', 
        '--image',
        default=os.path.abspath('example/dog.jpg'),
        help='image to be classified',
    )
    parser.add_argument(
        '-m', 
        '--model_file', 
        default=os.path.abspath('quant_model.tflite'), 
        help='.tflite model to be executed',
    )

    parser.add_argument(
        '-l', 
        '--label_file', 
        default=os.path.join('home', 'keras-YOLOv3-model-set', 'configs', 'coco_classes.txt'), 
        help='name of file containing labels',
    )
    
    parser.add_argument(
        '--model_anchors', 
        default=os.path.join('home', 'keras-YOLOv3-model-set', 'configs', 'tiny_yolo3_anchors.txt'), 
        help='anchor'
    )
    parser.add_argument(
        '-t', 
        '--threshold', 
        help='Detection threshold.'
    )
    args = parser.parse_args()

    interpreter = tf.lite.Interpreter(
        model_path=args.model_file,
    )
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    if (input_details[0]['dtype'] != np.uint8 or
            output_details[0]['dtype'] != np.uint8):
        print('Not an interger-only model.')

    height, width = input_details[0]['shape'][1:3]
    img = Image.open(args.image).resize((width, height))

    input_data = np.expand_dims(img, axis=0)

    interpreter.set_tensor(input_details[0]['index'], input_data)

    start_time = time.time()
    interpreter.invoke()
    stop_time = time.time()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    results = np.squeeze(output_data)

    print(f'output shape: {results.shape}]')

    print(f'time: {(stop_time - start_time) * 1000: .6f}ms')

