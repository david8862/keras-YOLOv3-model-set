import argparse
import time
from numpy.lib.function_base import select

import tensorflow as tf
from tensorflow.keras import backend as K
assert float(tf.__version__[:3]) >= 2.3
import numpy as np
from PIL import Image
import cv2
import os

from utils import *

MAX_BOXES = 30

def get_classes(path):
    """Read class names from file."""
    with open(path, 'r') as f:
        return [line.strip() for line in f.readlines()]

def get_anchors(path):
    """Read anchor values from file."""
    with open(path, 'r') as f:
        return np.array([float(x) for x in f.readline().split(',')]).reshape(-1, 2)

def letterbox_image(image, size):
    '''resize image with unchanged aspect ratio using padding'''
    iw, ih = image.shape[0:2][::-1]
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)
    image = cv2.resize(image, (nw,nh), interpolation=cv2.INTER_CUBIC)
    new_image = np.zeros((size[1], size[0], 3), np.uint8)
    new_image.fill(128)
    dx = (w-nw)//2
    dy = (h-nh)//2
    new_image[dy:dy+nh, dx:dx+nw,:] = image
    return new_image

def sigmoid(x):
    return 1. / (1 + np.exp(-x))

def iou(box1, box2):
    '''
    Compute the intersection over union of two boxes. 

    Boxes are in [[x_min, y_min], [x_max, y_max]] 
    format coordinates.
    '''
    xi1 = max(box1[0][0], box2[0][0])
    yi1 = max(box1[0][1], box2[0][1])
    xi2 = min(box1[1][0], box2[1][0])
    yi2 = min(box1[1][1], box2[1][1])
    inter_area = (xi2 - xi1)*(yi2 - yi1)
    # Formula: Union(A,B) = A + B - Inter(A,B)
    box1_area = (box1[1][1] - box1[0][1]) * (box1[1][0]- box1[0][0])
    box2_area = (box2[1][1] - box2[0][1]) * (box2[1][0]- box2[0][0])
    union_area = (box1_area + box2_area) - inter_area
    # compute the IoU
    IoU = inter_area / union_area

    return IoU

def run_model(interpreter, img_rgb):
    """Run a quantised model on a given image."""

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    if (input_details[0]['dtype'] != np.uint8 or
            output_details[0]['dtype'] != np.uint8):
        raise ValueError('Not an interger-only model.')

    if any(input_details[0]['shape'][1:3] != [416, 416]):
        raise ValueError(
            'Expect input size 416*416, '
            f"got {input_details[0]['shape'][1]}*{input_details[0]['shape'][2]}."
        )

    img_rgb = letterbox_image(img_rgb.copy(), (416, 416))
    input_data = np.expand_dims(img_rgb, axis=0)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    start_time = time.time()
    interpreter.invoke()
    stop_time = time.time()
    print(f'Net forward-pass time: {(stop_time - start_time) * 1000} ms.')

    o1_scale, o1_zero = output_details[0]['quantization']
    o2_scale, o2_zero = output_details[1]['quantization']

    output_s = (np.squeeze(interpreter.get_tensor(output_details[0]['index'])).astype(np.float32) - o1_zero) * o1_scale
    output_l = (np.squeeze(interpreter.get_tensor(output_details[1]['index'])).astype(np.float32) - o2_zero) * o2_scale

    return output_s, output_l

def decode(feat_map, anchors, n_classes, threshold, img_orig_shape, net_input_shape=(416, 416)):
    """
    Decode output map to obtain bounding boxes,
    together with their corresponding scores and
    classes. 
    """

    grid_shape = feat_map.shape[:-1]  # (13,13) or (26,26)
    n_anchors = len(anchors)

    grid_y_coord = np.tile(
        np.arange(grid_shape[0]).reshape(-1, 1), grid_shape[0]
    ).reshape(1, grid_shape[0], grid_shape[0], 1).astype(np.float32)
    grid_x_coord = grid_y_coord.copy().T.reshape(1, grid_shape[0], grid_shape[1], 1).astype(np.float32)

    outputs = feat_map.reshape(1, *grid_shape, n_anchors, -1)
    _anchors = anchors.reshape(1, 1, n_anchors, -1).astype(np.float32)

    # Following are all normalised, now in [0, 1]
    bx = (sigmoid(outputs[..., 0]) + grid_x_coord) / grid_shape[0]  
    by = (sigmoid(outputs[..., 1]) + grid_y_coord) / grid_shape[1]  
    bw = np.multiply(_anchors[..., 0] / net_input_shape[0], np.exp(outputs[..., 2]))
    bh = np.multiply(_anchors[..., 1] / net_input_shape[1], np.exp(outputs[..., 3]))
    
    # Get all scores
    scores = (sigmoid(np.expand_dims(outputs[..., 4], -1)) *
                sigmoid(outputs[..., 5:])).reshape(-1, n_classes)
    
    # Reshape boxes and scale back to original image size
    ratio = net_input_shape[1] / img_orig_shape[1]
    letterboxed_height = ratio * img_orig_shape[0]
    scale = net_input_shape[0] / letterboxed_height
    offset = (net_input_shape[0] - letterboxed_height) / 2 / net_input_shape[0]
    bx = bx.flatten()
    by = (by.flatten() - offset) * scale
    bw = bw.flatten()
    bh = bh.flatten() * scale
    half_bw = bw / 2.
    half_bh = bh / 2.

    tl_x = np.multiply(bx - half_bw, img_orig_shape[1])
    tl_y = np.multiply(by - half_bh, img_orig_shape[0]) 
    br_x = np.multiply(bx + half_bw, img_orig_shape[1])
    br_y = np.multiply(by + half_bh, img_orig_shape[0])

    # Get indices of boxes with score higher than threshold
    indices = np.argwhere(scores >= threshold)
    selected_boxes, selected_scores = [], []
    for i in indices:
        i = tuple(i)
        selected_boxes.append(((tl_x[i[0]], tl_y[i[0]]), (br_x[i[0]], br_y[i[0]])) )
        selected_scores.append(scores[i])
    
    selected_boxes = np.array(selected_boxes)
    selected_scores = np.array(selected_scores)
    selected_classes = indices[:, 1]

    return selected_boxes, selected_scores, selected_classes

def nms_boxes(boxes, scores, classes):
    """Perform non-max suppression on the boxes."""
    
    assert(boxes.shape[0] == scores.shape[0])
    assert(boxes.shape[0] == classes.shape[0])

    # Sort boxes based on score
    indices = np.arange(len(scores))
    scores, sorted_is = (list(l) for l in zip(*sorted(zip(scores, indices), reverse=True)))
    boxes = list(boxes[sorted_is])
    classes = list(classes[sorted_is])

    # Run NMS on each class
    i = 0
    while True:
        if len(boxes) == 1 or i >= len(boxes) or i == MAX_BOXES:
            break

        best_box = boxes[i]

        to_remove = []
        for j in range(i+1, len(boxes)):
            other_box = boxes[j]
            box_iou = iou(best_box, other_box)
            if box_iou > 0.5:
                to_remove.append(j)
        
        for r in to_remove[::-1]:
            del boxes[r]
            del scores[r]
            del classes[r]
        i += 1
        
    return boxes[:MAX_BOXES], scores[:MAX_BOXES], classes[:MAX_BOXES]

def draw_boxes(image, boxes, scores, pred_classes, class_names):
    global colors

    for idx, (topleft, botright) in enumerate(boxes):
        cl = int(pred_classes[idx])
        color = tuple(map(int, colors[cl])) 

        topleft = tuple(topleft.astype(int))
        botright = tuple(botright.astype(int))

        # Draw box and class
        cv2.rectangle(image, topleft, botright, color, 2)
        textpos = (topleft[0]-2, topleft[1] - 3)
        score = scores[idx] * 100
        cl_name = class_names[cl]
        text = f"{cl_name} ({score:.2f}%)"
        cv2.putText(image, text, textpos, cv2.FONT_HERSHEY_DUPLEX,
                0.45, color, 1, cv2.LINE_AA)

def inference(interpreter, img, anchors, classes, threshold):
    img_orig_shape = img.shape
    output_s, output_l = run_model(interpreter, img)
    n_classes = len(classes)

    # Decode
    start = time.time()
    _boxes1, _scores1, _classes1 = decode(output_s, anchors[[3, 4, 5]], 
            n_classes, threshold, img_orig_shape)
    _boxes2, _scores2, _classes2 = decode(output_l, anchors[[1, 2, 3]], 
            n_classes, threshold, img_orig_shape)
    box_time = time.time() - start
    print(f"Box computation time: {box_time*1000} ms.\n")

    # This is needed to be able to append nicely when the output layers don't
    # return any boxes
    if _boxes1.shape[0] == 0:
        _boxes1 = np.empty([0, 2, 2])
        _scores1 = np.empty([0,])
        _classes1 = np.empty([0,])
    if _boxes2.shape[0] == 0:
        _boxes2 = np.empty([0, 2, 2])
        _scores2 = np.empty([0,])
        _classes2 = np.empty([0,])

    boxes = np.append(_boxes1, _boxes2, axis=0)
    scores = np.append(_scores1, _scores2, axis=0)
    pred_classes = np.append(_classes1, _classes2, axis=0)

    if len(boxes) > 0:
        boxes, scores, pred_classes = nms_boxes(boxes, scores, pred_classes)

    return boxes, scores, pred_classes


parser = argparse.ArgumentParser(
        'Run integer-only-quantised TF-Lite Yolo-v3 Tiny inference.'
    )
parser.add_argument(
    '--image',
    help='image to be classified',
)
parser.add_argument(
    '--video', 
    help='Run inference on video.'
)

parser.add_argument(
    '--model', 
    default=os.path.abspath('quant_model.tflite'), 
    help='.tflite model to be executed',
)

parser.add_argument(
    '--classes', 
    default=os.path.join('home', 'keras-YOLOv3-model-set', 'configs', 'coco_classes.txt'), 
    help='name of file containing classes',
)

parser.add_argument(
    '--anchors', 
    default=os.path.join('home', 'keras-YOLOv3-model-set', 'configs', 'tiny_yolo3_anchors.txt'), 
    help='anchor'
)
parser.add_argument(
    '-t', 
    '--threshold', 
    default=0.25, 
    help='Detection threshold.'
)
args = parser.parse_args()


if __name__ == '__main__':
    anchors = get_anchors(args.anchors)
    classes = get_classes(args.classes)

    colors = np.random.uniform(30, 255, size=(len(classes), 3))

    if args.image:
        img = cv2.imread(args.image)
        interpreter = tf.lite.Interpreter(model_path=args.model)
        print('Allocating tensors ...\n')
        interpreter.allocate_tensors()

        boxes, scores, pred_classes = inference(interpreter, img, anchors, classes, float(args.threshold))

        draw_boxes(img, boxes, scores, pred_classes, classes)
        cv2.imshow('Image', img)
        cv2.waitKey(0)

    elif args.video:
        vidpath, vidname = os.path.dirname(args.video), os.path.basename(args.video)
        outpath = os.path.join(vidpath, 'inf_'+vidname)

        cap = cv2.VideoCapture(args.video)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        outfile = cv2.VideoWriter(outpath, fourcc, fps, (width, height))        

        interpreter = tf.lite.Interpreter(model_path=args.model)
        print('Allocating tensors ...')
        interpreter.allocate_tensors()

        while True:
            # Read frame from video
            _, frame = cap.read()

            start_inf = time.time()
            boxes, scores, pred_classes = inference(interpreter, frame, anchors, classes, float(args.threshold))
            if len(boxes) > 0:
                draw_boxes(frame, boxes, scores, pred_classes, classes)
            inf_time = time.time() - start_inf
            fps = 1. / inf_time
            #print(f"Inference time: {inf_time*1000} ms.")
            cv2.putText(frame, f"FPS: {fps:.1f}", (20, 20), cv2.FONT_HERSHEY_DUPLEX,
                0.45, (200, 0, 200), 1, cv2.LINE_AA)

            outfile.write(frame)
            cv2.imshow('Video', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        outfile.release()
        