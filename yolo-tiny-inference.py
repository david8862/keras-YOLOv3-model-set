import numpy as np
import tensorflow as tf
import sys
import cv2
from time import time
from utils import *
import argparse

EDGETPU_SHARED_LIB = "libedgetpu.so.1"

parser = argparse.ArgumentParser("Run TF-Lite YOLO-V3 Tiny inference.")
parser.add_argument("--model", required=True, help="Model to load.")
parser.add_argument("--anchors", required=True, help="Anchors file.")
parser.add_argument("--classes", required=True, help="Classes (.names) file.")
parser.add_argument("-t", "--threshold", help="Detection threshold.",
        type=float,
        default=0.25)
parser.add_argument("--edge_tpu", 
        help="Whether to delegate to Edge TPU or run on CPU.",
        action="store_true")
parser.add_argument("--quant",
        help="Indicates whether the model is quantized.",
        action="store_true")
parser.add_argument("--cam", help="Run inference on webcam.",
        action="store_true")
parser.add_argument("--image", help="Run inference on image.")
parser.add_argument("--video", help="Run inference on video.")
parser.add_argument("--output", help="inference output video. (video only)")
args = parser.parse_args()

def make_interpreter(model_path, edge_tpu=False):
    # Load the TF-Lite model and delegate to Edge TPU
    if edge_tpu:
        interpreter = tf.lite.Interpreter(model_path=model_path,
                experimental_delegates=[
                    tf.lite.experimental.load_delegate(EDGETPU_SHARED_LIB)
                    ])
    else:
        interpreter = tf.lite.Interpreter(model_path=model_path)

    return interpreter

# Run YOLO inference on the image, returns detected boxes
def inference(interpreter, img, anchors, n_classes, threshold):
    input_details, output_details, net_input_shape = \
            get_interpreter_details(interpreter)

    img_orig_shape = img.shape
    # Crop frame to network input shape
    img = letterbox_image(img.copy(), (416, 416))
    # Add batch dimension
    img = np.expand_dims(img, 0)

    if not args.quant:
        # Normalize image from 0 to 1
        img = np.divide(img, 255.).astype(np.float32)

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], img)

    start = time()

    # Run model
    interpreter.invoke()

    inf_time = time() - start
    print(f"Net forward-pass time: {inf_time*1000} ms.")

    # Retrieve outputs of the network
    out1 = interpreter.get_tensor(output_details[0]['index'])
    out2 = interpreter.get_tensor(output_details[1]['index'])

    # If this is a quantized model, dequantize the outputs
    if args.quant:
        # Dequantize output
        o1_scale, o1_zero = output_details[0]['quantization']
        out1 = (out1.astype(np.float32) - o1_zero) * o1_scale
        o2_scale, o2_zero = output_details[1]['quantization']
        out2 = (out2.astype(np.float32) - o2_zero) * o2_scale

    # Get boxes from outputs of network
    start = time()
    _boxes1, _scores1, _classes1 = featuresToBoxes(out1, anchors[[3, 4, 5]], 
            n_classes, net_input_shape, img_orig_shape, threshold)
    _boxes2, _scores2, _classes2 = featuresToBoxes(out2, anchors[[1, 2, 3]], 
            n_classes, net_input_shape, img_orig_shape, threshold)
    inf_time = time() - start
    print(f"Box computation time: {inf_time*1000} ms.")

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
    classes = np.append(_classes1, _classes2, axis=0)

    if len(boxes) > 0:
        boxes, scores, classes = nms_boxes(boxes, scores, classes)

    return boxes, scores, classes

def draw_boxes(image, boxes, scores, classes, class_names):
    i = 0
    for topleft, botright in boxes:
        # Detected class
        cl = int(classes[i])
        # This stupid thing below is needed for opencv to use as a color
        color = tuple(map(int, colors[cl])) 

        # Box coordinates
        topleft = (int(topleft[0]), int(topleft[1]))
        botright = (int(botright[0]), int(botright[1]))

        # Draw box and class
        cv2.rectangle(image, topleft, botright, color, 2)
        textpos = (topleft[0]-2, topleft[1] - 3)
        score = scores[i] * 100
        cl_name = class_names[cl]
        text = f"{cl_name} ({score:.1f}%)"
        cv2.putText(image, text, textpos, cv2.FONT_HERSHEY_DUPLEX,
                0.45, color, 1, cv2.LINE_AA)
        i += 1

def get_interpreter_details(interpreter):
    # Get input and output tensor details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]["shape"]

    return input_details, output_details, input_shape

def webcam_inf(interpreter, anchors, classes, threshold=0.25):
    cap = cv2.VideoCapture(0)

    input_details, output_details, input_shape = \
            get_interpreter_details(interpreter)

    n_classes = len(classes)

    # Load and process image
    while True:
        # Read frame from webcam
        ret, frame = cap.read()

        # Run inference, get boxes
        boxes, scores, pred_classes = inference(interpreter, frame, anchors, n_classes, threshold)

        if len(boxes) > 0:
            draw_boxes(frame, boxes, scores, pred_classes, classes)

        cv2.imshow("Image", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break

    cap.release()

def video_inf(interpreter, anchors, classes, path, output_path, threshold=0.25):
    cap = cv2.VideoCapture(path)

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) 
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    outfile = cv2.VideoWriter(output_path, fourcc, fps, (width,height))

    input_details, output_details, input_shape = \
            get_interpreter_details(interpreter)

    n_classes = len(classes)

    # Load and process image
    while True:
        # Read frame from webcam
        ret, frame = cap.read()

        # Run inference, get boxes
        start = time()
        boxes, scores, pred_classes = inference(interpreter, frame, anchors, n_classes, threshold)

        if len(boxes) > 0:
            draw_boxes(frame, boxes, scores, pred_classes, classes)

        inf_time = time() - start
        fps = 1. / inf_time
        print(f"Inference time: {inf_time*1000} ms.")

        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 20), cv2.FONT_HERSHEY_DUPLEX,
                0.45, (200, 0, 200), 1, cv2.LINE_AA)

        outfile.write(frame)
        cv2.imshow("Image", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break

    cap.release()
    outfile.release()

def image_inf(interpreter, anchors, img_fn, classes, threshold):
    img = cv2.imread(img_fn)

    n_classes = len(classes)

    input_details, output_details, input_shape = \
            get_interpreter_details(interpreter)

    # Run inference, get boxes
    boxes, scores, pred_classes = inference(interpreter, img, anchors, n_classes, threshold)

    if len(boxes) > 0:
        draw_boxes(img, boxes, scores, pred_classes, classes)

    cv2.imshow("Image", img)
    cv2.waitKey(0)

if __name__ == "__main__":
    anchors = get_anchors(args.anchors)
    classes = get_classes(args.classes)

    # Generate random colors for each detection
    colors = np.random.uniform(30, 255, size=(len(classes), 3))

    #interpreter = make_interpreter(args.model, args.edge_tpu)
    interpreter = tf.lite.Interpreter(model_path=args.model)
    print("Allocating tensors.")
    interpreter.allocate_tensors()

    if args.cam:
        webcam_inf(interpreter, anchors, classes, threshold=args.threshold)
    elif args.image:
        image_inf(interpreter, anchors, args.image, classes, threshold=args.threshold)
    elif args.video:
        video_inf(interpreter, anchors, classes, args.video, args.output, threshold=args.threshold)
