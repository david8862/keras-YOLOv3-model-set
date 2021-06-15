#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, argparse, glob
import cv2
import json
import numpy as np
from PIL import Image
from timeit import time
from collections import deque

# implementation of SORT tracker
from model.sort.sort import Sort
# implementation of DeepSORT tracker
from model.deep_sort import preprocessing
from model.deep_sort import nn_matching
from model.deep_sort.detection import Detection
from model.deep_sort.tracker import Tracker
from model.deep_sort.generate_detections import create_box_encoder

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
from yolo import YOLO, YOLO_np
from common.utils import get_classes


def get_frame(frame_capture, i, images_input):
    # get frame from video or image folder
    if images_input:
        if i >= len(frame_capture):
            ret = False
            frame = None
        else:
            ret = True
            image_file = frame_capture[i]
            frame = cv2.imread(image_file)
    else:
        ret, frame = frame_capture.read()

    return ret, frame


def get_tracking_object(out_boxes, out_classnames, out_scores, tracking_class_names=None, convert_box=True):
    tracking_boxes = []
    tracking_class_names = []
    tracking_scores = []
    for i, out_classname in enumerate(out_classnames):
        # if object class not for tracking, bypass it
        if tracking_class_names and (out_classname not in tracking_class_names):
           continue
        score = out_scores[i]

        # detection box in (xmin,ymin,xmax,ymax) format,
        # convert to (x,y,w,h)
        box = out_boxes[i]
        x = int(box[0])
        y = int(box[1])
        w = int(box[2]-box[0])
        h = int(box[3]-box[1])
        # adjust invalid box
        if x < 0:
            w = w + x
            x = 0
        if y < 0:
            h = h + y
            y = 0

        if convert_box:
            tracking_boxes.append([x,y,w,h])
        else:
            tracking_boxes.append(box)
        tracking_class_names.append(out_classname)
        tracking_scores.append(score)
    return tracking_boxes, tracking_class_names, tracking_scores


def deepsort(yolo, args):
    #nms_max_overlap = 0.3 #nms threshold

    images_input = True if os.path.isdir(args.input) else False
    if images_input:
        # get images list
        jpeg_files = glob.glob(os.path.join(args.input, '*.jpeg'))
        jpg_files = glob.glob(os.path.join(args.input, '*.jpg'))
        frame_capture = jpeg_files + jpg_files
        frame_capture.sort()
    else:
        # create video capture stream
        frame_capture = cv2.VideoCapture(0 if args.input == '0' else args.input)
        if not frame_capture.isOpened():
            raise IOError("Couldn't open webcam or video")

    # create video save stream if needed
    save_video = True if args.output != "" else False
    if save_video:
        if images_input:
            raise ValueError("image folder input could be saved to video file")

        # here we encode the video to MPEG-4 for better compatibility, you can use ffmpeg later
        # to convert it to x264 to reduce file size:
        # ffmpeg -i test.mp4 -vcodec libx264 -f mp4 test_264.mp4
        #
        #video_FourCC    = cv2.VideoWriter_fourcc(*'XVID') if args.input == '0' else cv2.VideoWriter_fourcc(*"mp4v")
        video_FourCC    = cv2.VideoWriter_fourcc(*"mp4v")
        video_fps       = frame_capture.get(cv2.CAP_PROP_FPS)
        video_size      = (int(frame_capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                            int(frame_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        out = cv2.VideoWriter(args.output, video_FourCC, (5. if args.input == '0' else video_fps), video_size)

    if args.tracking_classes_path:
        # load the object classes used in tracking if have, other class
        # from detector will be ignored
        tracking_class_names = get_classes(args.tracking_classes_path)
    else:
        tracking_class_names = None


    #create deep_sort box encoder
    encoder = create_box_encoder(args.deepsort_model_path, batch_size=1)

    #create deep_sort tracker
    max_cosine_distance = 0.5 #threshold for cosine distance
    nn_budget = None
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)


    # alloc a set of queues to record motion trace
    # for each track id
    motion_traces = [deque(maxlen=30) for _ in range(9999)]
    total_obj_counter = []

    # initialize a list of colors to represent each possible class label
    np.random.seed(100)
    COLORS = np.random.randint(0, 255, size=(200, 3), dtype="uint8")

    # init json dict for final output
    output_dict = {
        'filename': 'test',
        'class':    'video',
        'frames': []
    }

    i=0
    fps = 0.0
    while True:
        ret, frame = get_frame(frame_capture, i, images_input)
        if ret != True:
            break
        #time.sleep(0.2)
        i += 1

        start_time = time.time()
        image = Image.fromarray(frame[...,::-1]) # bgr to rgb

        # init json dict for 1 frame
        frame_dict = {
            'timestamp': float(i), # just use frame id as timestamp
            'num': int(i), # just use frame id as num
            'class': 'frame',
            'hypotheses': []
        }

        # detect object from image
        _, out_boxes, out_classnames, out_scores = yolo.detect_image(image)
        # get tracking objects and convert bbox from (xmin,ymin,xmax,ymax) to (x,y,w,h)
        boxes, class_names, scores = get_tracking_object(out_boxes, out_classnames, out_scores, tracking_class_names)

        # get encoded features of bbox area image
        features = encoder(frame, boxes)

        # form up detection records
        detections = [Detection(bbox, score, feature, class_name) for bbox, score, class_name, feature in zip(boxes, scores, class_names, features)]

        # Run non-maximum suppression.
        #nms_boxes = np.array([d.tlwh for d in detections])
        #nms_scores = np.array([d.confidence for d in detections])
        #indices = preprocessing.non_max_suppression(nms_boxes, nms_max_overlap, nms_scores)
        #detections = [detections[i] for i in indices]

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        # show all detection result as white box
        for det in detections:
            bbox = det.to_tlbr()
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255,255,255), 2)
            cv2.putText(frame, str(det.class_name), (int(bbox[0]), int(bbox[1]-20)), 0, 5e-3*150, (255,255,255), 2)

        track_indexes = []
        track_count = 0
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            # record tracking info and get bbox
            track_indexes.append(int(track.track_id))
            total_obj_counter.append(int(track.track_id))
            bbox = track.to_tlbr()

            # json dict for 1 track box
            track_dict = {
                'height': float(int(bbox[3]) - int(bbox[1])),
                'width': float(int(bbox[2]) - int(bbox[0])),
                'id': str(track.track_id),
                'y': float(int(bbox[1])),
                'x': float(int(bbox[0])),
            }
            # record box dict to frame
            frame_dict['hypotheses'].append(track_dict)

            # show all tracking result as color box
            color = [int(c) for c in COLORS[track_indexes[track_count] % len(COLORS)]]
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (color), 3)
            cv2.putText(frame, str(track.track_id), (int(bbox[0]), int(bbox[1]-20)), 0, 5e-3*150, (color), 2)
            if track.class_name:
               cv2.putText(frame, str(track.class_name), (int(bbox[0]+30), int(bbox[1]-20)), 0, 5e-3*150, (color), 2)

            track_count += 1

            # get center point (x,y) of current track bbox and record in queue
            center = (int(((bbox[0])+(bbox[2]))/2),int(((bbox[1])+(bbox[3]))/2))
            motion_traces[track.track_id].append(center)

            # draw current center point
            thickness = 5
            cv2.circle(frame,  (center), 1, color, thickness)
            #draw motion trace
            motion_trace = motion_traces[track.track_id]
            for j in range(1, len(motion_trace)):
                if motion_trace[j - 1] is None or motion_trace[j] is None:
                   continue
                thickness = int(np.sqrt(64 / float(j + 1)) * 2)
                cv2.line(frame, (motion_trace[j-1]), (motion_trace[j]), (color), thickness)

        # show tracking statistics
        total_obj_num = len(set(total_obj_counter))
        cv2.putText(frame, "Total Object Counter: " + str(total_obj_num), (int(20), int(120)), 0, 5e-3 * 200, (0,255,0), 2)
        cv2.putText(frame, "Current Object Counter: "+str(track_count), (int(20), int(80)), 0, 5e-3 * 200, (0,255,0), 2)
        cv2.putText(frame, "FPS: %f"%(fps), (int(20), int(40)), 0, 5e-3 * 200, (0,255,0), 3)

        # refresh window
        cv2.namedWindow("DeepSORT", 0);
        cv2.resizeWindow('DeepSORT', 1024, 768);
        cv2.imshow('DeepSORT', frame)

        # record frame dict to output
        output_dict['frames'].append(frame_dict)

        if save_video:
            #save a frame
            out.write(frame)

        end_time = time.time()
        fps = (fps + (1./(end_time - start_time))) / 2
        # Press q to stop video
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release everything if job is finished
    if not images_input:
        frame_capture.release()
    if save_video:
        out.release()
    if args.output_json:
        # save output json
        with open(args.output_json, 'w') as fp:
            json.dump([output_dict], fp, indent=4)

    cv2.destroyAllWindows()



def sort(yolo, args):
    images_input = True if os.path.isdir(args.input) else False
    if images_input:
        # get images list
        jpeg_files = glob.glob(os.path.join(args.input, '*.jpeg'))
        jpg_files = glob.glob(os.path.join(args.input, '*.jpg'))
        frame_capture = jpeg_files + jpg_files
        frame_capture.sort()
    else:
        # create video capture stream
        frame_capture = cv2.VideoCapture(0 if args.input == '0' else args.input)
        if not frame_capture.isOpened():
            raise IOError("Couldn't open webcam or video")

    # create video save stream if needed
    save_video = True if args.output != "" else False
    if save_video:
        if images_input:
            raise ValueError("image folder input could be saved to video file")

        # here we encode the video to MPEG-4 for better compatibility, you can use ffmpeg later
        # to convert it to x264 to reduce file size:
        # ffmpeg -i test.mp4 -vcodec libx264 -f mp4 test_264.mp4
        #
        #video_FourCC    = cv2.VideoWriter_fourcc(*'XVID') if args.input == '0' else int(frame_capture.get(cv2.CAP_PROP_FOURCC))
        video_FourCC    = cv2.VideoWriter_fourcc(*'XVID') if args.input == '0' else cv2.VideoWriter_fourcc(*"mp4v")
        video_fps       = frame_capture.get(cv2.CAP_PROP_FPS)
        video_size      = (int(frame_capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                            int(frame_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        out = cv2.VideoWriter(args.output, video_FourCC, (5. if args.input == '0' else video_fps), video_size)

    if args.tracking_classes_path:
        # load the object classes used in tracking if have, other class
        # from detector will be ignored
        tracking_class_names = get_classes(args.tracking_classes_path)
    else:
        tracking_class_names = None

    # create instance of the SORT tracker
    tracker = Sort(max_age=5, min_hits=3, iou_threshold=0.3)

    # alloc a set of queues to record motion trace
    # for each track id
    motion_traces = [deque(maxlen=30) for _ in range(9999)]
    total_obj_counter = []

    # initialize a list of colors to represent each possible class label
    np.random.seed(100)
    COLORS = np.random.randint(0, 255, size=(200, 3), dtype="uint8")

    # init json dict for final output
    output_dict = {
        'filename': 'test',
        'class':    'video',
        'frames': []
    }

    i=0
    fps = 0.0
    while True:
        ret, frame = get_frame(frame_capture, i, images_input)
        if ret != True:
            break
        #time.sleep(0.2)
        i += 1

        start_time = time.time()
        image = Image.fromarray(frame[...,::-1]) # bgr to rgb

        # init json dict for 1 frame
        frame_dict = {
            'timestamp': float(i), # just use frame id as timestamp
            'num': int(i), # just use frame id as num
            'class': 'frame',
            'hypotheses': []
        }

        # detect object from image
        _, out_boxes, out_classnames, out_scores = yolo.detect_image(image)
        # get tracking objects
        boxes, class_names, scores = get_tracking_object(out_boxes, out_classnames, out_scores, tracking_class_names, convert_box=False)

        # form up detection records
        if len(boxes) != 0:
            detections = np.array([bbox + [score] for bbox, score, class_name in zip(boxes, scores, class_names)])
        else:
            detections = np.empty((0, 5))

        # Call the tracker
        tracks = tracker.update(detections)

        # show all detection result as white box
        for j, bbox in enumerate(boxes):
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255,255,255), 2)
            cv2.putText(frame, class_names[j], (int(bbox[0]), int(bbox[1]-20)), 0, 5e-3*150, (255,255,255), 2)

        track_indexes = []
        track_count = 0
        for track in tracks:
            bbox = track[:4]
            track_id = int(track[4])

            # json dict for 1 track box
            track_dict = {
                'height': float(int(bbox[3]) - int(bbox[1])),
                'width': float(int(bbox[2]) - int(bbox[0])),
                'id': str(track_id),
                'y': float(int(bbox[1])),
                'x': float(int(bbox[0])),
            }
            # record box dict to frame
            frame_dict['hypotheses'].append(track_dict)

            # record tracking info and get bbox
            track_indexes.append(int(track_id))
            total_obj_counter.append(int(track_id))

            # show all tracking result as color box
            color = [int(c) for c in COLORS[track_id % len(COLORS)]]
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (color), 3)
            cv2.putText(frame, str(track_id), (int(bbox[0]), int(bbox[1]-20)), 0, 5e-3*150, (color), 2)

            #if track.class_name:
               #cv2.putText(frame, str(track.class_name), (int(bbox[0]+30), int(bbox[1]-20)), 0, 5e-3*150, (color), 2)

            track_count += 1

            # get center point (x,y) of current track bbox and record in queue
            center = (int(((bbox[0])+(bbox[2]))/2),int(((bbox[1])+(bbox[3]))/2))
            motion_traces[track_id].append(center)

            # draw current center point
            thickness = 5
            cv2.circle(frame,  (center), 1, color, thickness)
            #draw motion trace
            motion_trace = motion_traces[track_id]
            for j in range(1, len(motion_trace)):
                if motion_trace[j - 1] is None or motion_trace[j] is None:
                   continue
                thickness = int(np.sqrt(64 / float(j + 1)) * 2)
                cv2.line(frame, (motion_trace[j-1]), (motion_trace[j]), (color), thickness)

        # show tracking statistics
        total_obj_num = len(set(total_obj_counter))
        cv2.putText(frame, "Total Object Counter: " + str(total_obj_num), (int(20), int(120)), 0, 5e-3 * 200, (0,255,0), 2)
        cv2.putText(frame, "Current Object Counter: "+str(track_count), (int(20), int(80)), 0, 5e-3 * 200, (0,255,0), 2)
        cv2.putText(frame, "FPS: %f"%(fps), (int(20), int(40)), 0, 5e-3 * 200, (0,255,0), 3)

        # refresh window
        cv2.namedWindow("SORT", 0);
        cv2.resizeWindow('SORT', 1024, 768);
        cv2.imshow('SORT', frame)

        # record frame dict to output
        output_dict['frames'].append(frame_dict)

        if save_video:
            #save a frame
            out.write(frame)

        end_time = time.time()
        fps = (fps + (1./(end_time - start_time))) / 2
        # Press q to stop video
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release everything if job is finished
    if not images_input:
        frame_capture.release()
    if save_video:
        out.release()
    if args.output_json:
        # save output json
        with open(args.output_json, 'w') as fp:
            json.dump([output_dict], fp, indent=4)

    cv2.destroyAllWindows()



def main():
    # class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS, description='Demo of multi object tracking (MOT) with YOLO detection model')
    '''
    MOT model options
    '''
    parser.add_argument(
        '--tracking_model_type', type=str, default="sort", choices=['sort', 'deepsort'],
        help = "MOT model type (sort/deepsort), default=%(default)s"
    )

    parser.add_argument(
        '--tracking_classes_path', type=str, required=False,
        help='[Optional] Path to tracking class definitions, will track all detect classes if None, default=%(default)s', default=None
    )

    parser.add_argument(
        '--deepsort_model_path', type=str, required=False, default="tracking/weights/mars-small128.pb",
        help = "[Optional] DeepSORT encoder model path, default=%(default)s"
    )

    '''
    YOLO model options
    '''
    parser.add_argument(
        '--model_type', type=str,
        help='YOLO model type: yolo3_mobilenet_lite/tiny_yolo3_mobilenet/yolo3_darknet/..., default ' + YOLO.get_defaults("model_type")
    )

    parser.add_argument(
        '--weights_path', type=str,
        help='path to YOLO model weight file, default ' + YOLO.get_defaults("weights_path")
    )

    parser.add_argument(
        '--pruning_model', default=False, action="store_true",
        help='Whether using a pruning YOLO model/weights file, default ' + str(YOLO.get_defaults("pruning_model"))
    )

    parser.add_argument(
        '--anchors_path', type=str,
        help='path to YOLO anchor definitions, default ' + YOLO.get_defaults("anchors_path")
    )

    parser.add_argument(
        '--classes_path', type=str,
        help='path to YOLO detection class definitions, default ' + YOLO.get_defaults("classes_path")
    )

    parser.add_argument(
        '--model_image_size', type=str,
        help='YOLO detection model input size as <height>x<width>, default ' +
        str(YOLO.get_defaults("model_image_size")[0])+'x'+str(YOLO.get_defaults("model_image_size")[1]),
        default=str(YOLO.get_defaults("model_image_size")[0])+'x'+str(YOLO.get_defaults("model_image_size")[1])
    )

    parser.add_argument(
        '--score', type=float,
        help='score threshold for YOLO detection model, default ' + str(YOLO.get_defaults("score"))
    )

    parser.add_argument(
        '--iou', type=float,
        help='iou threshold for YOLO detection NMS, default ' + str(YOLO.get_defaults("iou"))
    )

    parser.add_argument(
        '--elim_grid_sense', default=False, action="store_true",
        help = "Whether to apply eliminate grid sensitivity in YOLO, default " + str(YOLO.get_defaults("elim_grid_sense"))
    )

    '''
    Command line positional arguments -- for video detection mode
    '''
    parser.add_argument(
        "--input", nargs='?', type=str, required=False, default='./path2your_video',
        help = "Input video file or images folder path"
    )

    parser.add_argument(
        "--output", nargs='?', type=str, default="",
        help = "[Optional] output video file path"
    )

    parser.add_argument(
        "--output_json", nargs='?', type=str, required=False, default=None,
        help = "[Optional] output json file for pymot eval"
    )

    args = parser.parse_args()
    # param parse
    if args.model_image_size:
        height, width = args.model_image_size.split('x')
        args.model_image_size = (int(height), int(width))
        assert (args.model_image_size[0]%32 == 0 and args.model_image_size[1]%32 == 0), 'model_image_size should be multiples of 32'

    # get YOLO wrapped detection object
    yolo = YOLO_np(**vars(args))

    # choose different MOT model
    if args.tracking_model_type == 'sort':
        sort(yolo, args)
    elif args.tracking_model_type == 'deepsort':
        deepsort(yolo, args)
    else:
        raise ValueError('Unsupported MOT model')


if __name__ == '__main__':
    main()
