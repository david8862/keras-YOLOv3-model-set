#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
script to convert MOT16 ground truth and detection txt annotation file
to pymot json format file
'''
import os, argparse
import json
import numpy as np
from tqdm import tqdm
#from collections import OrderedDict


def convert_mot16_annotation(annotation_file, output_json, ground_truth):
    """
    MOT16 txt annotation file line format
    ground truth (gt.txt):
        <frame_id>,<track_id>,<bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<ignore>,<classes>
        box are grouped with track id, like
            1,1,912,484,97,109,0,7,1
            2,1,912,484,97,109,0,7,1
            ...
            1,2,1338,418,167,379,1,1,1
            2,2,1342,417,168,380,1,1,1
            ...

    detection (det.txt):
        <frame_id>,<track_id(-1)>,<bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<confidence>,<mot3d_x(-1)>,<mot3d_y(-1)>,<mot3d_z(-1)>
        box are grouped with frame id, like
            1,-1,1359.1,413.27,120.26,362.77,2.3092,-1,-1,-1
            1,-1,571.03,402.13,104.56,315.68,1.5028,-1,-1,-1
            ...
            2,-1,1359.1,413.27,120.26,362.77,2.4731,-1,-1,-1
            2,-1,584.04,446.86,84.742,256.23,1.2369,-1,-1,-1
            ...
    """
    # init json dict for final output
    output_dict = {
        'filename': annotation_file,
        'class':    'video',
        'frames': []
    }

    # track statistic count
    total_track = []
    ignored_track = []

    # load annotation file
    with open(annotation_file, 'r') as f:
        annotation_lines = f.readlines()

    # get sorted frame id list
    frame_ids = [int(annotation.split(',')[0]) for annotation in annotation_lines]
    frame_ids = list(np.unique(frame_ids))

    pbar = tqdm(total=len(frame_ids), desc='annotation convert')
    for frame_id in frame_ids:
        pbar.update(1)

        # init json dict for 1 frame
        annotation_key = 'annotations' if ground_truth else 'hypotheses'
        frame_dict = {
            'timestamp': float(frame_id), # just use frame id as timestamp
            'num': int(frame_id), # just use frame id as num
            'class': 'frame',
            annotation_key: []
        }

        # walk through annotation lines to pick frame boxes
        for annotation in annotation_lines:
            annotation = annotation.split(',')
            annotation_frame_id = int(annotation[0])

            if annotation_frame_id == frame_id:
                # prepare json dict for 1 box
                track_dict = {
                    'height': float(annotation[5]),
                    'width': float(annotation[4]),
                    'id': annotation[1],
                    'y': float(annotation[3]),
                    'x': float(annotation[2]),
                }

                # set dco flag by 'ignore' for ground truth
                if ground_truth:
                    track_dict['dco'] = not bool(int(annotation[6]))

                # count track
                if track_dict['id'] not in total_track:
                    total_track.append(track_dict['id'])
                if (track_dict['dco'] == True) and (track_dict['id'] not in ignored_track):
                    ignored_track.append(track_dict['id'])

                frame_dict[annotation_key].append(track_dict)

        output_dict['frames'].append(frame_dict)
    pbar.close()

    # save output json
    with open(output_json, 'w') as fp:
        json.dump([output_dict], fp, indent=4)

    # print out track statistic
    print('\nDone for %s. Related statistic:'%(annotation_file))
    print('frame number: %d'%(len(frame_ids)))
    print('bbox number: %d'%(len(annotation_lines)))
    print('total track number: %d'%(len(total_track)))
    print('ignored track number: %d'%(len(ignored_track)))



def main():
    parser = argparse.ArgumentParser(description='convert MOT16 annotations to pymot json format')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--ground_truth_file', type=str, default=None, help="converted ground truth annotation file")
    group.add_argument('--detection_file', type=str, default=None, help="converted detection file")

    parser.add_argument('--output_json', type=str, required=True, help='Output json file')

    args = parser.parse_args()

    # specify annotation_file and output_path
    annotation_file = args.ground_truth_file if args.ground_truth_file else args.detection_result_file
    # a trick: using args.ground_truth_file as flag to check if we're converting a ground truth annotation
    convert_mot16_annotation(annotation_file, args.output_json, args.ground_truth_file)


if __name__ == "__main__":
    main()
