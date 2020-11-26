#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
A wrapper scipt to evaluate TIDE dAP with COCO style dataset

Reference paper & code:
https://arxiv.org/abs/2008.08115
https://github.com/dbolya/tide
'''
import os, argparse
from tidecv import TIDE, datasets


def tide_eval(annotation_file, result_file):
    tide = TIDE()
    tide.evaluate_range(datasets.COCO(annotation_file), datasets.COCOResult(result_file), mode=TIDE.BOX) # Use TIDE.MASK for masks
    tide.summarize()  # Summarize the results as tables in the console
    tide.plot()       # Show a summary figure. Specify a folder and it'll output a png to that folder.


def main():
    parser = argparse.ArgumentParser(description='evaluate TIDE dAP with tidecv')
    parser.add_argument('--coco_annotation_json', required=True, type=str, help='coco json annotation file')
    parser.add_argument('--coco_result_json', required=True, type=str, help='coco json result file')
    args = parser.parse_args()

    tide_eval(args.coco_annotation_json, args.coco_result_json)


if __name__ == "__main__":
    main()
