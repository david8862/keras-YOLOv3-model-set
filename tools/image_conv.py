#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
run this scipt to convert grey-scale images to RGB
'''

from PIL import Image
import numpy as np
import os, glob
import csv
import argparse


def RGB_convert(path, dst):
    '''
    Solution1：use PIL image.convert()
    :param path: input image full path
    :param dst: output store folder path
    :return: sequence prefix with path
    '''
    b = Image.open(path)
    if b.mode != 'RGB':
        b = b.convert('RGB')
    # to numpy array
    rgb_array = np.asarray(b)
    # back to RGB image
    rgb_image = Image.fromarray(rgb_array)

    file_name = path.split(os.path.sep)[-1]

    # Touch output class path
    os.makedirs(dst, exist_ok=True)

    # Store fakeRGB image to output path
    output_full_path = os.path.join(dst, file_name)
    rgb_image.save(output_full_path)
    print(output_full_path)




def main():
    parser = argparse.ArgumentParser(description='convert grey-scale images to RGB')
    parser.add_argument('--output_path', help='Output path for the converted image', type=str, default=os.path.join(os.path.dirname(__file__), 'output'))
    args = parser.parse_args()

    # Scan "./" to get class folder
    class_folders = ['JPEGImages']
    for class_folder in class_folders:
        # Get the image file list. Now handle both jpg and bmp file
        jpg_files = glob.glob(os.path.join(class_folder, '*.jpg'))
        bmp_files = glob.glob(os.path.join(class_folder, '*.bmp'))
        img_files = jpg_files + bmp_files

        for img_file in img_files:
            RGB_convert(img_file, args.output_path)

if __name__ == "__main__":
    main()

