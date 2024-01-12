#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, argparse
import math, glob
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm

def nv12_convert_bk(image_file, width, height):
    """
    Reference from:
    https://gist.github.com/fzakaria/2472889
    """
    f_y = open(image_file, "rb")
    f_uv = open(image_file, "rb")
    uv_start = width * height

    # check if it's a single frame image file
    size_of_file = os.path.getsize(image_file)
    size_of_frame = (3.0/2.0) * height * width
    assert (size_of_file == size_of_frame), 'not single frame file or incorrect image shape'

    image = Image.new("RGB", (width, height))
    pixels = image.load()

    # lets get our y cursor ready
    for j in range(0, height):
        for i in range(0, width):
            # uv_index starts at the end of the yframe.  The UV is 1/2 height so we multiply it by j/2
            # We need to floor i/2 to get the start of the UV byte
            uv_index = int(uv_start + (width * math.floor(j/2)) + (math.floor(i/2))*2)
            f_uv.seek(uv_index)

            y = ord(f_y.read(1))
            u = ord(f_uv.read(1))
            v = ord(f_uv.read(1))
            b = 1.164 * (y-16) + 2.018 * (u - 128)
            g = 1.164 * (y-16) - 0.813 * (v - 128) - 0.391 * (u - 128)
            r = 1.164 * (y-16) + 1.596*(v - 128)

            pixels[i,j] = int(r), int(g), int(b)
    return np.array(image)


def yuv420_convert_bk(image_file, width, height):
    """
    Reference from:
    https://blog.csdn.net/u012005313/article/details/70304922
    """
    # create Y
    Y = np.zeros((height, width), np.uint8)
    # create U,V
    U = np.zeros((height//2, width//2), np.uint8)
    V = np.zeros((height//2, width//2), np.uint8)

    # read Y,U,V data
    with open(image_file, 'rb') as reader:
        for i in range(0, height):
            for j in range(0, width):
                Y[i, j] = ord(reader.read(1))

        for i in range(0, height//2):
            for j in range(0, width//2):
                U[i, j] = ord(reader.read(1))

        for i in range(0, height//2):
            for j in range(0, width//2):
                V[i, j] = ord(reader.read(1))

    # resize U,V array to Y size
    enlarge_U = cv2.resize(U, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    enlarge_V = cv2.resize(V, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

    # merge Y,U,V channel
    img_YUV = cv2.merge([Y, enlarge_U, enlarge_V])
    image_array = cv2.cvtColor(img_YUV, cv2.COLOR_YUV2RGB)

    return image_array


def y_mode_convert_bk(image_file, width, height):
    # create Y
    Y = np.zeros((height, width), np.uint8)

    # read Y data
    with open(image_file, 'rb') as reader:
        for i in range(0, height):
            for j in range(0, width):
                Y[i, j] = ord(reader.read(1))

    # merge gray image to 3 channels
    image_array = cv2.merge([Y, Y, Y])
    return image_array




def nv12_convert(image_file, width, height):
    image_array = np.fromfile(image_file, dtype=np.uint8).reshape((int(height*1.5), width))
    image_array = cv2.cvtColor(image_array, cv2.COLOR_YUV2RGB_NV12)

    return image_array


def yuv420_convert(image_file, width, height):
    image_array = np.fromfile(image_file, dtype=np.uint8).reshape((int(height*1.5), width))
    image_array = cv2.cvtColor(image_array, cv2.COLOR_YUV2RGB_I420)

    return image_array


def y_mode_convert(image_file, width, height):
    image_array = np.fromfile(image_file, dtype=np.uint8).reshape((height, width))
    image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)

    return image_array



def main():
    parser = argparse.ArgumentParser(description='convert YUV image to jpg')
    parser.add_argument('--image_path', help='image file or directory to convert', type=str, required=True)
    parser.add_argument('--image_shape', help='image shape as <height>x<width>, default=%(default)s', type=str, default='480x640')
    parser.add_argument('--image_mode', help = "YUV image mode (NV12/YUV420/Y), default=%(default)s", type=str, required=False, default='NV12', choices=['NV12', 'YUV420', 'Y'])
    parser.add_argument('--output_path', help='output path to save target image, default=%(default)s', type=str, required=False, default=None)
    parser.add_argument('--target_format', type=str, required=False, default='jpg', choices=['jpg', 'png', 'bmp'],
                        help='target image file format. default=%(default)s')

    args = parser.parse_args()

    # param parse
    height, width = args.image_shape.split('x')
    height, width = (int(height), int(width))

    # get image file list or single image
    if os.path.isdir(args.image_path):
        image_files = glob.glob(os.path.join(args.image_path, '*'))
        assert args.output_path, 'need to specify output path if you use image directory as input.'
    else:
        image_files = [args.image_path]

    pbar = tqdm(total=len(image_files), desc='Converting YUV %s images'%(args.image_mode))
    # loop the sample list to convert
    for image_file in image_files:
        pbar.update(1)
        # support NV12 convert
        if args.image_mode == 'NV12':
            image = nv12_convert(image_file, width, height)
        # support YUV420 convert
        elif args.image_mode == 'YUV420':
            image = yuv420_convert(image_file, width, height)
        # support Y mode convert
        elif args.image_mode == 'Y':
            image = y_mode_convert(image_file, width, height)
        else:
            raise ValueError('unsupport image mode')

        # save or show result
        if args.output_path:
            os.makedirs(args.output_path, exist_ok=True)
            output_file = os.path.join(args.output_path, os.path.splitext(os.path.basename(image_file))[0]+'.'+args.target_format)
            Image.fromarray(image).save(output_file)
        else:
            Image.fromarray(image).show()
    pbar.close()


if __name__ == "__main__":
    main()
