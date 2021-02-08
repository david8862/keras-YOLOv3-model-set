#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, argparse, glob
from PIL import Image
import imageio
from skimage.transform import resize
from tqdm import tqdm


# get number of image file
def getFileNum(elem):
    assert type(elem) is str, 'Invalid image file name.'

    file_name = os.path.basename(elem)
    file_name_no_ext = file_name.split('.')[0]
    number = int(file_name_no_ext)
    return number


def main():
    parser = argparse.ArgumentParser(description='Simple tool to create animated GIF image from JPEG image sequence')
    parser.add_argument('--image_path', help='Input path for images to be picked', type=str, required=True)
    parser.add_argument('--gif_path', help='Output GIF image file path', type=str, required=True)
    args = parser.parse_args()

    jpeg_files = glob.glob(os.path.join(args.image_path, '*.jpeg'))
    jpg_files = glob.glob(os.path.join(args.image_path, '*.jpg'))
    image_files = jpeg_files + jpg_files
    image_files.sort(key=getFileNum)

    assert (args.gif_path.lower().endswith('.gif')), 'output file should be .gif'

    images = []
    pbar = tqdm(total=len(image_files), desc='Load images')
    for image_file in image_files:
        #image = Image.open(image_file)
        #image = image.resize((640, 360), Image.BICUBIC)

        image = imageio.imread(image_file)
        image = resize(image, (360, 640))
        images.append(image)
        pbar.update(1)

    pbar.close()
    #images[0].save(args.gif_path, save_all=True, append_images=images[1:], optimize=True, duration=33, loop=0)
    imageio.mimsave(args.gif_path, images, 'GIF', duration=0.033, fps=30)


if __name__ == "__main__":
    main()

