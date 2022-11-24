#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
generate featuremap activation of specified layer for input images
with trained YOLO model

Reference:
    https://blog.csdn.net/qq_37781464/article/details/122946523
'''
import os, sys, argparse
import glob
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import Model, load_model
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.backend as K

# compatible with TF 2.x
if tf.__version__.startswith('2'):
    import tensorflow.compat.v1 as tf
    from tensorflow.compat.v1.keras import backend as K
    tf.disable_eager_execution()

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))
from common.utils import get_custom_objects, optimize_tf_gpu
from common.data_utils import preprocess_image

optimize_tf_gpu(tf, K)



def generate_featuremap(image_path, model_path, model_input_shape, layer_name, featuremap_path):
    # load model
    custom_object_dict = get_custom_objects()
    model = load_model(model_path, compile=False, custom_objects=custom_object_dict)
    K.set_learning_phase(0)
    model.summary()

    # create featuremap model
    featuremap_output = model.get_layer(layer_name).output
    featuremap_model = Model(inputs=model.input, outputs=featuremap_output)

    # get image file list or single image
    if os.path.isdir(image_path):
        jpeg_files = glob.glob(os.path.join(image_path, '*.jpeg'))
        jpg_files = glob.glob(os.path.join(image_path, '*.jpg'))
        image_list = jpeg_files + jpg_files

        #assert os.path.isdir(featuremap_path), 'need to provide a path for output featuremap'
        os.makedirs(featuremap_path, exist_ok=True)
        featuremap_list = [os.path.join(featuremap_path, os.path.splitext(os.path.basename(image_name))[0]+'.jpg') for image_name in image_list]
    else:
        image_list = [image_path]
        featuremap_list = [featuremap_path]

    # loop the sample list to generate all featuremaps
    for i, (image_file, featuremap_file) in enumerate(zip(image_list, featuremap_list)):
        # process input
        img = Image.open(image_file).convert('RGB')
        image = np.array(img, dtype='uint8')
        image_data = preprocess_image(img, model_input_shape)

        # get featuremap output
        featuremap = featuremap_model.predict([image_data])

        # prepare featuremap display grid array
        images_per_row = 16
        feature_height, feature_width, feature_num = featuremap.shape[1:]
        column_num = feature_num // images_per_row
        display_grid = np.zeros((feature_height*column_num, feature_width*images_per_row))

        # fill one grid with a featuremap
        for col in range(column_num):
            for row in range(images_per_row):
                channel_image = featuremap[0,:, :,col * images_per_row + row]

                # rescale featuremap to more visable value for display
                channel_image -= channel_image.mean()
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')

                display_grid[col * feature_height : (col + 1) * feature_height,
                             row * feature_width : (row + 1) * feature_width] = channel_image

        # adjust display size for grid image
        col_scale = 1. / feature_height
        row_scale = 1. / feature_width
        plt.figure(figsize=(row_scale * display_grid.shape[1],
                            col_scale * display_grid.shape[0]))

        plt.title('Feature map of layer '+layer_name+'\nHeight:'+str(feature_height)+', Width:'+str(feature_width)+', Channel:'+str(feature_num))
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')

        # save chart image & show it
        plt.savefig(featuremap_file, dpi=75)
        #plt.show()
        print('Feature map of layer {} for image {} has been saved'.format(layer_name, image_file))



def main():
    parser = argparse.ArgumentParser(description='check featuremap activation of specified layer with trained YOLO model')
    parser.add_argument('--model_path', type=str, required=True, help='model file to predict')
    parser.add_argument('--image_path', type=str, required=True, help='image file or directory for input')
    parser.add_argument('--model_input_shape', help='model image input shape as <height>x<width>, default=%(default)s', type=str, default='416x416')
    parser.add_argument('--layer_name', type=str, required=True, help='layer name to check feature map')
    parser.add_argument('--featuremap_path', type=str, required=True, help='output featuremap file or directory')

    args = parser.parse_args()

    height, width = args.model_input_shape.split('x')
    model_input_shape = (int(height), int(width))
    assert (model_input_shape[0]%32 == 0 and model_input_shape[1]%32 == 0), 'model_input_shape should be multiples of 32'

    generate_featuremap(args.image_path, args.model_path, model_input_shape, args.layer_name, args.featuremap_path)


if __name__ == "__main__":
    main()
