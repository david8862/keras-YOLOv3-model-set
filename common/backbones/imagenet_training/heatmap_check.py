#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
generate heatmap for input images to verify trained Imagenet classification model
'''
import os, sys, argparse
import glob
import numpy as np
from PIL import Image
import cv2
from tensorflow.keras.models import load_model
#import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.applications.resnet50 import decode_predictions

from data_utils import normalize_image, denormalize_image

# compatible with TF 2.x
if tf.__version__.startswith('2'):
    import tensorflow.compat.v1 as tf
    from tensorflow.compat.v1.keras import backend as K
    tf.disable_eager_execution()

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', '..'))
from common.utils import get_custom_objects, optimize_tf_gpu

optimize_tf_gpu(tf, K)


def layer_type(layer):
    # TODO: use isinstance() instead.
    return str(layer)[10:].split(" ")[0].split(".")[-1]

def detect_last_conv(model):
    # Names (types) of layers from end to beggining
    inverted_list_layers = [layer_type(layer) for layer in model.layers[::-1]]
    i = len(model.layers)

    for layer in inverted_list_layers:
        i -= 1
        # support Conv2D, DepthwiseConv2D and SeparableConv2D
        if layer == "Conv2D" or layer == "DepthwiseConv2D" or layer == "SeparableConv2D":
            return i

def get_target_size(model):
    if K.image_data_format() == 'channels_first':
        return model.input_shape[2:4]
    else:
        return model.input_shape[1:3]


def generate_heatmap(image_path, model_path, heatmap_path):
    # load model
    custom_object_dict = get_custom_objects()
    model = load_model(model_path, custom_objects=custom_object_dict)
    K.set_learning_phase(0)
    model.summary()

    # get image file list or single image
    if os.path.isdir(image_path):
        jpeg_files = glob.glob(os.path.join(image_path, '*.jpeg'))
        jpg_files = glob.glob(os.path.join(image_path, '*.jpg'))
        image_list = jpeg_files + jpg_files

        #assert os.path.isdir(heatmap_path), 'need to provide a path for output heatmap'
        os.makedirs(heatmap_path, exist_ok=True)
        heatmap_list = [os.path.join(heatmap_path, os.path.splitext(os.path.basename(image_name))[0]+'.jpg') for image_name in image_list]
    else:
        image_list = [image_path]
        heatmap_list = [heatmap_path]

    # loop the sample list to generate all heatmaps
    for i, (image_file, heatmap_file) in enumerate(zip(image_list, heatmap_list)):
        # process input
        target_size=get_target_size(model)

        img = Image.open(image_file).convert('RGB')
        img = img.resize(target_size, Image.BICUBIC)
        img = np.asarray(img).astype('float32')
        x = normalize_image(img)
        x = np.expand_dims(x, axis=0)

        # predict and get output
        preds = model.predict(x)
        index = np.argmax(preds[0])
        score = preds[0][index]
        max_output = model.output[:, index]

        # detect last conv layer
        last_conv_index = detect_last_conv(model)
        last_conv_layer = model.layers[last_conv_index]
        # get gradient of the last conv layer to the predicted class
        grads = K.gradients(max_output, last_conv_layer.output)[0]
        # pooling to get the feature gradient
        pooled_grads = K.mean(grads, axis=(0, 1, 2))
        # run the predict to get value
        iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
        pooled_grads_value, conv_layer_output_value = iterate([x])

        # apply the activation to each channel of the conv'ed feature map
        for j in range(pooled_grads_value.shape[0]):
            conv_layer_output_value[:, :, j] *= pooled_grads_value[j]

        # get mean of each channel, which is the heatmap
        heatmap = np.mean(conv_layer_output_value, axis=-1)
        # normalize heatmap to 0~1
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)
        #plt.matshow(heatmap)
        #plt.show()

        # overlap heatmap to frame image
        #img = cv2.imread(image_file)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, target_size)
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap = denormalize_image(heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed_img = heatmap * 0.4 + img

        # show predict class index or name on image
        result = decode_predictions(preds)
        print('Predict result:', result)
        cv2.putText(superimposed_img, '{name}:{conf:.3f}'.format(name=result[0][0][1], conf=float(result[0][0][2])),
                    (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    color=(0,0,255),
                    thickness=1,
                    lineType=cv2.LINE_AA)

        # save overlaped image
        cv2.imwrite(heatmap_file, superimposed_img)
        print("generate heatmap file {} ({}/{})".format(heatmap_file, i+1, len(image_list)))


def main():
    parser = argparse.ArgumentParser(description='check heatmap activation for Imagenet classification model (h5) with test images')
    parser.add_argument('--model_path', type=str, required=True, help='model file to predict')
    parser.add_argument('--image_path', type=str, required=True, help='Image file or directory to predict')
    parser.add_argument('--heatmap_path', type=str, required=True, help='output heatmap file or directory')

    args = parser.parse_args()

    generate_heatmap(args.image_path, args.model_path, args.heatmap_path)



if __name__ == "__main__":
    main()

