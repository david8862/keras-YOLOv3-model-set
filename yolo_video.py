import sys
import argparse
from yolo import YOLO, detect_video
from PIL import Image

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth=True   #dynamic alloc GPU resource
config.gpu_options.per_process_gpu_memory_fraction = 0.3  #GPU memory threshold 0.3
session = tf.Session(config=config)

def detect_img(yolo):
    while True:
        img = input('Input image filename:')
        try:
            image = Image.open(img)
        except:
            print('Open Error! Try again!')
            continue
        else:
            #yolo.predict(image)
            r_image = yolo.detect_image2(image)
            r_image.show()
    yolo.close_session()

FLAGS = None

if __name__ == '__main__':
    # class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument(
        '--model_path', type=str,
        help='path to model weight file, default ' + YOLO.get_defaults("model_path")
    )

    parser.add_argument(
        '--anchors_path', type=str,
        help='path to anchor definitions, default ' + YOLO.get_defaults("anchors_path")
    )

    parser.add_argument(
        '--classes_path', type=str,
        help='path to class definitions, default ' + YOLO.get_defaults("classes_path")
    )

    parser.add_argument(
        '--gpu_num', type=int,
        help='Number of GPU to use, default ' + str(YOLO.get_defaults("gpu_num"))
    )

    parser.add_argument(
        '--image', default=False, action="store_true",
        help='Image detection mode, will ignore all positional arguments'
    )
    '''
    Command line positional arguments -- for video detection mode
    '''
    parser.add_argument(
        "--input", nargs='?', type=str,required=False,default='./path2your_video',
        help = "Video input path"
    )

    parser.add_argument(
        "--output", nargs='?', type=str, default="",
        help = "[Optional] Video output path"
    )
    '''
    Command line positional arguments -- for model dump
    '''
    parser.add_argument(
        '--dump_model', default=False, action="store_true",
        help='Dump out training model to inference model'
    )

    parser.add_argument(
        '--output_model_file', type=str,
        help='output inference model file'
    )

    FLAGS = parser.parse_args()

    if FLAGS.dump_model:
        """
        Dump out training model to inference model
        """
        if not FLAGS.output_model_file:
            raise ValueError('output model file is not specified')

        print('Dumping out training model to inference model')
        YOLO(**vars(FLAGS)).dump_model_file(FLAGS.output_model_file)
        sys.exit()


    if FLAGS.image:
        """
        Image detection mode, disregard any remaining command line arguments
        """
        print("Image detection mode")
        if "input" in FLAGS:
            print(" Ignoring remaining command line arguments: " + FLAGS.input + "," + FLAGS.output)
        detect_img(YOLO(**vars(FLAGS)))
    elif "input" in FLAGS:
        detect_video(YOLO(**vars(FLAGS)), FLAGS.input, FLAGS.output)
    else:
        print("Must specify at least video_input_path.  See usage with --help.")
