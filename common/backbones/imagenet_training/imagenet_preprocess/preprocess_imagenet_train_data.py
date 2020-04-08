#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract sub tarball for Imagenet 2012 train dataset
"""
import os, glob, argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_path', type=str, required=True, help='path to Imagenet train data')
    args = parser.parse_args()

    class_tar_files = glob.glob(os.path.join(args.train_data_path, '*.tar'))

    for i, class_tar_file in enumerate(class_tar_files):
        class_label = class_tar_file.split(os.path.sep)[-1].split('.')[-2]
        class_folder = os.path.join(args.train_data_path, class_label)
        os.makedirs(class_folder, exist_ok=True)

        print(class_folder)
        os.system('tar xf ' + class_tar_file + ' -C ' + class_folder)
        os.system('rm -rf ' + class_tar_file)


if __name__ == "__main__":
    main()

