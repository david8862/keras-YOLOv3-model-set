#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare cosine similarity/distance of 2 tensors. The tensor is loaded
from text file with 1 line for a value, like:

    -1.630859
    0.275391
    -6.382324
    -5.061035
    -0.250488
    1.953613
"""
import os, argparse
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, paired_distances


def main():
    '''
    '''
    parser = argparse.ArgumentParser(description='compare cosine similarity/distance of 2 tensors from text file')
    parser.add_argument('--tensor_file_1', type=str, required=True)
    parser.add_argument('--tensor_file_2', type=str, required=True)
    args = parser.parse_args()

    tensor_1 = np.loadtxt(args.tensor_file_1).reshape(-1, 1).squeeze()
    tensor_2 = np.loadtxt(args.tensor_file_2).reshape(-1, 1).squeeze()


    simi = cosine_similarity([tensor_1], [tensor_2])
    print('cosine similarity:', simi)

    dist = paired_distances([tensor_1], [tensor_2], metric='cosine')
    print('cosine distance:', dist)


if __name__ == "__main__":
    main()
