#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Do kmeans-evolved clustering with genetic algorithm to generate anchors on selected dataset

Reference:
https://github.com/ultralytics/yolov5/blob/master/utils/autoanchor.py
"""
import os, sys, argparse
import numpy as np
from PIL import Image
from scipy.cluster.vq import kmeans
from sklearn.cluster import KMeans
import warnings
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))
from common.utils import get_classes, get_anchors, get_dataset


def parse_dataset(dataset):
    """
    Parse image shapes and annotation bboxes from dataset

    Parameters
    ----------
    dataset: list of data samples load from annotation file

    Returns
    -------
    shapes: numpy array of image shapes in (w,h) format, shape=(N, 2)
    bboxes: list of numpy array with bbox info for each image
            bbox format (x_min, y_min, x_max, y_max, class_id)
    """
    shapes = []
    bboxes = []

    for annotation_line in tqdm(dataset, desc="loading dataset"):
        line = annotation_line.split()

        # get shape by opening image
        image = Image.open(line[0]).convert('RGB')
        image_size = image.size # (w,h) format
        shapes.append(image_size)

        # contunue if no annotation bbox
        if len(line) == 1:
            continue

        # parse bbox info
        boxes = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])
        bboxes.append(boxes)

    if len(dataset) != len(bboxes):
        warnings.warn('bbox list mismatch with dataset list. maybe there is image without bbox.')

    return np.array(shapes), bboxes


def save_anchors(anchors, output_file):
    f = open(output_file, 'w')

    for i in range(len(anchors)):
        if i == 0:
            anchor = "%d,%d" % (anchors[i][0], anchors[i][1])
        else:
            anchor = ", %d,%d" % (anchors[i][0], anchors[i][1])
        f.write(anchor)

    f.write("\n")
    f.close()


def check_anchors(dataset, anchors, img_size, ratio_threshold=4.0, output_file=None):
    """
    Check anchor fit to data, recompute if necessary
    """
    print('\nAnalyzing anchors... ')

    # parse dataset
    img_shapes, bboxes = parse_dataset(dataset)

    # get image shapes aligned with img_size, and a random scale jitter
    shapes = np.array(img_size) * img_shapes / img_shapes.max(axis=1, keepdims=True)
    scales = np.random.uniform(0.9, 1.1, size=(shapes.shape[0], 1))

    # get random scaled bbox width and height
    bbox_wh = np.concatenate([(bbox[:, 2:4] - bbox[:, :2]) * shape * scale / img_shape for shape, scale, img_shape, bbox in zip(shapes, scales, img_shapes, bboxes)])

    def metric(anc):
        # calculate width & height ratio between bboxes and anchors
        # ratio.shape = (num_bbox, num_anchors, 2)
        ratio = bbox_wh[:, None] / anc[None]

        # pick the min ratio in width and height as ratio,
        # for each bbox and each anchor
        ratio = np.minimum(ratio, 1./ratio).min(axis=2)

        # choose the max ratio in all anchors as best ratio,
        # for each bbox
        best_ratio = ratio.max(axis=1)

        # best possible recall & anchors above ratio_threshold
        bpr = (best_ratio > 1./ratio_threshold).mean()  # best possible recall
        aat = (ratio > 1./ratio_threshold).sum(axis=1).mean()  # anchors above threshold
        return bpr, aat

    bpr, aat = metric(np.array(anchors).reshape(-1, 2))
    print('Anchors Above Threshold (AAT) = %.2f, Best Possible Recall (BPR) = %.4f' % (aat, bpr))

    if bpr < 0.98:  # threshold to recompute
        print('Attempting to improve anchors, please wait...')
        num_anchors = len(anchors)  # number of anchors

        new_anchors = kmean_anchors(dataset, num_anchors=num_anchors, img_size=img_size, ratio_threshold=ratio_threshold)

        new_bpr, _ = metric(new_anchors.reshape(-1, 2))
        if new_bpr > bpr:  # replace anchors
            print('Got better new anchors. You can use it for training.')
            if output_file:
                save_anchors(new_anchors, output_file)
                print('new anchors has been saved to', output_file)
        else:
            print('Original anchors better than new anchors. Just return original anchors.')
            new_anchors = anchors
    else:
        print('Original anchors are good enough.')
        new_anchors = anchors

    return new_anchors



def kmean_anchors(dataset, num_anchors, img_size, ratio_threshold=4.0, generation=1000):
    """
    Create kmeans-evolved anchors from training dataset

    Arguments:
        path: path to dataset *.yaml, or a loaded dataset
        num_anchors: number of anchors
        img_size: image size used for training
        ratio_threshold: anchor-label wh ratio threshold hyperparameter hyp['anchor_t'] used for training, default=4.0
        generation: generations to evolve anchors using genetic algorithm

    Return:
        anchors: kmeans evolved anchors
    """
    ratio_threshold = 1./ratio_threshold

    def metric(anchors, bbox_wh):
        # calculate width & height ratio between bboxes and anchors
        # ratio.shape = (num_bbox, num_anchors, 2)
        ratio = bbox_wh[:, None] / anchors[None]

        # pick the min ratio in width and height as ratio,
        # for each bbox and each anchor
        ratio = np.minimum(ratio, 1./ratio).min(axis=2)

        # choose the max ratio in all anchors as best ratio,
        # for each bbox
        best_ratio = ratio.max(axis=1)

        return ratio, best_ratio

    def anchor_fitness(anchors):
        '''
        check anchors' fitness after mutation,
        would be used during anchors evolve
        '''
        # get best ratio for each bbox
        _, best_ratio = metric(anchors, bbox_wh)

        # use average ratio above threshold as fitness metric
        fitness = (best_ratio * (best_ratio > ratio_threshold)).mean()

        return fitness

    def print_results(anchors):
        # check metric on raw bbox
        ratio, best_ratio = metric(anchors, bbox_wh_raw)

        # best possible recall & anchors above ratio_threshold
        bpr = (best_ratio > ratio_threshold).mean()
        aot = (ratio > ratio_threshold).mean() * num_anchors

        print('best possible recall=%.4f, average anchors over ratio_threshold=%.2f' % (bpr, aot))
        print('ratio between bbox & anchor: average=%.3f, best average=%.3f' % (ratio.mean(), best_ratio.mean()))
        print('average ratio in past ratio_threshold=%.3f' % (ratio[ratio > ratio_threshold].mean()))
        print('anchors:')
        for i, a in enumerate(anchors):
            print('%i,%i' % (round(a[0]), round(a[1])), end=', ' if i < len(anchors) - 1 else '\n\n')


    # parse dataset
    img_shapes, bboxes = parse_dataset(dataset)

    # get raw bbox width and height and align with img_size
    shapes = np.array(img_size) * img_shapes / img_shapes.max(axis=1, keepdims=True)
    bbox_wh_raw = np.concatenate([(bbox[:, 2:4] - bbox[:, :2]) * shape / img_shape for shape, img_shape, bbox in zip(shapes, img_shapes, bboxes)])

    # check if there's any small object (width or height < 3.0)
    small_obj_num = (bbox_wh_raw < 3.0).any(axis=1).sum()
    if small_obj_num:
        warnings.warn('Extremely small objects found. %g of %g labels are < 3 pixels in width or height.' % (small_obj_num, len(bbox_wh_raw)))

    # filter bbox to keep only width and height > 2 pixels
    bbox_wh = bbox_wh_raw[(bbox_wh_raw >= 2.0).any(1)]

    print('Running kmeans for %g anchors on %g points...' % (num_anchors, len(bbox_wh)))
    # get width & height std for value whitening
    bbox_std = bbox_wh.std(axis=0)

    # kmeans clustering to get new anchors
    anchors, _ = kmeans(bbox_wh/bbox_std, num_anchors, iter=30)  # points, mean distance
    #kmeans = KMeans(n_clusters=num_anchors, init='k-means++', n_init=10, max_iter=30).fit(bbox_wh/bbox_std)
    #anchors = kmeans.cluster_centers_

    anchors *= bbox_std

    # sort anchors with size, from small to large
    # "anchors.prod(axis=1)" calculate the anchors' size
    anchors = anchors[np.argsort(anchors.prod(axis=1))]

    print('Before applying Genetic Algorithm:')
    print_results(anchors)

    # start to evolve anchors
    fitness = anchor_fitness(anchors)

    anchor_shape = anchors.shape
    mutation_prob = 0.9
    mutation_range = 3.0 # mutation coefficient range: (1./mutation_range, mutation_range)
    sigma = 0.1

    pbar = tqdm(range(generation), desc='Evolving anchors with Genetic Algorithm')  # progress bar
    for _ in pbar:
        # init the mutation coefficient array
        mutate_coef = np.ones(anchor_shape)

        # keep mutate until a change occurs
        while (mutate_coef == 1).all():
            mutate_coef = ((np.random.random(anchor_shape) < mutation_prob) * np.random.random() * np.random.randn(*anchor_shape) * sigma + 1).clip(1./mutation_range, mutation_range)

        # multiply mutation coefficient with anchors,
        # to get mutated anchors
        anchor_mutate = (anchors.copy() * mutate_coef).clip(min=2.0)

        # check fitness for mutated anchors
        fitness_mutate = anchor_fitness(anchor_mutate)

        if fitness_mutate > fitness:
            # keep mutation result if fitness improved
            fitness, anchors = fitness_mutate, anchor_mutate.copy()
            anchors = anchors[np.argsort(anchors.prod(axis=1))]
            pbar.desc = 'Evolving anchors with Genetic Algorithm: fitness = %.4f' % fitness

    print('After applying Genetic Algorithm:')
    print_results(anchors)
    return np.around(anchors).astype(np.int32)



def main():
    parser = argparse.ArgumentParser(description='Do kmeans-evolved clustering to generate anchors or optimize existing anchors on selected dataset')
    parser.add_argument('--annotation_file', type=str, required=True,
            help='annotation txt file for ground truth anchors')
    parser.add_argument('--input_shape', type=str, required=False, default='416x416',
            help="model input shape as <height>x<width>, default=%(default)s")
    parser.add_argument('--output_file', type=str, required=False, default='./anchors.txt',
            help='output path for augmented images, default=%(default)s')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--anchors_path', type=str, default=None,
            help='exist anchors file to optimize')
    group.add_argument('--anchor_number', type=int, default=None,
            help='anchor numbers to cluster')

    args = parser.parse_args()
    height, width = args.input_shape.split('x')

    # here we use (w,h) format img_size to align with bbox order
    img_size = (int(width), int(height))

    # get dataset
    dataset = get_dataset(args.annotation_file, shuffle=False)

    if args.anchors_path:
        anchors = get_anchors(args.anchors_path)
        new_anchors = check_anchors(dataset, anchors, img_size=img_size)

    elif args.anchor_number:
        if args.anchor_number != 9 and args.anchor_number != 6 and args.anchor_number != 5:
            warnings.warn('You choose to generate {} anchor clusters, but default YOLO anchor number should 5, 6 or 9'.format(args.anchor_number))

        new_anchors = kmean_anchors(dataset, num_anchors=args.anchor_number, img_size=img_size)

    save_anchors(new_anchors, args.output_file)
    print('new anchors has been saved to', args.output_file)



if __name__ == "__main__":
    main()
