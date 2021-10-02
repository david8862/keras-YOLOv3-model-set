#!/usr/bin/python3
# -*- coding=utf-8 -*-
"""Data process utility functions."""
import numpy as np
import random
import math
import cv2
from PIL import Image, ImageEnhance, ImageFilter
import imgaug.augmenters as iaa
#from matplotlib.colors import rgb_to_hsv, hsv_to_rgb


def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a


def letterbox_resize(image, target_size, return_padding_info=False):
    """
    Resize image with unchanged aspect ratio using padding

    # Arguments
        image: origin image to be resize
            PIL Image object containing image data
        target_size: target image size,
            tuple of format (width, height).
        return_padding_info: whether to return padding size & offset info
            Boolean flag to control return value

    # Returns
        new_image: resized PIL Image object.

        padding_size: padding image size (keep aspect ratio).
            will be used to reshape the ground truth bounding box
        offset: top-left offset in target image padding.
            will be used to reshape the ground truth bounding box
    """
    src_w, src_h = image.size
    target_w, target_h = target_size

    # calculate padding scale and padding offset
    scale = min(target_w/src_w, target_h/src_h)
    padding_w = int(src_w * scale)
    padding_h = int(src_h * scale)
    padding_size = (padding_w, padding_h)

    dx = (target_w - padding_w)//2
    dy = (target_h - padding_h)//2
    offset = (dx, dy)

    # create letterbox resized image
    image = image.resize(padding_size, Image.BICUBIC)
    new_image = Image.new('RGB', target_size, (128,128,128))
    new_image.paste(image, offset)

    if return_padding_info:
        return new_image, padding_size, offset
    else:
        return new_image


def random_resize_crop_pad(image, target_size, aspect_ratio_jitter=0.3, scale_jitter=0.5):
    """
    Randomly resize image and crop|padding to target size. It can
    be used for data augment in training data preprocess

    # Arguments
        image: origin image to be resize
            PIL Image object containing image data
        target_size: target image size,
            tuple of format (width, height).
        aspect_ratio_jitter: jitter range for random aspect ratio,
            scalar to control the aspect ratio of random resized image.
        scale_jitter: jitter range for random resize scale,
            scalar to control the resize scale of random resized image.

    # Returns
        new_image: target sized PIL Image object.
        padding_size: random generated padding image size.
            will be used to reshape the ground truth bounding box
        padding_offset: random generated offset in target image padding.
            will be used to reshape the ground truth bounding box
    """
    target_w, target_h = target_size

    # generate random aspect ratio & scale for resize
    rand_aspect_ratio = target_w/target_h * rand(1-aspect_ratio_jitter,1+aspect_ratio_jitter)/rand(1-aspect_ratio_jitter,1+aspect_ratio_jitter)
    rand_scale = rand(scale_jitter, 1/scale_jitter)

    # calculate random padding size and resize
    if rand_aspect_ratio < 1:
        padding_h = int(rand_scale * target_h)
        padding_w = int(padding_h * rand_aspect_ratio)
    else:
        padding_w = int(rand_scale * target_w)
        padding_h = int(padding_w / rand_aspect_ratio)
    padding_size = (padding_w, padding_h)
    image = image.resize(padding_size, Image.BICUBIC)

    # get random offset in padding image
    dx = int(rand(0, target_w - padding_w))
    dy = int(rand(0, target_h - padding_h))
    padding_offset = (dx, dy)

    # create target image
    new_image = Image.new('RGB', (target_w, target_h), (128,128,128))
    new_image.paste(image, padding_offset)

    return new_image, padding_size, padding_offset


def reshape_boxes(boxes, src_size, target_size, padding_size, offset, horizontal_flip=False, vertical_flip=False):
    """
    Reshape bounding boxes from src_size image to target_size image,
    usually for training data preprocess

    # Arguments
        boxes: Ground truth object bounding boxes,
            numpy array of shape (num_boxes, 5),
            box format (xmin, ymin, xmax, ymax, cls_id).
        src_size: origin image size,
            tuple of format (width, height).
        target_size: target image size,
            tuple of format (width, height).
        padding_size: padding image shape,
            tuple of format (width, height).
        offset: top-left offset when padding target image.
            tuple of format (dx, dy).
        horizontal_flip: whether to do horizontal flip.
            boolean flag.
        vertical_flip: whether to do vertical flip.
            boolean flag.

    # Returns
        boxes: reshaped bounding box numpy array
    """
    if len(boxes)>0:
        src_w, src_h = src_size
        target_w, target_h = target_size
        padding_w, padding_h = padding_size
        dx, dy = offset

        # shuffle and reshape boxes
        np.random.shuffle(boxes)
        boxes[:, [0,2]] = boxes[:, [0,2]]*padding_w/src_w + dx
        boxes[:, [1,3]] = boxes[:, [1,3]]*padding_h/src_h + dy
        # horizontal flip boxes if needed
        if horizontal_flip:
            boxes[:, [0,2]] = target_w - boxes[:, [2,0]]
        # vertical flip boxes if needed
        if vertical_flip:
            boxes[:, [1,3]] = target_h - boxes[:, [3,1]]

        # check box coordinate range
        boxes[:, 0:2][boxes[:, 0:2] < 0] = 0
        boxes[:, 2][boxes[:, 2] > target_w] = target_w
        boxes[:, 3][boxes[:, 3] > target_h] = target_h

        # check box width and height to discard invalid box
        boxes_w = boxes[:, 2] - boxes[:, 0]
        boxes_h = boxes[:, 3] - boxes[:, 1]
        boxes = boxes[np.logical_and(boxes_w>1, boxes_h>1)] # discard invalid box

    return boxes


def random_hsv_distort(image, hue=.1, sat=1.5, val=1.5):
    """
    Random distort image in HSV color space
    usually for training data preprocess

    # Arguments
        image: origin image for HSV distort
            PIL Image object containing image data
        hue: distort range for Hue
            scalar
        sat: distort range for Saturation
            scalar
        val: distort range for Value(Brightness)
            scalar

    # Returns
        new_image: distorted PIL Image object.
    """
    # get random HSV param
    hue = rand(-hue, hue)
    sat = rand(1, sat) if rand()<.5 else 1/rand(1, sat)
    val = rand(1, val) if rand()<.5 else 1/rand(1, val)

    # transform color space from RGB to HSV
    x = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2HSV)

    # distort image
    # cv2 HSV value range:
    #     H: [0, 179]
    #     S: [0, 255]
    #     V: [0, 255]
    x = x.astype(np.float64)
    x[..., 0] = (x[..., 0] * (1 + hue)) % 180
    x[..., 1] = x[..., 1] * sat
    x[..., 2] = x[..., 2] * val
    x[..., 1:3][x[..., 1:3]>255] = 255
    x[..., 1:3][x[..., 1:3]<0] = 0
    x = x.astype(np.uint8)

    # back to PIL RGB distort image
    x = cv2.cvtColor(x, cv2.COLOR_HSV2RGB)
    new_image = Image.fromarray(x)

    return new_image


def random_brightness(image, jitter=.5):
    """
    Random adjust brightness for image

    # Arguments
        image: origin image for brightness change
            PIL Image object containing image data
        jitter: jitter range for random brightness,
            scalar to control the random brightness level.

    # Returns
        new_image: adjusted PIL Image object.
    """
    enh_bri = ImageEnhance.Brightness(image)
    brightness = rand(jitter, 1/jitter)
    new_image = enh_bri.enhance(brightness)

    return new_image


def random_chroma(image, jitter=.5):
    """
    Random adjust chroma (color level) for image

    # Arguments
        image: origin image for chroma change
            PIL Image object containing image data
        jitter: jitter range for random chroma,
            scalar to control the random color level.

    # Returns
        new_image: adjusted PIL Image object.
    """
    enh_col = ImageEnhance.Color(image)
    color = rand(jitter, 1/jitter)
    new_image = enh_col.enhance(color)

    return new_image


def random_contrast(image, jitter=.5):
    """
    Random adjust contrast for image

    # Arguments
        image: origin image for contrast change
            PIL Image object containing image data
        jitter: jitter range for random contrast,
            scalar to control the random contrast level.

    # Returns
        new_image: adjusted PIL Image object.
    """
    enh_con = ImageEnhance.Contrast(image)
    contrast = rand(jitter, 1/jitter)
    new_image = enh_con.enhance(contrast)

    return new_image


def random_sharpness(image, jitter=.5):
    """
    Random adjust sharpness for image

    # Arguments
        image: origin image for sharpness change
            PIL Image object containing image data
        jitter: jitter range for random sharpness,
            scalar to control the random sharpness level.

    # Returns
        new_image: adjusted PIL Image object.
    """
    enh_sha = ImageEnhance.Sharpness(image)
    sharpness = rand(jitter, 1/jitter)
    new_image = enh_sha.enhance(sharpness)

    return new_image


def random_horizontal_flip(image, prob=.5):
    """
    Random horizontal flip for image

    # Arguments
        image: origin image for horizontal flip
            PIL Image object containing image data
        prob: probability for random flip,
            scalar to control the flip probability.

    # Returns
        image: adjusted PIL Image object.
        flip: boolean flag for horizontal flip action
    """
    flip = rand() < prob
    if flip:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)

    return image, flip


def random_vertical_flip(image, prob=.2):
    """
    Random vertical flip for image

    # Arguments
        image: origin image for vertical flip
            PIL Image object containing image data
        prob: probability for random flip,
            scalar to control the flip probability.

    # Returns
        image: adjusted PIL Image object.
        flip: boolean flag for vertical flip action
    """
    flip = rand() < prob
    if flip:
        image = image.transpose(Image.FLIP_TOP_BOTTOM)

    return image, flip


def random_grayscale(image, prob=.2):
    """
    Random convert image to grayscale

    # Arguments
        image: origin image for grayscale convert
            PIL Image object containing image data
        prob: probability for grayscale convert,
            scalar to control the convert probability.

    # Returns
        image: adjusted PIL Image object.
    """
    convert = rand() < prob
    if convert:
        #convert to grayscale first, and then
        #back to 3 channels fake RGB
        image = image.convert('L')
        image = image.convert('RGB')

    return image


def random_blur(image, prob=.1):
    """
    Random add normal blur to image

    # Arguments
        image: origin image for blur
            PIL Image object containing image data
        prob: probability for blur,
            scalar to control the blur probability.

    # Returns
        image: adjusted PIL Image object.
    """
    blur = rand() < prob
    if blur:
        image = image.filter(ImageFilter.BLUR)

    return image


def random_motion_blur(image, prob=.1):
    """
    Random add motion blur on image

    # Arguments
        image: origin image for motion blur
            PIL Image object containing image data
        prob: probability for blur,
            scalar to control the blur probability.

    # Returns
        image: adjusted PIL Image object.
    """
    motion_blur = rand() < prob
    if motion_blur:
        img = np.array(image)
        # random blur severity from 1 to 5
        severity = np.random.randint(1, 6)

        seq = iaa.Sequential([iaa.imgcorruptlike.MotionBlur(severity=severity)])
        #seq = iaa.Sequential([iaa.MotionBlur(k=30)])

        img = seq(images=np.expand_dims(img, 0))
        image = Image.fromarray(img[0])

    return image


def random_rotate(image, boxes, rotate_range=20, prob=0.1):
    """
    Random rotate for image and bounding boxes

    reference:
        https://github.com/ultralytics/yolov5/blob/master/utils/datasets.py#L824

    NOTE: bbox area will be expand in many cases after
          rotate, like:
     _____________________________
    |                             |
    |                             |
    |    _____________            |
    |   |             |           |
    |   |   _______   |           |
    |   |  /\      \  |           |
    |   | /  \______\ |           |
    |   | |  |      | |           |
    |   | |__|______| |           |
    |   |             |           |
    |   |_____________|           |
    |                             |
    |                             |
    ------------------------------

    # Arguments
        image: origin image for rotate
            PIL Image object containing image data
        boxes: Ground truth object bounding boxes,
            numpy array of shape (num_boxes, 5),
            box format (xmin, ymin, xmax, ymax, cls_id).

        prob: probability for random rotate,
            scalar to control the rotate probability.

    # Returns
        image: rotated PIL Image object.
        boxes: rotated bounding box numpy array
    """
    if rotate_range:
        angle = random.gauss(mu=0.0, sigma=rotate_range)
    else:
        angle = 0.0

    warpAffine = rand() < prob
    if warpAffine and rotate_range:
        width, height = image.size
        scale = 1.0

        # get rotate matrix and apply for image
        M = cv2.getRotationMatrix2D(center=(width//2, height//2), angle=angle, scale=scale)
        img = cv2.warpAffine(np.array(image), M, (width, height), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)#, borderValue=(114, 114, 114))

        # rotate boxes coordinates
        n = len(boxes)
        if n:
            # form up 4 corner points ([xmin,ymin], [xmax,ymax], [xmin,ymax], [xmax,ymin])
            # from (xmin, ymin, xmax, ymax), every coord is [x,y,1] format for applying
            # rotation matrix
            corner_points = np.ones((n * 4, 3))
            corner_points[:, :2] = boxes[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2) # [xmin,ymin], [xmax,ymax], [xmin,ymax], [xmax,ymin]

            # apply rotation transform
            corner_points = corner_points @ M.T

            # pick rotated corner (x,y) and reshape to 1 column
            corner_points = corner_points[:, :2].reshape(n, 8)
            # select x lines and y lines
            corner_x = corner_points[:, [0, 2, 4, 6]]
            corner_y = corner_points[:, [1, 3, 5, 7]]

            # create new bounding boxes according to rotated corner points boundary
            rotate_boxes = np.concatenate((corner_x.min(axis=-1), corner_y.min(axis=-1), corner_x.max(axis=-1), corner_y.max(axis=-1))).reshape(4, n).T

            # clip boxes with image size
            # NOTE: use (width-1) & (height-1) as max to avoid index overflow
            rotate_boxes[:, [0, 2]] = rotate_boxes[:, [0, 2]].clip(0, width-1)
            rotate_boxes[:, [1, 3]] = rotate_boxes[:, [1, 3]].clip(0, height-1)

            # apply new boxes
            boxes[:, :4] = rotate_boxes

            # filter candidates
            #i = box_candidates(box1=boxes[:, :4].T * scale, box2=rotate_boxes.T)
            #boxes = boxes[i]
            #boxes[:, :4] = rotate_boxes[i]

        image = Image.fromarray(img)

    return image, boxes



def box_candidates(box1, box2, wh_thr=2, ar_thr=20, area_thr=0.1):  # box1(4,n), box2(4,n)
    # Compute candidate boxes: box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + 1e-16), h2 / (w2 + 1e-16))  # aspect ratio
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + 1e-16) > area_thr) & (ar < ar_thr)  # candidates



class Grid(object):
    def __init__(self, d1, d2, rotate=360, ratio=0.5, mode=1, prob=1.):
        self.d1 = d1
        self.d2 = d2
        self.rotate = rotate
        self.ratio = ratio
        self.mode=mode
        self.st_prob = self.prob = prob

    def set_prob(self, epoch, max_epoch):
        self.prob = self.st_prob * min(1, epoch / max_epoch)

    def __call__(self, img):
        h = img.size[1]
        w = img.size[0]

        if np.random.rand() > self.prob:
            return img, np.ones((h, w), np.float32)

        # 1.5 * h, 1.5 * w works fine with the squared images
        # But with rectangular input, the mask might not be able to recover back to the input image shape
        # A square mask with edge length equal to the diagnoal of the input image
        # will be able to cover all the image spot after the rotation. This is also the minimum square.
        hh = math.ceil((math.sqrt(h*h + w*w)))

        d = np.random.randint(self.d1, self.d2)
        #d = self.d

        # maybe use ceil? but i guess no big difference
        self.l = math.ceil(d*self.ratio)

        mask = np.ones((hh, hh), np.float32)
        st_h = np.random.randint(d)
        st_w = np.random.randint(d)
        for i in range(-1, hh//d+1):
                s = d*i + st_h
                t = s+self.l
                s = max(min(s, hh), 0)
                t = max(min(t, hh), 0)
                mask[s:t,:] *= 0
        for i in range(-1, hh//d+1):
                s = d*i + st_w
                t = s+self.l
                s = max(min(s, hh), 0)
                t = max(min(t, hh), 0)
                mask[:,s:t] *= 0
        r = np.random.randint(self.rotate)
        mask = Image.fromarray(np.uint8(mask))
        mask = mask.rotate(r)
        mask = np.asarray(mask)
        mask = mask[(hh-h)//2:(hh-h)//2+h, (hh-w)//2:(hh-w)//2+w]

        if self.mode == 1:
            mask = 1-mask

        #mask = mask.expand_as(img)
        img = np.array(img) * np.expand_dims(mask, -1)

        return Image.fromarray(img), mask


def random_gridmask(image, boxes, prob=0.2):
    """
    Random add GridMask augment for image

    reference:
        https://arxiv.org/abs/2001.04086
        https://github.com/Jia-Research-Lab/GridMask/blob/master/imagenet_grid/utils/grid.py

    # Arguments
        image: origin image for GridMask
            PIL Image object containing image data
        boxes: Ground truth object bounding boxes,
            numpy array of shape (num_boxes, 5),
            box format (xmin, ymin, xmax, ymax, cls_id).

        prob: probability for GridMask,
            scalar to control the GridMask probability.

    # Returns
        image: adjusted PIL Image object.
        boxes: rotated bounding box numpy array
    """
    grid = Grid(d1=image.size[0]//7, d2=image.size[0]//3, rotate=360, ratio=0.5, prob=prob)
    image, mask = grid(image)

    n = len(boxes)
    if n:
        # check box width and height to discard invalid box
        boxes_w = boxes[:, 2] - boxes[:, 0]
        boxes_h = boxes[:, 3] - boxes[:, 1]
        boxes = boxes[np.logical_and(boxes_w>1, boxes_h>1)] # discard invalid box

        new_boxes = []
        # filter out box which is heavily masked
        for box in boxes:
            xmin, ymin, xmax, ymax = box[:4]
            box_mask = mask[ymin:ymax, xmin:xmax]
            box_area = (xmax - xmin) * (ymax - ymin)
            box_valid_area = box_mask.sum()
            if box_valid_area > (box_area * 0.3): # only keep box when valid_area > 30%
                new_boxes.append(box)

        boxes = np.vstack(new_boxes) if len(new_boxes) >= 1 else np.array([])

    return image, boxes


def merge_mosaic_bboxes(bboxes, crop_x, crop_y, image_size):
    # adjust & merge mosaic samples bboxes as following area order:
    # -----------
    # |     |   |
    # |  0  | 3 |
    # |     |   |
    # -----------
    # |  1  | 2 |
    # -----------
    assert bboxes.shape[0] == 4, 'mosaic sample number should be 4'
    max_boxes = bboxes.shape[1]
    height, width = image_size
    merge_bbox = []
    for i in range(bboxes.shape[0]):
        for box in bboxes[i]:
            x_min, y_min, x_max, y_max = box[0], box[1], box[2], box[3]

            if i == 0: # bboxes[0] is for top-left area
                if y_min > crop_y or x_min > crop_x:
                    continue
                if y_max > crop_y and y_min < crop_y:
                    y_max = crop_y
                if x_max > crop_x and x_min < crop_x:
                    x_max = crop_x

            if i == 1: # bboxes[1] is for bottom-left area
                if y_max < crop_y or x_min > crop_x:
                    continue
                if y_max > crop_y and y_min < crop_y:
                    y_min = crop_y
                if x_max > crop_x and x_min < crop_x:
                    x_max = crop_x

            if i == 2: # bboxes[2] is for bottom-right area
                if y_max < crop_y or x_max < crop_x:
                    continue
                if y_max > crop_y and y_min < crop_y:
                    y_min = crop_y
                if x_max > crop_x and x_min < crop_x:
                    x_min = crop_x

            if i == 3: # bboxes[3] is for top-right area
                if y_min > crop_y or x_max < crop_x:
                    continue
                if y_max > crop_y and y_min < crop_y:
                    y_max = crop_y
                if x_max > crop_x and x_min < crop_x:
                    x_min = crop_x

            if abs(x_max-x_min) < max(10, width*0.01) or abs(y_max-y_min) < max(10, height*0.01):
                #if the adjusted bbox is too small, bypass it
                continue

            merge_bbox.append([x_min, y_min, x_max, y_max, box[4]])

    if len(merge_bbox) > max_boxes:
        merge_bbox = merge_bbox[:max_boxes]

    box_data = np.zeros((max_boxes,5))
    if len(merge_bbox) > 0:
        box_data[:len(merge_bbox)] = merge_bbox
    return box_data


def random_mosaic_augment(image_data, boxes_data, prob=.1):
    """
    Random add mosaic augment on batch images and boxes, from YOLOv4

    reference:
        https://github.com/klauspa/Yolov4-tensorflow/blob/master/data.py
        https://github.com/clovaai/CutMix-PyTorch
        https://github.com/AlexeyAB/darknet

    # Arguments
        image_data: origin images for mosaic augment
            numpy array for normalized batch image data
        boxes_data: origin bboxes for mosaic augment
            numpy array for batch bboxes
        prob: probability for augment ,
            scalar to control the augment probability.

    # Returns
        image_data: augmented batch image data.
        boxes_data: augmented batch bboxes data.
    """
    do_augment = rand() < prob
    if not do_augment:
        return image_data, boxes_data
    else:
        batch_size = len(image_data)
        assert batch_size >= 4, 'mosaic augment need batch size >= 4'

        def get_mosaic_samples():
            # random select 4 images from batch as mosaic samples
            random_index = random.sample(list(range(batch_size)), 4)

            random_images = []
            random_bboxes = []
            for idx in random_index:
                random_images.append(image_data[idx])
                random_bboxes.append(boxes_data[idx])
            return random_images, np.array(random_bboxes)

        min_offset = 0.2
        new_images = []
        new_boxes = []
        height, width = image_data[0].shape[:2]
        #each batch has batch_size images, so we also need to
        #generate batch_size mosaic images
        for i in range(batch_size):
            images, bboxes = get_mosaic_samples()

            #crop_x = np.random.randint(int(width*min_offset), int(width*(1 - min_offset)))
            #crop_y = np.random.randint(int(height*min_offset), int(height*(1 - min_offset)))
            crop_x = int(random.uniform(int(width*min_offset), int(width*(1-min_offset))))
            crop_y = int(random.uniform(int(height*min_offset), int(height*(1 - min_offset))))

            merged_boxes = merge_mosaic_bboxes(bboxes, crop_x, crop_y, image_size=(height, width))
            #no valid bboxes, drop this loop
            #if merged_boxes is None:
                #i = i - 1
                #continue

            # crop out selected area as following mosaic sample images order:
            # -----------
            # |     |   |
            # |  0  | 3 |
            # |     |   |
            # -----------
            # |  1  | 2 |
            # -----------
            area_0 = images[0][:crop_y, :crop_x, :]
            area_1 = images[1][crop_y:, :crop_x, :]
            area_2 = images[2][crop_y:, crop_x:, :]
            area_3 = images[3][:crop_y, crop_x:, :]

            #merge selected area to new image
            area_left = np.concatenate([area_0, area_1], axis=0)
            area_right = np.concatenate([area_3, area_2], axis=0)
            merged_image = np.concatenate([area_left, area_right], axis=1)

            new_images.append(merged_image)
            new_boxes.append(merged_boxes)

        new_images = np.stack(new_images)
        new_boxes = np.array(new_boxes)
        return new_images, new_boxes


def random_mosaic_augment_v5(image_data, boxes_data, prob=.1):
    """
    Random mosaic augment from YOLOv5 implementation

    reference:
        https://github.com/ultralytics/yolov5/blob/develop/utils/datasets.py

    # Arguments
        image_data: origin images for mosaic augment
            numpy array for normalized batch image data
        boxes_data: origin bboxes for mosaic augment
            numpy array for batch bboxes
        prob: probability for augment ,
            scalar to control the augment probability.

    # Returns
        image_data: augmented batch image data.
        boxes_data: augmented batch bboxes data.
    """
    do_augment = rand() < prob
    if not do_augment:
        return image_data, boxes_data
    else:
        batch_size = len(image_data)
        assert batch_size >= 4, 'mosaic augment need batch size >= 4'

        def get_mosaic_samples():
            # random select 4 images from batch as mosaic samples
            random_index = random.sample(list(range(batch_size)), 4)

            random_images = []
            random_bboxes = []
            for idx in random_index:
                random_images.append(image_data[idx])
                random_bboxes.append(boxes_data[idx])
            return random_images, np.array(random_bboxes)

        new_images = []
        new_boxes = []
        input_height, input_width, input_channel = image_data[0].shape[:3]

        #each batch has batch_size images, so we also need to
        #generate batch_size mosaic images
        for j in range(batch_size):
            images, bboxes = get_mosaic_samples()

            # mosaic center x, y
            mosaic_border = (input_width//2, input_height//2)
            x_center = int(random.uniform(mosaic_border[0], input_width*2-mosaic_border[0]))
            y_center = int(random.uniform(mosaic_border[1], input_height*2-mosaic_border[1]))

            # create large mosaic image with size (input_height*2, input_width*2)
            mosaic_image = np.full((input_height*2, input_width*2, input_channel), 128, dtype=np.uint8)
            mosaic_bbox = []
            max_boxes = bboxes.shape[1]

            for i in range(4):
                image = images[i]
                bbox = bboxes[i]
                height, width = image.shape[:2]

                # calculate padding area in each src & target image
                if i == 0:  # top left
                    #x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
                    #x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
                    xmin_target = max(x_center - width, 0)
                    ymin_target = max(y_center - height, 0)
                    xmax_target = x_center
                    ymax_target = y_center

                    xmin_src = width - (xmax_target - xmin_target)
                    ymin_src = height - (ymax_target - ymin_target)
                    xmax_src = width
                    ymax_src = height
                elif i == 1:  # top right
                    #x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                    #x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
                    xmin_target = x_center
                    ymin_target = max(y_center - height, 0)
                    xmax_target = min(x_center + width, width * 2)
                    ymax_target = y_center

                    xmin_src = 0
                    ymin_src = height - (ymax_target - ymin_target)
                    xmax_src = min(width, xmax_target - xmin_target)
                    ymax_src = height
                elif i == 2:  # bottom left
                    #x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                    #x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
                    xmin_target = max(x_center - width, 0)
                    ymin_target = y_center
                    xmax_target = x_center
                    ymax_target = min(height * 2, y_center + height)

                    xmin_src = width - (xmax_target - xmin_target)
                    ymin_src = 0
                    xmax_src = width
                    ymax_src = min(ymax_target - ymin_target, height)
                elif i == 3:  # bottom right
                    #x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                    #x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
                    xmin_target = x_center
                    ymin_target = y_center
                    xmax_target = min(x_center + width, width * 2)
                    ymax_target = min(height * 2, y_center + height)

                    xmin_src = 0
                    ymin_src = 0
                    xmax_src = min(width, xmax_target - xmin_target)
                    ymax_src = min(ymax_target - ymin_target, height)

                # padding src image to corresponding mosaic area
                mosaic_image[ymin_target:ymax_target, xmin_target:xmax_target] = image[ymin_src:ymax_src, xmin_src:xmax_src]
                # padding width & height for bbox
                padding_width = xmin_target - xmin_src
                padding_height = ymin_target - ymin_src

                # adjust bbox to new mosaic image, with padding width & height
                for box in bbox:
                    # break loop when reach invalid box line
                    if (box[:4] == 0).all():
                        break
                    x_min, y_min, x_max, y_max = box[0], box[1], box[2], box[3]
                    x_min += padding_width
                    y_min += padding_height
                    x_max += padding_width
                    y_max += padding_height

                    mosaic_bbox.append([x_min, y_min, x_max, y_max, box[4]])

            if len(mosaic_bbox) > max_boxes:
                mosaic_bbox = mosaic_bbox[:max_boxes]

            box_data = np.zeros((max_boxes, 5))
            if len(mosaic_bbox) > 0:
                box_data[:len(mosaic_bbox)] = mosaic_bbox

            # clip boxes to valid image area
            np.clip(box_data[..., 0], 0, input_width*2-1, out=box_data[..., 0])
            np.clip(box_data[..., 1], 0, input_height*2-1, out=box_data[..., 1])
            np.clip(box_data[..., 2], 0, input_width*2-1, out=box_data[..., 2])
            np.clip(box_data[..., 3], 0, input_height*2-1, out=box_data[..., 3])

            # resize image & box back to input shape
            mosaic_image = cv2.resize(mosaic_image, (input_width, input_height), cv2.INTER_AREA)
            box_data[..., :4] //= 2

            new_images.append(mosaic_image)
            new_boxes.append(box_data)

        new_images = np.stack(new_images)
        new_boxes = np.array(new_boxes)
        return new_images, new_boxes


def merge_cutmix_bboxes(bboxes, cut_xmin, cut_ymin, cut_xmax, cut_ymax, image_size):
    # adjust & merge cutmix samples bboxes as following area order:
    # -----------------
    # |               |
    # |  0            |
    # |       ____    |
    # |      |    |   |
    # |      |  1 |   |
    # |      |____|   |
    # |               |
    # -----------------
    assert bboxes.shape[0] == 2, 'cutmix sample number should be 2'
    max_boxes = bboxes.shape[1]
    height, width = image_size
    merge_bbox = []
    for i in range(bboxes.shape[0]):
        for box in bboxes[i]:
            x_min, y_min, x_max, y_max = box[0], box[1], box[2], box[3]

            if i == 0: # bboxes[0] is for background area
                if x_min > cut_xmin and x_max < cut_xmax and y_min > cut_ymin and y_max < cut_ymax:
                    # all box in padding area, drop it
                    continue
                elif x_min > cut_xmax or x_max < cut_xmin or y_min > cut_ymax or y_max < cut_ymin:
                    # all box in background area, do nothing
                    pass
                else:
                    # TODO: currently it is a BAD strategy to adjust box in background area, so seems
                    # CutMix could not be used directly in object detection data augment
                    if x_max > cut_xmin and x_max < cut_xmax:
                        x_max = cut_xmin
                    elif x_min > cut_xmin and x_min < cut_xmax:
                        x_min = cut_xmax
                    if y_max > cut_ymin and y_max < cut_ymax:
                        y_max = cut_ymin
                    elif y_min > cut_ymin and y_min < cut_ymax:
                        y_min = cut_ymax

            if i == 1: # bboxes[1] is for padding area
                if x_min > cut_xmin and x_max < cut_xmax and y_min > cut_ymin and y_max < cut_ymax :
                    # all box in padding area, do nothing
                    pass
                elif x_min > cut_xmax or x_max < cut_xmin or y_min > cut_ymax or y_max < cut_ymin:
                    # all box in background area, drop it
                    continue
                else:
                    # limit box in padding area
                    if x_max > cut_xmax:
                        x_max = cut_xmax
                    if x_min < cut_xmin:
                        x_min = cut_xmin
                    if y_max > cut_ymax:
                        y_max = cut_ymax
                    if y_min < cut_ymin:
                        y_min = cut_ymin

            if abs(x_max-x_min) < max(10, width*0.01) or abs(y_max-y_min) < max(10, height*0.01):
                #if the adjusted bbox is too small, bypass it
                continue

            merge_bbox.append([x_min, y_min, x_max, y_max, box[4]])

    if len(merge_bbox) > max_boxes:
        merge_bbox = merge_bbox[:max_boxes]

    box_data = np.zeros((max_boxes,5))
    if len(merge_bbox) > 0:
        box_data[:len(merge_bbox)] = merge_bbox
    return box_data


def random_cutmix_augment(image_data, boxes_data, prob=.1):
    """
    Random add cutmix augment on batch images and boxes

    Warning: currently it is a BAD strategy and could not be used in object detection data augment

    # Arguments
        image_data: origin images for cutmix augment
            numpy array for normalized batch image data
        boxes_data: origin bboxes for cutmix augment
            numpy array for batch bboxes
        prob: probability for augment,
            scalar to control the augment probability.

    # Returns
        image_data: augmented batch image data.
        boxes_data: augmented batch bboxes data.
    """
    do_augment = rand() < prob
    if not do_augment:
        return image_data, boxes_data
    else:
        batch_size = len(image_data)
        assert batch_size >= 2, 'cutmix augment need batch size >= 2'

        def get_cutmix_samples():
            # random select 2 images from batch as cutmix samples
            random_index = random.sample(list(range(batch_size)), 2)

            random_images = []
            random_bboxes = []
            for idx in random_index:
                random_images.append(image_data[idx])
                random_bboxes.append(boxes_data[idx])
            return random_images, np.array(random_bboxes)

        def get_cutmix_box(image_size, lamda):
            height, width = image_size
            min_offset = 0.1

            # get width and height for cut area
            cut_rat = np.sqrt(1. - lamda)
            cut_w = np.int(width * cut_rat)
            cut_h = np.int(height * cut_rat)

            # get center point for cut area
            center_x = np.random.randint(width)
            center_y = np.random.randint(height)

            # limit cut area to allowed image size
            cut_xmin = np.clip(center_x - cut_w // 2, int(width*min_offset), int(width*(1-min_offset)))
            cut_ymin = np.clip(center_y - cut_h // 2, int(height*min_offset), int(height*(1-min_offset)))
            cut_xmax = np.clip(center_x + cut_w // 2, int(width*min_offset), int(width*(1-min_offset)))
            cut_ymax = np.clip(center_y + cut_h // 2, int(height*min_offset), int(height*(1-min_offset)))

            return cut_xmin, cut_ymin, cut_xmax, cut_ymax

        new_images = []
        new_boxes = []
        height, width = image_data[0].shape[:2]
        #each batch has batch_size images, so we also need to
        #generate batch_size mosaic images
        for i in range(batch_size):
            images, bboxes = get_cutmix_samples()
            lamda = np.random.beta(5, 5)

            cut_xmin, cut_ymin, cut_xmax, cut_ymax = get_cutmix_box(image_size=(height, width), lamda=lamda)
            merged_boxes = merge_cutmix_bboxes(bboxes, cut_xmin, cut_ymin, cut_xmax, cut_ymax, image_size=(height, width))
            #no valid bboxes, drop this loop
            #if merged_boxes is None:
                #i = i - 1
                #continue

            # crop and pad selected area as following cutmix sample images order:
            # -----------------
            # |               |
            # |  0            |
            # |       ____    |
            # |      |    |   |
            # |      |  1 |   |
            # |      |____|   |
            # |               |
            # -----------------
            bg_image = images[0].copy()
            pad_image = images[1].copy()

            #crop and pad selected area to background image
            bg_image[cut_ymin:cut_ymax, cut_xmin:cut_xmax, :] = pad_image[cut_ymin:cut_ymax, cut_xmin:cut_xmax, :]
            merged_image = bg_image

            new_images.append(merged_image)
            new_boxes.append(merged_boxes)

        new_images = np.stack(new_images)
        new_boxes = np.array(new_boxes)
        return new_images, new_boxes


def normalize_image(image):
    """
    normalize image array from 0 ~ 255
    to 0.0 ~ 1.0

    # Arguments
        image: origin input image
            numpy image array with dtype=float, 0.0 ~ 255.0

    # Returns
        image: numpy image array with dtype=float, 0.0 ~ 1.0
    """
    image = image.astype(np.float32) / 255.0

    return image


def denormalize_image(image):
    """
    Denormalize image array from 0.0 ~ 1.0
    to 0 ~ 255

    # Arguments
        image: normalized image array with dtype=float, -1.0 ~ 1.0

    # Returns
        image: numpy image array with dtype=uint8, 0 ~ 255
    """
    image = (image * 255.0).astype(np.uint8)

    return image


def preprocess_image(image, model_input_shape):
    """
    Prepare model input image data with letterbox
    resize, normalize and dim expansion

    # Arguments
        image: origin input image
            PIL Image object containing image data
        model_input_shape: model input image shape
            tuple of format (height, width).

    # Returns
        image_data: numpy array of image data for model input.
    """
    #resized_image = cv2.resize(image, model_input_shape[::-1], cv2.INTER_AREA)
    resized_image = letterbox_resize(image, model_input_shape[::-1])
    image_data = np.asarray(resized_image).astype('float32')
    image_data = normalize_image(image_data)
    image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
    return image_data

