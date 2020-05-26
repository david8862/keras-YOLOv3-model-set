#!/usr/bin/python3
# -*- coding=utf-8 -*-
import numpy as np
import copy
from scipy.special import expit, softmax


def yolo3_head(predictions, anchors, num_classes, input_dims):
    """
    YOLO Head to process predictions from YOLO models

    :param num_classes: Total number of classes
    :param anchors: YOLO style anchor list for bounding box assignment
    :param input_dims: Input dimensions of the image
    :param predictions: A list of three tensors with shape (N, 19, 19, 255), (N, 38, 38, 255) and (N, 76, 76, 255)
    :return: A tensor with the shape (N, num_boxes, 85)
    """
    assert len(predictions) == len(anchors)//3, 'anchor numbers does not match prediction.'

    if len(predictions) == 3: # assume 3 set of predictions is YOLOv3
        anchor_mask = [[6,7,8], [3,4,5], [0,1,2]]
    elif len(predictions) == 2: # 2 set of predictions is YOLOv3-tiny
        anchor_mask = [[3,4,5], [0,1,2]]
    else:
        raise ValueError('Unsupported prediction length: {}'.format(len(predictions)))

    results = []
    for i, prediction in enumerate(predictions):
        results.append(_yolo3_head(prediction, num_classes, anchors[anchor_mask[i]], input_dims))

    return np.concatenate(results, axis=1)


def _yolo3_head(prediction, num_classes, anchors, input_dims):
    batch_size = np.shape(prediction)[0]
    num_anchors = len(anchors)

    grid_size = np.shape(prediction)[1:3]
    #check if stride on height & width are same
    assert input_dims[0]//grid_size[0] == input_dims[1]//grid_size[1], 'model stride mismatch.'
    stride = input_dims[0] // grid_size[0]

    prediction = np.reshape(prediction,
                            (batch_size, num_anchors * grid_size[0] * grid_size[1], num_classes + 5))

    ################################
    # generate x_y_offset grid map
    grid_y = np.arange(grid_size[0])
    grid_x = np.arange(grid_size[1])
    x_offset, y_offset = np.meshgrid(grid_x, grid_y)

    x_offset = np.reshape(x_offset, (-1, 1))
    y_offset = np.reshape(y_offset, (-1, 1))

    x_y_offset = np.concatenate((x_offset, y_offset), axis=1)
    x_y_offset = np.tile(x_y_offset, (1, num_anchors))
    x_y_offset = np.reshape(x_y_offset, (-1, 2))
    x_y_offset = np.expand_dims(x_y_offset, 0)
    ################################

    # Log space transform of the height and width
    #anchors = [(a[0] / stride, a[1] / stride) for a in anchors]
    anchors = np.tile(anchors, (grid_size[0] * grid_size[1], 1))
    anchors = np.expand_dims(anchors, 0)

    box_wh = np.exp(prediction[:, :, 2:4]) * anchors
    # Resize detection map back to the input image size
    #box_wh *= stride


    box_xy = expit(prediction[:, :, :2]) + x_y_offset  # t_x (box x and y coordinates)
    box_xy *= stride
    # Convert centoids to top left coordinates
    box_xy -= box_wh / 2


    # Sigmoid objectness scores
    objectness = expit(prediction[:, :, 4])  # p_o (objectness score)
    objectness = np.expand_dims(objectness, 2)  # To make the same number of values for axis 0 and 1
    # Sigmoid class scores
    class_scores = expit(prediction[:, :, 5:])
    #class_scores = softmax(prediction[:, :, 5:], axis=-1)

    return np.concatenate([box_xy, box_wh, objectness, class_scores], axis=2)


def yolo3_postprocess_np(yolo_outputs, image_shape, anchors, num_classes, model_image_size, max_boxes=100, confidence=0.1, iou_threshold=0.4):
    predictions = yolo3_head(yolo_outputs, anchors, num_classes, input_dims=model_image_size)

    boxes, classes, scores = yolo3_handle_predictions(predictions,
                                                max_boxes=max_boxes,
                                                confidence=confidence,
                                                iou_threshold=iou_threshold)
    boxes = yolo3_adjust_boxes(boxes, image_shape, model_image_size)

    return boxes, classes, scores


def yolo3_handle_predictions(predictions, max_boxes=100, confidence=0.1, iou_threshold=0.4):
    boxes = predictions[:, :, :4]
    box_confidences = np.expand_dims(predictions[:, :, 4], -1)
    box_class_probs = predictions[:, :, 5:]

    box_scores = box_confidences * box_class_probs
    box_classes = np.argmax(box_scores, axis=-1)
    box_class_scores = np.max(box_scores, axis=-1)
    pos = np.where(box_class_scores >= confidence)

    boxes = boxes[pos]
    classes = box_classes[pos]
    scores = box_class_scores[pos]

    # Boxes, Classes and Scores returned from NMS
    n_boxes, n_classes, n_scores = nms_boxes(boxes, classes, scores, iou_threshold, confidence=confidence, use_diou=False, is_soft=False)

    if n_boxes:
        boxes = np.concatenate(n_boxes)
        classes = np.concatenate(n_classes)
        scores = np.concatenate(n_scores)
        boxes, classes, scores = filter_boxes(boxes, classes, scores, max_boxes)

        return boxes, classes, scores

    else:
        return [], [], []


def filter_boxes(boxes, classes, scores, max_boxes):
    '''
    Sort the prediction boxes according to score
    and only pick top "max_boxes" ones
    '''
    # sort result according to scores
    sorted_indices = np.argsort(scores)
    sorted_indices = sorted_indices[::-1]
    nboxes = boxes[sorted_indices]
    nclasses = classes[sorted_indices]
    nscores = scores[sorted_indices]

    # only pick max_boxes
    nboxes = nboxes[:max_boxes]
    nclasses = nclasses[:max_boxes]
    nscores = nscores[:max_boxes]

    return nboxes, nclasses, nscores


def box_iou(boxes):
    """
    Calculate iou on box array

    Parameters
    ----------
    boxes: bbox numpy array, shape=(N, 4), xywh
           x,y are top left coordinates

    Returns
    -------
    iou: numpy array, shape=(N-1,)
         IoU value of boxes[:-1] with boxes[-1]
    """
    # get box coordinate and area
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2]
    h = boxes[:, 3]
    areas = w * h

    # check IoU
    inter_xmin = np.maximum(x[:-1], x[-1])
    inter_ymin = np.maximum(y[:-1], y[-1])
    inter_xmax = np.minimum(x[:-1] + w[:-1], x[-1] + w[-1])
    inter_ymax = np.minimum(y[:-1] + h[:-1], y[-1] + h[-1])

    inter_w = np.maximum(0.0, inter_xmax - inter_xmin + 1)
    inter_h = np.maximum(0.0, inter_ymax - inter_ymin + 1)

    inter = inter_w * inter_h
    iou = inter / (areas[:-1] + areas[-1] - inter)
    return iou


def box_diou(boxes):
    """
    Calculate diou on box array
    Reference Paper:
        "Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression"
        https://arxiv.org/abs/1911.08287

    Parameters
    ----------
    boxes: bbox numpy array, shape=(N, 4), xywh
           x,y are top left coordinates

    Returns
    -------
    diou: numpy array, shape=(N-1,)
         IoU value of boxes[:-1] with boxes[-1]
    """
    # get box coordinate and area
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2]
    h = boxes[:, 3]
    areas = w * h

    # check IoU
    inter_xmin = np.maximum(x[:-1], x[-1])
    inter_ymin = np.maximum(y[:-1], y[-1])
    inter_xmax = np.minimum(x[:-1] + w[:-1], x[-1] + w[-1])
    inter_ymax = np.minimum(y[:-1] + h[:-1], y[-1] + h[-1])

    inter_w = np.maximum(0.0, inter_xmax - inter_xmin + 1)
    inter_h = np.maximum(0.0, inter_ymax - inter_ymin + 1)

    inter = inter_w * inter_h
    iou = inter / (areas[:-1] + areas[-1] - inter)

    # box center distance
    x_center = x + w/2
    y_center = y + h/2
    center_distance = np.power(x_center[:-1] - x_center[-1], 2) + np.power(y_center[:-1] - y_center[-1], 2)

    # get enclosed area
    enclose_xmin = np.minimum(x[:-1], x[-1])
    enclose_ymin = np.minimum(y[:-1], y[-1])
    enclose_xmax = np.maximum(x[:-1] + w[:-1], x[-1] + w[-1])
    enclose_ymax = np.maximum(x[:-1] + w[:-1], x[-1] + w[-1])
    enclose_w = np.maximum(0.0, enclose_xmax - enclose_xmin + 1)
    enclose_h = np.maximum(0.0, enclose_ymax - enclose_ymin + 1)
    # get enclosed diagonal distance
    enclose_diagonal = np.power(enclose_w, 2) + np.power(enclose_h, 2)
    # calculate DIoU, add epsilon in denominator to avoid dividing by 0
    diou = iou - 1.0 * (center_distance) / (enclose_diagonal + np.finfo(float).eps)

    return diou


def nms_boxes(boxes, classes, scores, iou_threshold, confidence=0.1, use_diou=False, is_soft=False, use_exp=False, sigma=0.5):
    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        # handle data for one class
        inds = np.where(classes == c)
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]

        # make a data copy to avoid breaking
        # during nms operation
        b_nms = copy.deepcopy(b)
        c_nms = copy.deepcopy(c)
        s_nms = copy.deepcopy(s)

        while len(s_nms) > 0:
            # pick the max box and store, here
            # we also use copy to persist result
            i = np.argmax(s_nms, axis=-1)
            nboxes.append(copy.deepcopy(b_nms[i]))
            nclasses.append(copy.deepcopy(c_nms[i]))
            nscores.append(copy.deepcopy(s_nms[i]))

            # swap the max line and last line
            b_nms[[i,-1],:] = b_nms[[-1,i],:]
            c_nms[[i,-1]] = c_nms[[-1,i]]
            s_nms[[i,-1]] = s_nms[[-1,i]]

            if use_diou:
                iou = box_diou(b_nms)
            else:
                iou = box_iou(b_nms)

            # drop the last line since it has been record
            b_nms = b_nms[:-1]
            c_nms = c_nms[:-1]
            s_nms = s_nms[:-1]

            if is_soft:
                # Soft-NMS
                if use_exp:
                    # score refresh formula:
                    # score = score * exp(-(iou^2)/sigma)
                    s_nms = s_nms * np.exp(-(iou * iou) / sigma)
                else:
                    # score refresh formula:
                    # score = score * (1 - iou) if iou > threshold
                    depress_mask = np.where(iou > iou_threshold)[0]
                    s_nms[depress_mask] = s_nms[depress_mask]*(1-iou[depress_mask])
                keep_mask = np.where(s_nms >= confidence)[0]
            else:
                # normal Hard-NMS
                keep_mask = np.where(iou <= iou_threshold)[0]

            # keep needed box for next loop
            b_nms = b_nms[keep_mask]
            c_nms = c_nms[keep_mask]
            s_nms = s_nms[keep_mask]

    # reformat result for output
    nboxes = [np.array(nboxes)]
    nclasses = [np.array(nclasses)]
    nscores = [np.array(nscores)]
    return nboxes, nclasses, nscores


def yolo3_adjust_boxes(boxes, img_shape, model_image_size):
    if boxes is None or len(boxes) == 0:
        return []

    # model_image_size & image_shape should be (height, width) format
    model_image_size = np.array(model_image_size, dtype='float32')
    image_shape = np.array(img_shape, dtype='float32')
    height, width = image_shape
    #width, height = image_shape

    new_shape = np.round(image_shape * np.min(model_image_size/image_shape))
    offset = (model_image_size-new_shape)/2.
    # here scale should be a unique value
    # for both width and height
    scale = np.min(image_shape/new_shape)

    #ratio_x = width / model_image_size[1]
    #ratio_y = height / model_image_size[0]

    adjusted_boxes = []
    for box in boxes:
        x, y, w, h = box

        w *= scale
        h *= scale
        xmin = (x - offset[1]) * scale
        ymin = (y - offset[0]) * scale
        xmax = (x - offset[1]) * scale + w
        ymax = (y - offset[0]) * scale + h

        # Rescale box coordinates
        #xmin = int(x * ratio_x)
        #ymin = int(y * ratio_y)
        #xmax = int((x + w) * ratio_x)
        #ymax = int((y + h) * ratio_y)

        ymin = max(0, np.floor(ymin + 0.5).astype('int32'))
        xmin = max(0, np.floor(xmin + 0.5).astype('int32'))
        ymax = min(height, np.floor(ymax + 0.5).astype('int32'))
        xmax = min(width, np.floor(xmax + 0.5).astype('int32'))
        adjusted_boxes.append([xmin,ymin,xmax,ymax])

    return np.array(adjusted_boxes,dtype=np.int32)

