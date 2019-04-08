import numpy as np
from scipy.special import expit


def yolo_head(predictions, num_classes, input_dims):
    """
    YOLO Head to process predictions from Darknet

    :param num_classes: Total number of classes
    :param input_dims: Input dimensions of the image
    :param predictions: A list of three tensors with shape (N, 19, 19, 255), (N,38, 38, 255) and (N, 76, 76, 255)
    :return: A tensor with the shape (N, num_boxes, 85)
    """

    anchors = [
        [[116, 90], [156, 198], [373, 326]],
        [[30, 61], [62, 45], [59, 119]],
        [[10, 13], [16, 30], [33, 23]]
    ]

    tiny_anchors = [
        [[81, 82], [135, 169], [344, 319]],
        [[10, 14], [23, 27], [37, 58]]
    ]
    results = []

    print(len(predictions))
    if len(predictions) == 3: # assume 3 set of predictions is YOLOv3
        model_anchors = anchors
    elif len(predictions) == 2: # 2 set of predictions is YOLOv3-tiny
        model_anchors = tiny_anchors
    else:
        raise ValueError('Unsupported prediction length: {}'.format(len(predictions)))

    for i, prediction in enumerate(predictions):
        results.append(_yolo_head(prediction, num_classes, model_anchors[i], input_dims))

    return np.concatenate(results, axis=1)


def _yolo_head(prediction, num_classes, anchors, input_dims):
    batch_size = np.shape(prediction)[0]
    stride = input_dims[0] // np.shape(prediction)[1]
    grid_size = input_dims[0] // stride
    num_anchors = len(anchors)

    prediction = np.reshape(prediction,
                            (batch_size, num_anchors * grid_size * grid_size, num_classes + 5))

    box_xy = expit(prediction[:, :, :2])  # t_x (box x and y coordinates)
    objectness = expit(prediction[:, :, 4])  # p_o (objectness score)
    objectness = np.expand_dims(objectness, 2)  # To make the same number of values for axis 0 and 1

    grid = np.arange(grid_size)
    a, b = np.meshgrid(grid, grid)

    x_offset = np.reshape(a, (-1, 1))
    y_offset = np.reshape(b, (-1, 1))

    x_y_offset = np.concatenate((x_offset, y_offset), axis=1)
    x_y_offset = np.tile(x_y_offset, (1, num_anchors))
    x_y_offset = np.reshape(x_y_offset, (-1, 2))
    x_y_offset = np.expand_dims(x_y_offset, 0)

    box_xy += x_y_offset

    # Log space transform of the height and width
    anchors = [(a[0] / stride, a[1] / stride) for a in anchors]
    anchors = np.tile(anchors, (grid_size * grid_size, 1))
    anchors = np.expand_dims(anchors, 0)

    box_wh = np.exp(prediction[:, :, 2:4]) * anchors

    # Sigmoid class scores
    class_scores = expit(prediction[:, :, 5:])

    # Resize detection map back to the input image size
    box_xy *= stride
    box_wh *= stride

    # Convert centoids to top left coordinates
    box_xy -= box_wh / 2

    return np.concatenate([box_xy, box_wh, objectness, class_scores], axis=2)
