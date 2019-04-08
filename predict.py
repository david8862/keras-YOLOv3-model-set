import cv2
import numpy as np

from yolo_head import yolo_head


def predict(model, orig, config, confidence=0.5, iou_threshold=0.4):
    image, image_data = preprocess_image(orig, model_image_size=(config['width'], config['height']))

    boxes, classes, scores = handle_predictions(model.predict([image_data]),
                                                confidence=confidence,
                                                iou_threshold=iou_threshold)

    draw_boxes(image, boxes, classes, scores, config)

    return np.array(image)


def predict_with_yolo_head(model, orig, config, confidence=0.5, iou_threshold=0.4):
    image, image_data = preprocess_image(orig, model_image_size=(config['width'], config['height']))

    predictions = yolo_head(model.predict([image_data]), num_classes=80,
                            input_dims=(config['width'], config['height']))

    boxes, classes, scores = handle_predictions(predictions,
                                                confidence=confidence,
                                                iou_threshold=iou_threshold)

    draw_boxes(image, boxes, classes, scores, config)

    return np.array(image)


def handle_predictions(predictions, confidence=0.6, iou_threshold=0.5):
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
    n_boxes, n_classes, n_scores = nms_boxes(boxes, classes, scores, iou_threshold)

    if n_boxes:
        boxes = np.concatenate(n_boxes)
        classes = np.concatenate(n_classes)
        scores = np.concatenate(n_scores)

        return boxes, classes, scores

    else:
        return [], [], []


def preprocess_image(img_arr, model_image_size):
    image = img_arr.astype('uint8')
    resized_image = cv2.resize(image, tuple(reversed(model_image_size)), cv2.INTER_AREA)
    image_data = resized_image.astype('float32')
    image_data /= 255.
    image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
    return image, image_data


def nms_boxes(boxes, classes, scores, iou_threshold):
    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]

        x = b[:, 0]
        y = b[:, 1]
        w = b[:, 2]
        h = b[:, 3]

        areas = w * h
        order = s.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x[i], x[order[1:]])
            yy1 = np.maximum(y[i], y[order[1:]])
            xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
            yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

            w1 = np.maximum(0.0, xx2 - xx1 + 1)
            h1 = np.maximum(0.0, yy2 - yy1 + 1)

            inter = w1 * h1
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= iou_threshold)[0]
            order = order[inds + 1]

        keep = np.array(keep)

        nboxes.append(b[keep])
        nclasses.append(c[keep])
        nscores.append(s[keep])
    return nboxes, nclasses, nscores


def draw_label(image, text, color, coords):
    font = cv2.FONT_HERSHEY_PLAIN
    font_scale = 1.
    (text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=1)[0]

    padding = 5
    rect_height = text_height + padding * 2
    rect_width = text_width + padding * 2

    (x, y) = coords

    cv2.rectangle(image, (x, y), (x + rect_width, y - rect_height), color, cv2.FILLED)
    cv2.putText(image, text, (x + padding, y - text_height + padding), font,
                fontScale=font_scale,
                color=(255, 255, 255),
                lineType=cv2.LINE_AA)

    return image


def draw_boxes(image, boxes, classes, scores, config):
    if classes is None or len(classes) == 0:
        return

    height, width = image.shape[:2]

    labels = config['labels']
    colors = config['colors']

    ratio_x = width / config['width']
    ratio_y = height / config['height']

    for box, cls, score in zip(boxes, classes, scores):
        x, y, w, h = box

        # Rescale box coordinates
        x1 = int(x * ratio_x)
        y1 = int(y * ratio_y)
        x2 = int((x + w) * ratio_x)
        y2 = int((y + h) * ratio_y)

        print("Class: {}, Score: {}".format(labels[cls], score))
        cv2.rectangle(image, (x1, y1), (x2, y2), colors[cls], 1, cv2.LINE_AA)

        text = '{0} {1:.2f}'.format(labels[cls], score)
        image = draw_label(image, text, colors[cls], (x1, y1))

    cv2.imwrite("out.png", image)
