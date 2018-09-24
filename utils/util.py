from Bbox import Bbox
import numpy as np


def compute_iou(box1, box2):

    intersect_w = _interval_overlap(
        [box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
    intersect_h = _interval_overlap(
        [box1.ymin, box1.ymax], [box2.ymin, box2.ymax])

    intersect = intersect_w * intersect_h

    w1, h1 = box1.xmax - box1.xmin, box1.ymax - box1.ymin
    w2, h2 = box2.xmax - box2.xmin, box2.ymax - box2.ymin

    union = w1 * h1 + w2 * h2 - intersect

    return float(intersect) / union


def _interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b

    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2, x4) - x1
    else:
        if x2 < x3:
            return 0
        else:
            return min(x2, x4) - x3


def get_bbox_vals(cords, w, h):

    cords[0] = cords[0] * w
    cords[1] = cords[1] * h
    cords[2] = cords[2] * w
    cords[3] = cords[3] * h

    xmax = (2 * cords[0] + w) / 2
    xmin = xmax - cords[2]
    ymax = (2 * cords[1] + h) / 2
    ymin = ymax - cords[3]

    return [xmin, ymin, xmax, ymax]


def convert_to_bbox(labels_all):
    objs = []
    for label in labels_all:
        label = label.split("\n")
        x, y, w, h = label[1].split(" ")
        objs.append(Bbox(int(x), int(y), int(w), int(h), int(label[0])))

    return objs
