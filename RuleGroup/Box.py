import os
import cv2
import json
import numpy
import math
from PIL import Image, ImageDraw, ImageFont
import copy
from tqdm import tqdm
from icecream import ic


def get_point(points, threshold):
    count = 0
    points_clean = []
    for point in points:
        if point["score"] > threshold:
            count += 1
            points_clean.append(point)
    return points_clean


def GroupBox(image, keys_raw, hybrids_raw):
    keys = []
    hybrids = []

    for temp in keys_raw.values():
        for point in temp:
            bbox = [point[3], point[4], 6, 6]
            bbox = [float(e) for e in bbox]
            category_id = int(point[2])
            tag = float(point[1])
            score = float(point[0])
            keys.append(
                {"bbox": bbox, "category_id": category_id, "score": score, "tag": tag}
            )

    for temp in hybrids_raw.values():
        for point in temp:
            bbox = [point[3], point[4], 6, 6]
            bbox = [float(e) for e in bbox]
            category_id = int(point[2])
            tag = float(point[1])
            score = float(point[0])
            hybrids.append(
                {"bbox": bbox, "category_id": category_id, "score": score, "tag": tag}
            )

    keys = get_point(keys, 0.4)
    hybrids = get_point(hybrids, 0.4)

    # ic(tls)
    # ic(brs)

    for key in keys:
        bbox = key["bbox"]
        x, y = int(bbox[0]), int(bbox[1])
        cv2.circle(image, (x, y), radius=3, color=(0, 255, 0), thickness=-1)

    for key in hybrids:
        bbox = key["bbox"]
        x, y = int(bbox[0]), int(bbox[1])
        cv2.circle(image, (x, y), radius=3, color=(0, 255, 0), thickness=-1)

    return image
