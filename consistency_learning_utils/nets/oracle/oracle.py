from hashlib import new
import os
import pandas as pd
import numpy as np
import zarr
from tqdm import tqdm

def bb_intersection_over_union(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def oracle_all(pred, truth):
    selected_pseudo_labels = [0] * len(pred)
    IOU_threshold = 0.5

    for t in truth:
        truth_bbox = [t[0], t[1], t[2], t[3]]
        for p in range(0, len(pred)):
            pred_label = pred[p][4]
            if pred_label == t[4]:
                pred_bbox = [pred[p][0], pred[p][1], pred[p][2], pred[p][3]]
                IOU = bb_intersection_over_union(truth_bbox, pred_bbox)
                if IOU >= IOU_threshold:
                    selected_pseudo_labels[p] = IOU

    return selected_pseudo_labels

def oracle_top(pred, truth):
    selected_pseudo_labels = [0] * len(pred)
    IOU_threshold = 0.5

    for t in truth:
        truth_bbox = [t[0], t[1], t[2], t[3]]
        bestIOU = 0
        best_p = None
        for p in range(0, len(pred)):
            pred_label = pred[p][4]
            if pred_label == t[4]:
                pred_bbox = [pred[p][0], pred[p][1], pred[p][2], pred[p][3]]
                IOU = bb_intersection_over_union(truth_bbox, pred_bbox)
                if IOU >= IOU_threshold:
                    if IOU >= bestIOU:
                        best_p = p
                        bestIOU = IOU

        if best_p:
            selected_pseudo_labels[best_p] = 1

    return selected_pseudo_labels

def get_target(label_path):
    target = []

    with open(label_path) as f:
        for line in f:
            line_arr = [float(t) for t in line.split(' ')]
            label, xmin, ymin, xmax, ymax = line_arr
            label = int(label)
            xmin = int(xmin)
            ymin = int(ymin)
            xmax = int(xmax)
            ymax = int(ymax)

            if ymax <= ymin:
                ymax += 1
            if xmax <= xmin:
                xmax += 1
            box = [xmin, ymin, xmax, ymax, label, 1]
            target.append(box)

    return target


def get_dataset(zarr_path):
    zarr_file = zarr.open(zarr_path, mode='r')
    keys = zarr_file.keys()
    samples = [[[], []], [[], []], [[], []]]
    for key in tqdm(keys):
        if key.split('_')[-1][0] == 'f':
            newKey = key[:len(key) - 5]
            features_key = newKey + '_feat'
            features = zarr_file[features_key]
            cl = int(features[1][0][0])
            conf  = float(features[0][0][0])
            # samples[cl - 1].append(newKey)
            if conf > 0.5:
                samples[cl - 1][1].append(newKey)
            else:
                samples[cl - 1][0].append(newKey)
    return samples


def get_max_IOU(label_file, bbox):
    target = get_target(label_file)
    max_iou = 0
    for truth in target:
        truth_bbox = [truth[0], truth[1], truth[2], truth[3]]
        # if cl == truth[4]:
        max_iou = max(max_iou, bb_intersection_over_union(bbox, truth_bbox))
    return max_iou
