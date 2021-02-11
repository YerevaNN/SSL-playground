# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from fvcore.common.file_io import PathManager
import os
import numpy as np
import xml.etree.ElementTree as ET

from fsdet.structures import BoxMode
from fsdet.data import DatasetCatalog, MetadataCatalog
import cv2


__all__ = ["register_api_dataset"]


def load_api_instances(dirname: str, image_file: str, class_names: list, has_box: bool):
    """
    Load api dataset to Detectron2 format.

    Args:
        dirname: Contain "Annotations", "ImageSets", "JPEGImages"
        split (str): one of "train", "test", "val", "trainval"
    """
    with PathManager.open(image_file) as f:
        fileids = np.loadtxt(f, dtype=np.str)

    dicts = []
    for fileid in fileids:
        anno_file = os.path.join(dirname, "Annotations", fileid + ".txt")
        jpeg_file = os.path.join(dirname, "JPEGImages", fileid + ".jpg")

        input_image = cv2.imread(jpeg_file)

        r = {
            "file_name": jpeg_file,
            "image_id": fileid,
            "height": input_image.shape[0],
            "width": input_image.shape[1],
        }
        instances = []

        if has_box:
            with open(anno_file) as annotations:
                for anno in annotations:
                    cls, xmin, ymin, xmax, ymax = anno
                    bbox = [float(xmin), float(ymin), float(xmax), float(ymax)]
                    # Original annotations are integers in the range [1, W or H]
                    # Assuming they mean 1-based pixel indices (inclusive),
                    # a box with annotation (xmin=1, xmax=W) covers the whole image.
                    # In coordinate space this is represented by (xmin=0, xmax=W)
                    bbox[0] -= 1.0
                    bbox[1] -= 1.0
                    instances.append(
                        {"category_id": class_names.index(cls), "bbox": bbox, "bbox_mode": BoxMode.XYXY_ABS}
                    )
        r["annotations"] = instances
        dicts.append(r)
    return dicts


def register_api_dataset(name, dirname, image_file, class_names, has_box):
    DatasetCatalog.register(name, lambda: load_api_instances(dirname, image_file, class_names, has_box))
    MetadataCatalog.get(name).set(
        thing_classes=class_names, dirname=dirname, evaluator_type="api_dataset"
    )


# ===============================================================================================
# old codes for testing on PASCAL VOC dataset
# ===============================================================================================


def load_voc_instances(dirname: str, image_file: str, class_names: list, has_box: bool):
    """
    Load api dataset to Detectron2 format.

    Args:
        dirname: Contain "Annotations", "ImageSets", "JPEGImages"
        split (str): one of "train", "test", "val", "trainval"
    """
    with PathManager.open(image_file) as f:
        fileids = np.loadtxt(f, dtype=np.str)

    dicts = []
    for fileid in fileids:
        anno_file = os.path.join(dirname, "Annotations", fileid + ".xml")
        jpeg_file = os.path.join(dirname, "JPEGImages", fileid + ".jpg")

        tree = ET.parse(anno_file)

        r = {
            "file_name": jpeg_file,
            "image_id": fileid,
            "height": int(tree.findall("./size/height")[0].text),
            "width": int(tree.findall("./size/width")[0].text),
        }
        instances = []
        if has_box:
            for obj in tree.findall("object"):
                cls = obj.find("name").text
                # We include "difficult" samples in training.
                # Based on limited experiments, they don't hurt accuracy.
                # difficult = int(obj.find("difficult").text)
                # if difficult == 1:
                # continue
                bbox = obj.find("bndbox")
                bbox = [float(bbox.find(x).text) for x in ["xmin", "ymin", "xmax", "ymax"]]
                # Original annotations are integers in the range [1, W or H]
                # Assuming they mean 1-based pixel indices (inclusive),
                # a box with annotation (xmin=1, xmax=W) covers the whole image.
                # In coordinate space this is represented by (xmin=0, xmax=W)
                bbox[0] -= 1.0
                bbox[1] -= 1.0
                instances.append(
                    {"category_id": class_names.index(cls), "bbox": bbox, "bbox_mode": BoxMode.XYXY_ABS}
                )
        r["annotations"] = instances
        dicts.append(r)
    return dicts


def register_api_dataset_old(name, dirname, image_file, class_names, has_box):
    DatasetCatalog.register(name, lambda: load_voc_instances(dirname, image_file, class_names, has_box))
    MetadataCatalog.get(name).set(
        thing_classes=class_names, base_classes=['aeroplane', 'bicycle', 'boat', 'bottle', 'car', 'cat', 'chair',
        'diningtable', 'dog', 'horse', 'person', 'pottedplant', 'sheep',
        'train', 'tvmonitor'], novel_classes=['bird', 'bus', 'cow', 'motorbike', 'sofa'], dirname=dirname, split="test", year=2007, evaluator_type="pascal_voc"
    )
