import os
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm

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


# def get_class_masks(teacher_preds, split_idx):
#     train_set = dict(list(teacher_preds.items())[:split_idx])
#     test_set = dict(list(teacher_preds.items())[split_idx:])

#     training_images = list(train_set.keys())
#     testing_images = list(test_set.keys())
#     training_cl_masks = []
#     test_cl_masks = []

#     for training_img in training_images:
#         training_cl_mask = {
#             j: np.array([x[4] == j for x in train_set[training_img]]) for j in [1, 2, 3]
#         }
#         training_cl_masks.append(training_cl_mask)

#     for testing_img in testing_images:
#         test_cl_mask = {
#             j: np.array([x[4] == j for x in test_set[testing_img]]) for j in [1, 2, 3]
#         }
#         test_cl_masks.append(test_cl_mask)

#     return training_cl_masks, test_cl_masks

def get_dataset(csv_path, label_root, split_idx):
    truth, predictions = process_data(csv_path, label_root)
    # train_cl_masks, test_cl_masks = get_class_masks(predictions, split_idx)

    image_paths = list(predictions.keys())
    labels = []
    samples = []
    classes = []

    for image in tqdm(image_paths):
        preds = predictions[image]
        gt = truth[image]

        for i, pred in enumerate(preds):
            samples.append(pred)
            classes.append(gt[i][4]) # check

        selected_pseudo_label = oracle_all(preds, gt)
        for spl in selected_pseudo_label:
            labels.append(spl)

    return samples, labels, classes #, train_cl_masks, test_cl_masks

def process_data(csv_path, label_root):
    df = pd.read_csv(csv_path)
    print("yeeeeeeeet")
    image_names = df['img_path'].to_list()
    bboxes = df['bbox'].to_list()
    classes = df['class'].to_list()
    confidences = df['confidence'].to_list()
    features = df['features'].to_list()

    processed_bboxes = []
    for box in tqdm(bboxes):
        bbox = box.replace('[', '').replace(']', '').split(',')
        new_bbox = []
        for b in range(len(bbox)):
            new_bbox.append(float(bbox[b]))
        processed_bboxes.append(new_bbox)

    processed_features = []
    processed_features = []
    for f in features:
        feature = f.replace('[', '').replace(']', '').split(',')
        new_feature = []
        for ft in range(len(feature)):
            new_feature.append(float(feature[ft]))
        processed_features.append(new_feature)

    predictions = {image: [] for image in image_names}

    for i in tqdm(range(len(image_names))):
        pred = [processed_bboxes[i][0], processed_bboxes[i][1], processed_bboxes[i][2], processed_bboxes[i][3],
                int(classes[i]), confidences[i]]
        for k in range(len(processed_features[i])):
            pred.append(processed_features[i][k])
        predictions[image_names[i]].append(pred)

    images = list(predictions.keys())

    truth = {image: [] for image in images}

    for j in tqdm(range(len(images))):
        label_file = images[j].split('/')[-1]
        label_file = '.'.join(label_file.split('.')[:-1]) + '.txt'
        label_file = os.path.join(label_root, label_file)
        target = get_target(label_file)
        truth[images[j]] = target

    return truth, predictions
