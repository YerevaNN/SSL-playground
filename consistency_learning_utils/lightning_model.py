from argparse import Namespace
import os
import random
import csv
from tkinter import W
import numpy as np
from .nets.oracle.convnet import Oracle
import json

import torch
from .nets.detection.faster_rcnn import fasterrcnn_resnet50_fpn
import torch.optim as optim
from PIL import Image
import torchvision.transforms as T

from torch import nn
from collections import OrderedDict
from pytorch_lightning import Trainer
from torchvision.utils import save_image

from mean_average_precision import MetricBuilder

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from aim.pytorch_lightning import AimLogger

from .dataloader import get_train_test_loaders
import zarr
from .nets.oracle.convnet import inference

def model_changed_classifier(reuse_classifier=False, initialize=False, class_num=20, gamma=1, box_score_thresh=0.05):
    """
    Args:
        reuse_classifier:
            - False: a regular Faster RCNN is used.
            - Otherwise: take a Faster RCNN and add a new layer on top of:
                - 'add': the original logits
                - 'concatenate': the concatenation of original logits and the feature vector
        initialize: initialization of Faster RCNN
            - False: random
            - 'backbone': ImageNet-pretrained ResNet backbone only
            - 'full': COCO pretrained Faster RCNN
        class_num: number of foreground classes (the code will add one for background)
    Returns:
        Initialized model
    """
    pretrained = initialize == 'full'
    backbone = initialize == 'backbone'

    if reuse_classifier is False:
        model = fasterrcnn_resnet50_fpn(pretrained=pretrained, progress=True,
                                        num_classes=class_num+1,
                                        pretrained_backbone=backbone, gamma=gamma,
                                        box_score_thresh=box_score_thresh)
        return model

    model = fasterrcnn_resnet50_fpn(pretrained=pretrained, progress=True,
                                    num_classes=91, pretrained_backbone=backbone,
                                    gamma=gamma, box_score_thresh=box_score_thresh)

    if reuse_classifier == 'add':
        old_cls_score = model.roi_heads.box_predictor.cls_score
        old_bbox_pred = model.roi_heads.box_predictor.bbox_pred

        new_cls_score = nn.Sequential(
            old_cls_score,  # 1024 -> 91
            nn.Linear(in_features=91, out_features=class_num+1, bias=True)
        )

        new_bbox_pred = nn.Sequential(
            old_bbox_pred,
            nn.Linear(in_features=364, out_features=4*(class_num+1), bias=True)
        )
    elif reuse_classifier == 'concatenate':
        old_cls_score = SkipConnection(model.roi_heads.box_predictor.cls_score)
        old_bbox_pred = SkipConnection(model.roi_heads.box_predictor.bbox_pred)

        new_cls_score = nn.Sequential(
            old_cls_score,  # 1024 -> 91
            nn.Linear(in_features=91+1024, out_features=class_num+1, bias=True)
        )

        new_bbox_pred = nn.Sequential(
            old_bbox_pred,
            nn.Linear(in_features=364+1024, out_features=4*(class_num+1), bias=True)
        )
    else:
        raise NotImplementedError

    model.roi_heads.box_predictor.cls_score = new_cls_score
    model.roi_heads.box_predictor.bbox_pred = new_bbox_pred

    return model


def make_target_from_y(y):
    """
    Converts a list of M objects like
        [{"label":1, "bndbox":{"xmin":1,"ymin":2,"xmax":3,"ymax":4}}]
    to an object of lists like
        {"boxes": cuda tensor (M,4), "labels": cuda tensor (M)}
    """
    target = []
    for i in range(len(y)):
        target_boxes = []
        target_labels = []
        for box in y[i]:
            xmin = int(box['bndbox']['xmin'])
            xmax = int(box['bndbox']['xmax'])
            ymin = int(box['bndbox']['ymin'])
            ymax = int(box['bndbox']['ymax'])
            label = int(box['label'])
            target_boxes.append([xmin, ymin, xmax, ymax])
            target_labels.append(label)
        tensor_boxes = torch.cuda.FloatTensor(target_boxes, device='cuda')
        tensor_labels = torch.cuda.LongTensor(target_labels, device='cuda')
        target.append({'boxes': tensor_boxes, 'labels': tensor_labels})
    return target


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

def get_target(label_root, image_name):
        image_name = '.'.join(image_name.split('.')[:-1]) + '.txt'  # replace .jpg (or whatever) to .txt
        label_path = os.path.join(label_root, image_name)
        target = []
        with open(label_path) as f:
            for line in f:
                line_arr = [float(t) for t in line.split(' ')]
                label, xmin, ymin, xmax, ymax = line_arr
                box = {}
                box['label'] = int(label)
                box['bndbox'] = {}
                box['bndbox']['xmin'] = int(xmin)
                box['bndbox']['ymin'] = int(ymin)
                box['bndbox']['xmax'] = int(xmax)
                box['bndbox']['ymax'] = int(ymax)
                if box['bndbox']['ymax'] <= box['bndbox']['ymin']:
                    box['bndbox']['ymax'] += 1
                if box['bndbox']['xmax'] <= box['bndbox']['xmin']:
                    box['bndbox']['xmax'] += 1
                target.append(box)
        return target

def break_batch(batch):
    x, y, paths = [], [], []
    for img in batch:
        x.append(img[0])
        y.append(img[1])
        paths.append(img[2])
    return x, y, paths


class SkipConnection(nn.Module):

    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):
        z = self.module(x)
        return torch.cat((x, z), dim=-1)


def constant_thresholding(pred, conf):
    features, boxes, scores, labels = pred
    kept_boxes = []
    kept_scores = []
    kept_labels = []
    for box, score, label in zip(boxes, scores, labels):
        keep_indices = torch.where(score >= conf)
        kept_boxes.append(box[keep_indices])
        kept_scores.append(score[keep_indices])
        kept_labels.append(label[keep_indices])
    return (kept_boxes, kept_scores, kept_labels)


def dynamic_thresholding(pred, class_num, conf, gamma):
    _, boxes, scores, labels = pred
    kept_boxes = []
    kept_scores = []
    kept_labels = []
    for box, score, label in zip(boxes, scores, labels):
        keep_indices = []

        classes = range(1, class_num+1)
        class_boxes = {i: [] for i in classes}
        for l in range(len(label)):
            class_boxes[int(label[l].item())].append(score[l].item())
        

        scores_sums = {cl : sum(class_boxes[cl]) for cl in classes}
        scaled_threshold = np.power(list(scores_sums.values()), gamma)
        scaled_threshold = scaled_threshold / scaled_threshold.max() * conf

        box_labels = (label.int() - 1).cpu()
        thresholds = [scaled_threshold[x] for x in box_labels]
        # thresholds = scaled_threshold[box_labels]
        thresholds = torch.from_numpy(np.array(thresholds))
        keep_indices = score.cpu() >= thresholds
        kept_boxes.append(box[keep_indices])
        kept_scores.append(score[keep_indices])
        kept_labels.append(label[keep_indices])

    return (kept_boxes, kept_scores, kept_labels)


def oracle(pred, truth, IOU_threshold=0.5):
    # Returning the best prediction over the threshold
    selected_pseudo_labels = []
    for i, im_pred in enumerate(pred):
        image_truth = truth[i]
        indices = []
        for t in image_truth:
            tbox = t['bndbox']
            truth_bbox = [tbox['xmin'], tbox['ymin'], tbox['xmax'], tbox['ymax']]
            bestIOU = IOU_threshold
            best_p = None
            for p in range(0, len(im_pred)):
                pred_label = im_pred[p][4]
                if pred_label == t['label']:
                    pred_bbox = [im_pred[p][0], im_pred[p][1], im_pred[p][2], im_pred[p][3]]
                    IOU = bb_intersection_over_union(truth_bbox, pred_bbox)
                    if IOU >= bestIOU:
                        best_p = p
                        bestIOU = IOU
            if best_p:
                indices.append(best_p)
        selected_labels_of_image = im_pred[sorted(indices)]
        selected_pseudo_labels.append(selected_labels_of_image)
    return selected_pseudo_labels


def oracle2(pred, truth, IOU_threshold=0.7, conf_threshold=-1.0):
    # Returning all the predictions over the threshold
    selected_pseudo_labels = []
    for i, im_pred in enumerate(pred):
        image_truth = truth[i]
        indices = []
        for t in image_truth:
            tbox = t['bndbox']
            truth_bbox = [tbox['xmin'], tbox['ymin'], tbox['xmax'], tbox['ymax']]
            best_ps = []
            for p in range(0, len(im_pred)):
                pred_label = im_pred[p][4]
                pred_conf = im_pred[p][5]
                if pred_label == t['label']:
                    pred_bbox = [im_pred[p][0], im_pred[p][1], im_pred[p][2], im_pred[p][3]]
                    IOU = bb_intersection_over_union(truth_bbox, pred_bbox)
                    if IOU >= IOU_threshold and pred_conf >= conf_threshold:
                        best_ps.append(p)

            for best_p in best_ps:
                indices.append(best_p)
        selected_labels_of_image = im_pred[sorted(indices)]
        selected_pseudo_labels.append(selected_labels_of_image)
    return selected_pseudo_labels


def change_prediction_format(unlab_pred, phd_pred):
    all_features = []
    all_boxes = []
    all_scores = []
    all_labels = []
    for i, sample_pred in enumerate(unlab_pred):
        boxes = sample_pred['boxes']
        all_boxes.append(boxes)
        labels = sample_pred['labels']
        all_labels.append(labels)
        scores = sample_pred['scores']
        all_scores.append(scores)
        features = phd_pred[i]
        all_features.append(features)
    return (all_features, all_boxes, all_scores, all_labels)


def filter_predictions(type, pred, class_num=None, truth=None, conf=None, gamma=None, iou_thresh=None,
                       model_path=None, teacher_thresholds=None, conv_thresholds=None):
    if type == 'constant':
        if conf is not None:
            selected_pseudo_labels = constant_thresholding(pred, conf)
        else:
            raise NotImplementedError
    elif type == 'dynamic':
        if class_num is not None and conf is not None and gamma is not None:
            selected_pseudo_labels = dynamic_thresholding(pred, class_num, conf, gamma)
        else:
            raise NotImplementedError
    elif type == 'oracle':
        if truth is not None:
            selected_pseudo_labels = oracle(pred, truth)
        else:
            raise NotImplementedError
    elif type == 'oracle2':
        if truth is not None and iou_thresh is not None:
            selected_pseudo_labels = oracle2(pred, truth,IOU_threshold=iou_thresh,
                                             conf_threshold=conf)
        else:
            raise NotImplementedError
    elif type == 'convnet':
        if model_path is not None:
            selected_pseudo_labels = inference(pred, model_path, iou_thresh=iou_thresh)
        else:
            raise NotImplementedError
    elif type == 'convnet_and_constant':
        if iou_thresh is not None and model_path is not None:
            old_selected_pseudo_labels = inference(pred, model_path, iou_thresh=iou_thresh)
            old_boxes, old_scores, old_labels = old_selected_pseudo_labels
            boxes, scores, labels = [], [], []
            for box, score, label in zip(old_boxes, old_scores, old_labels):
                indices = torch.where(score >= conf)
                boxes.append(box[indices])
                scores.append(score[indices])
                labels.append(label[indices])
            selected_pseudo_labels = (boxes, scores, labels)
        else:
            raise NotImplementedError
    elif type == 'convnet_and_optimal_constant':
        if iou_thresh is not None and model_path is not None and teacher_thresholds is not None:
            old_selected_pseudo_labels = inference(pred, model_path, iou_thresh=iou_thresh,
                                                   conv_thresholds=conv_thresholds)
            old_boxes, old_scores, old_labels = old_selected_pseudo_labels
            boxes, scores, labels = [], [], []
            for box, score, label in zip(old_boxes, old_scores, old_labels):
                box_labels = (label.int() - 1).cpu()
                thresholds = [teacher_thresholds[x] for x in box_labels]
                indices = torch.where(score >= torch.Tensor(thresholds).cuda())
                boxes.append(box[indices])
                scores.append(score[indices])
                labels.append(label[indices])
            selected_pseudo_labels = (boxes, scores, labels)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    return selected_pseudo_labels


class STAC(pl.LightningModule):
    def __init__(self, hparams: Namespace) -> None:
        super().__init__()

        self.hparams = hparams

        self.lr = self.hparams['learning_rate']
        self.stage = self.hparams['stage']
        self.confidence_threshold = self.hparams['confidence_threshold']
        self.thresholding_method = self.hparams['thresholding_method']
        self.last_unsupervised_loss = 0
        self.training_map = 1
        self.best_teacher_val = 1000
        self.best_student_val = 1000
        self.best_val_loss = 10000000
        self.validation_counter = 0
        self.validation_part = self.hparams["validation_part"]
        self.lam = self.hparams['consistency_lambda']
        self.momentum = self.hparams['momentum']
        self.weight_decay = self.hparams['weight_decay']
        self.consistency_criterion = self.hparams['consistency_criterion']
        self.testWithStudent = False
        self.onlyBurnIn = False
        self.no_val = False

        bs = self.hparams['batch_size']
        env = os.getenv('CUDA_VISIBLE_DEVICES')
        if env is not None:
            self.available_gpus = env.split(',')
            gpu_num = len(self.available_gpus)
        else:
            self.available_gpus = '0'
            gpu_num = 1

        print("GPUs count {}, GPU ids {}".format(gpu_num, self.available_gpus))
        self.hparams['batches_per_epoch'] = max(1, int((self.hparams['labeled_num'] + bs - 1) / bs / max(1, gpu_num)))
        self.batches_per_epoch = self.hparams['batches_per_epoch']
        self.check_val_epochs = max(
            1, self.hparams['check_val_steps'] // self.hparams['batches_per_epoch'])

        self.onTeacher = True  # as opposed to "on student"

        version_folder = os.path.join(self.hparams['phase_folder'], self.hparams['version_name'])

        self.save_dir_name_teacher = os.path.join(version_folder, 'teacher')
        self.save_dir_name_student = os.path.join(version_folder, 'student')
        self.output_csv = os.path.join(version_folder, 'output.csv')
        print("Creating Teacher & Student with {} initialization and reuse_classifier={}".format(
            self.hparams['initialization'], self.hparams['reuse_classifier']
        ))
        self.feature_folder = os.path.join('.', self.hparams['version_name'], 'features')
        os.makedirs(self.feature_folder, exist_ok=True)
        os.makedirs(version_folder, exist_ok=True)
        # self.zarr_file = zarr.open(os.path.join(self.feature_folder, 'features.zarr'), mode='w')

        self.teacher_init = 'full' if (self.hparams['teacher_init_path'] and (not self.hparams['skip_burn_in'])) else \
            self.hparams['initialization']

        self.teacher = model_changed_classifier(
            initialize=self.teacher_init,
            reuse_classifier=self.hparams['reuse_classifier'],
            class_num=self.hparams['class_num'],
            gamma=self.hparams['gamma'],
            box_score_thresh=self.hparams['box_score_thresh'])

        self.student = model_changed_classifier(
            initialize=self.hparams['initialization'],
            reuse_classifier=self.hparams['reuse_classifier'],
            class_num=self.hparams['class_num'],
            gamma=self.hparams['gamma'],
            box_score_thresh=self.hparams['box_score_thresh'])
        
        self.phd = model_changed_classifier(
            initialize='backbone',
            reuse_classifier=self.hparams['reuse_classifier'],
            class_num=self.hparams['class_num'],
            gamma=0,
            box_score_thresh=self.hparams['box_score_thresh'])
        
        self.teacher.to('cuda')
        self.student.to('cuda')
        self.phd.to('cuda')

        self.aim_logger = AimLogger(
            experiment=self.hparams['version_name']
        )

        self.make_teacher_trainer()
        self.make_student_trainer()

        with open("./train_logits.json", "w+") as jsonFile:
            json.dump([], jsonFile)

        with open("./logits.json", "w+") as jsonFile:
            jsonFile.write('')

        self.custom_validation_start()

    def make_feature_data_file(self, label_root, experiment_name):
        feature_data_path = "/home/hkhachatrian/SSL-playground/consistency_learning_utils/nets/oracle/feature_data/" + experiment_name + ".txt"
        if os.path.exists(feature_data_path):
            os.remove(feature_data_path)
        print("Starting to write new rfature file at: " + feature_data_path)
        with open(os.path.join(self.hparams['phase_folder'], 'train_' + str(self.stage) + '.txt')) as images_path:
            for im_path in images_path:
                im_path = im_path[:-1]
                with open(im_path, 'rb') as f:
                    image = Image.open(f)
                    image = image.convert("RGB")
                    width, height = image.size
                    image = np.array(image)
                if len(image.shape) == 2:
                    image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
                transform = T.Compose([T.ToTensor()])
                image = transform(image)
                image = torch.unsqueeze(image, 0)
                image = image.cuda()
                self.teacher.cuda()
                pred_teacher = self.teacher.forward(image)
                teacher_boxes = pred_teacher[0]['boxes']
                teacher_labels = pred_teacher[0]['labels']
                teacher_scores = pred_teacher[0]['scores']
                target_boxes = get_target(label_root, im_path.split('/')[-1])

                # generating negatives
                ratios = [0.33, 0.67, 1, 2, 3]
                num_pixels = [i for i in range(1000, width * height, 1000)]
                cnt = 150
                while cnt > 0:
                    ratio = random.choice(ratios)
                    num_pixel = random.choice(num_pixels)
                    # print(ratio, num_pixel)
                    # print(width, height)
                    w = int((num_pixel * ratio) ** 0.5)
                    h = num_pixel // w
                    if w + 2 >= width or h + 2 >= height:
                        continue
                    # print(w, h)
                    n_xmin = random.randrange(1, width - w - 1)
                    n_ymin = random.randrange(1, height - h - 1)
                    # print(n_xmin, n_ymin)
                    n_xmax = n_xmin + w
                    n_ymax = n_ymin + h
                    max_iou = 0
                    for gt in target_boxes:
                        xmin = gt['bndbox']['xmin']
                        ymin = gt['bndbox']['ymin']
                        xmax = gt['bndbox']['xmax']
                        ymax = gt['bndbox']['ymax']
                        iou = bb_intersection_over_union([n_xmin, n_ymin, n_xmax, n_ymax], [xmin, ymin, xmax, ymax])
                        if iou > max_iou:
                            max_iou = iou
                    if max_iou < 0.5:
                        cnt -= 1
                        str_positive = im_path + ' ' + '[' + str(n_xmin) + ',' + str(n_ymin) + ',' + str(n_xmax) + ',' + str(n_ymax) + '] ' + str(max_iou)+ ' ' + str(0) + ' ' + str(0) + '\n'
                        f1 = open(feature_data_path, "a+")
                        f1.writelines(str_positive)
                        f1.close()

                # generating positives
                for gt in target_boxes:
                    cnt = 100 // len(target_boxes)
                    while cnt > 0:
                        xmin = gt['bndbox']['xmin']
                        ymin = gt['bndbox']['ymin']
                        xmax = gt['bndbox']['xmax']
                        ymax = gt['bndbox']['ymax']
                        gam = 0.1
                        p_xmin = int(random.uniform(xmin - gam * width, xmin + gam * width))
                        p_ymin = int(random.uniform(ymin - gam * height, ymin + gam * height))
                        p_xmax = int(random.uniform(xmax - gam * width, xmax + gam * width))
                        p_ymax = int(random.uniform(ymax - gam * height, ymax + gam * height))
                        if p_xmax <= p_xmin or p_ymax <= p_ymin or p_xmin <= 0 or p_xmax >= width or p_ymin <= 0 or p_ymax >= height:
                            continue
                        iou = bb_intersection_over_union([p_xmin, p_ymin, p_xmax, p_ymax], [xmin, ymin, xmax, ymax])
                        if iou >= 0.5:
                            cnt -= 1
                            str_positive = im_path + ' ' + '[' + str(p_xmin) + ',' + str(p_ymin) + ',' + str(p_xmax) + ',' + str(p_ymax) + '] ' + str(iou)+ ' ' + str(0) + ' ' + str(0) + '\n'
                            f1 = open(feature_data_path, "a+")
                            f1.writelines(str_positive)
                            f1.close()

                # generating using teacher
                for i in range(len(teacher_boxes)):
                    teacher_max_iou = 0
                    t_xmin = int(teacher_boxes[i][0].item())
                    t_ymin = int(teacher_boxes[i][1].item())
                    t_xmax = int(teacher_boxes[i][2].item())
                    t_ymax = int(teacher_boxes[i][3].item())
                    t_label = teacher_labels[i].item()
                    t_score = teacher_scores[i].item()
                    for gt in target_boxes:
                        xmin = gt['bndbox']['xmin']
                        ymin = gt['bndbox']['ymin']
                        xmax = gt['bndbox']['xmax']
                        ymax = gt['bndbox']['ymax']
                        iou = bb_intersection_over_union([t_xmin, t_ymin, t_xmax, t_ymax], [xmin, ymin, xmax, ymax])
                        if iou > teacher_max_iou:
                            teacher_max_iou = iou
                    if teacher_max_iou <= 0.3 or teacher_max_iou >= 0.7:
                        str_teacher = im_path + ' ' + '[' + str(t_xmin) + ',' + str(t_ymin) + ',' + str(t_xmax) + ',' + str(t_ymax) + '] ' + str(teacher_max_iou)+ ' ' + str(t_score) + ' ' + str(t_label) + '\n'
                        f1 = open(feature_data_path, "a+")
                        f1.writelines(str_teacher)
                        f1.close()
                

        return feature_data_path

    def train_cnn_on_labeled(self):
        label_root = os.path.join(self.hparams['phase_folder'], 'labels')
        experiment_name = self.hparams['version_name']


        if 'oracle_feature_data_path' in self.hparams.keys():
            feature_data_path = self.hparams['oracle_feature_data_path']
        else:
            feature_data_path = self.make_feature_data_file(label_root, experiment_name)
        print("feature_data_path: ", feature_data_path)



        class_num = self.hparams['class_num']
        box_score_thresh = 0.005
        lr = 0.001
        epochs_per_stage = [
            1,
            4,
            50,
            30,
            1,
            1,
            1,
            1,
            1
        ]
        train_epochs = epochs_per_stage[self.stage]

        model = Oracle(experiment_name, lr = lr, class_num=class_num, train_epochs=train_epochs)

        if self.hparams["oracle_pretrained"]:
            checkpoint_path = self.hparams["oracle_model_path_init"]
            model.load_from_path(checkpoint_path)

        split = 0.9
        batch_size = 16
        model.set_datasets(feature_data_path, label_root, split, batch_size, class_num, box_score_thresh)
        model.fit_model()
        
        new_cnn_path = os.getcwd() + "/checkpoints/{}/".format(experiment_name) + 'our_best.ckpt'
        return new_cnn_path

    def custom_validation_start(self):
        self.student_mAP = MetricBuilder.build_evaluation_metric("map_2d", async_mode=True,
                                                         num_classes=self.hparams['class_num'] + 1)
        self.teacher_mAP = MetricBuilder.build_evaluation_metric("map_2d", async_mode=True,
                                                         num_classes=self.hparams['class_num'] + 1)
        self.prediction_cache = {}
        self.validation_student_boxes = 0
        self.validation_teacher_boxes = 0
        self.validation_images = 0


    def set_test_with_student(self, val1, val2):
        self.testWithStudent = val1
        self.onlyBurnIn = val2

    def save_checkpoint(self, path):
        torch.save(self.student.state_dict(), path)

    def load_from_checkpoint_without_last_layers(self, path):
        pretrained_dict = torch.load(path)
        model_dict = self.student.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if 'cls_score' not in k and 'bbox_pred' not in k}
        model_dict.update(pretrained_dict)
        self.student.load_state_dict(model_dict)

    def load_best_teacher(self):
        checkpoint_name = ''
        for ckpt_name in os.listdir(self.save_dir_name_teacher):
            if ckpt_name.endswith('ckpt'):
                checkpoint_name = ckpt_name
                break
        checkpoint_path = os.path.join(self.save_dir_name_teacher, checkpoint_name)
        best_dict = torch.load(checkpoint_path)
        actual_dict = {k[8:]: v for k, v in best_dict['state_dict'].items() if k.startswith('teacher')}
        self.teacher.load_state_dict(actual_dict)

    def copy_student_from_current_teacher(self):
        actual_dict = self.teacher.state_dict()
        self.student.load_state_dict(actual_dict)
    
    def copy_phd_from_current_teacher(self):
        actual_dict = self.teacher.state_dict()
        self.phd.load_state_dict(actual_dict)

    def load_checkpoint_teacher(self, checkpoint_path, skip_last_layer=False):
        checkpoint = torch.load(checkpoint_path)
        model_dict = self.teacher.state_dict()
        loaded_dict = {k[8:]: v for k, v in checkpoint['state_dict'].items() if k.startswith('teacher')
                       and (('cls_score' not in k and 'bbox_pred' not in k) if skip_last_layer else True)}
        model_dict.update(loaded_dict)
        self.teacher.load_state_dict(model_dict)

    def load_checkpoint_student(self, checkpoint_path, skip_last_layer=False):
        checkpoint = torch.load(checkpoint_path)
        model_dict = self.student.state_dict()
        loaded_dict = {k[8:]: v for k, v in checkpoint['state_dict'].items() if k.startswith('student')
                       and (('cls_score' not in k and 'bbox_pred' not in k) if skip_last_layer else True)}
        model_dict.update(loaded_dict)
        self.student.load_state_dict(model_dict)

    def test_from_checkpoint(self, checkpoint_path):
        print('Testing with this checkpoint: {}'.format(checkpoint_path))
        # self.load_from_checkpoint(checkpoint_path) # this one returns a new model!!!
        self.load_checkpoint_teacher(checkpoint_path)
        self.load_checkpoint_student(checkpoint_path)

        self.test()

    def test_from_best_checkpoint(self):
        if self.onlyBurnIn:
            ckpt_path = self.save_dir_name_teacher
        else:
            ckpt_path = self.save_dir_name_student
        
        checkpoint_name = ''
        for ckpt_name in os.listdir(ckpt_path):
            if ckpt_name.endswith('ckpt'):
                checkpoint_name = ckpt_name
                break
        checkpoint_path = os.path.join(ckpt_path, checkpoint_name)
        self.test_from_checkpoint(checkpoint_path)

    def update_teacher_EMA(self, keep_rate=0.996):
        student_model_dict = {
            key: value for key, value in self.student.state_dict().items()
        }

        new_teacher_dict = OrderedDict()
        for key, value in self.teacher.state_dict().items():
            if key in student_model_dict.keys():
                new_teacher_dict[key] = (
                    student_model_dict[key] * (1 - keep_rate) + value * keep_rate
                )
            else:
                raise Exception("{} is not found in student model".format(key))

        self.teacher.load_state_dict(new_teacher_dict)
        # key = 'roi_heads.box_head.fc6.weight'
        # with open('{}_gpu{}.log'.format(self.hparams['version_name'], self.global_rank), 'a') as f:
        #     f.write("After EMA: GR={} key={} max value={}\n".format(
        #         self.global_rank, key, self.teacher.state_dict()[key].max()
        #     ))

    def set_datasets(self, labeled_file_path, unlabeled_file_path, testing_file_path,
                     external_val_file_path, external_val_label_root, label_root, skip_data_path=None):

        loaders = get_train_test_loaders(labeled_file_path,
                                         unlabeled_file_path,
                                         testing_file_path,
                                         external_val_file_path,
                                         external_val_label_root,
                                         label_root,
                                         self.hparams['batch_size'],
                                         self.hparams['num_workers'],
                                         stage=self.stage,
                                         validation_part=self.validation_part,
                                         augmentation=self.hparams['augmentation'],
                                         thresholding=self.hparams['thresholding_method'],
                                         skip_data_path=skip_data_path)
        if (os.path.isfile(external_val_file_path)):
            self.train_loader, self.test_loader, self.val_loader = loaders
        else:
            self.train_loader, self.test_loader = loaders
            print("No validation set")
            self.no_val = True


    def make_teacher_trainer(self):
        self.t_checkpoint_callback = ModelCheckpoint(
            # monitor='val_loss',
            dirpath=self.save_dir_name_teacher,
            filename='{epoch}',
            verbose=True,
            save_last=True,
            period=1
        )
        self.teacher_trainer = Trainer(
            gpus=-1, checkpoint_callback=True, # what is this?
            accelerator='ddp',
            callbacks=[self.t_checkpoint_callback],
            num_sanity_val_steps=0,
            logger=self.aim_logger,
            log_every_n_steps=10, progress_bar_refresh_rate=1,
            gradient_clip_val=self.hparams['gradient_clip_threshold'],
            min_steps=self.hparams['total_steps_teacher'],
            max_steps=self.hparams['total_steps_teacher'],
            check_val_every_n_epoch=self.check_val_epochs,
            deterministic=True,
            accumulate_grad_batches=self.hparams['gradac_batches']
        )
        self.teacher_test_trainer = Trainer(
            gpus=1, checkpoint_callback=True,  # what is this?
            callbacks=[self.t_checkpoint_callback],
            num_sanity_val_steps=0,
            logger=self.aim_logger,
            log_every_n_steps=10, progress_bar_refresh_rate=1,
            gradient_clip_val=self.hparams['gradient_clip_threshold'],
            min_steps=self.hparams['total_steps_teacher'],
            max_steps=self.hparams['total_steps_teacher'],
            check_val_every_n_epoch=self.check_val_epochs,
            deterministic=True,
            accumulate_grad_batches=self.hparams['gradac_batches']
        )


    def make_student_trainer(self):
        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            dirpath=self.save_dir_name_student,
            filename='{epoch}',
            verbose=True,
            save_last=True,
            period=1
        )

        self.student_trainer = Trainer(
            gpus=-1, checkpoint_callback=True, # what is this?
            accelerator='ddp',
            callbacks=[checkpoint_callback],
            logger=self.aim_logger,
            num_sanity_val_steps=0,
            log_every_n_steps=10, progress_bar_refresh_rate=1,
            gradient_clip_val=self.hparams['gradient_clip_threshold'],
            min_steps=self.hparams['total_steps_student'],
            max_steps=self.hparams['total_steps_student'],
            check_val_every_n_epoch=self.check_val_epochs,
            deterministic=True,
            accumulate_grad_batches=self.hparams['gradac_batches']
        )
        self.student_test_trainer = Trainer(
            gpus=1, checkpoint_callback=True,  # what is this?
            callbacks=[checkpoint_callback],
            logger=self.aim_logger,
            num_sanity_val_steps=0,
            log_every_n_steps=10, progress_bar_refresh_rate=1,
            gradient_clip_val=self.hparams['gradient_clip_threshold'],
            min_steps=self.hparams['total_steps_student'],
            max_steps=self.hparams['total_steps_student'],
            check_val_every_n_epoch=self.check_val_epochs,
            deterministic=True,
            accumulate_grad_batches=self.hparams['gradac_batches']
        )

    def student_forward(self, x, image_paths):
        return self.student.forward(x, image_paths=image_paths)

    def teacher_forward(self, x, image_paths):
        return self.teacher.forward(x, image_paths=image_paths)

    def phd_forward(self, x, image_paths):
        return self.phd.forward(x, image_paths=image_paths)

    def forward(self, x, image_paths):
        if self.onTeacher:
            return self.teacher_forward(x, image_paths=image_paths)
        return self.student_forward(x, image_paths=image_paths)

    # @pl.data_loader
    def train_dataloader(self):
        return self.train_loader

    def frcnn_loss(self, res):
        final_loss = res['loss_classifier'] + 10 * res['loss_box_reg'] + \
                     res['loss_objectness'] + res['loss_rpn_box_reg']
        return final_loss

    def teacher_supervised_step(self, sup_batch):
        if self.hparams['augmentation'] == 3:
            x, y, image_paths = [], [], []
            for i in sup_batch:
                _, strong = i
                x.append(strong[0])
                y.append(strong[1])
                image_paths.append(strong[2])
        else:
            x, y, image_paths = break_batch(sup_batch)

        target = make_target_from_y(y)

        y_hat = self.teacher(x, target, image_paths)

        with open('{}_gpu{}.log'.format(self.hparams['version_name'], self.global_rank), 'a') as f:
            f.write("GR={} images=({})\n".format(
                self.global_rank, ' '.join([os.path.basename(i[2]) for i in sup_batch])))
        return y_hat

    def student_supervised_step(self, sup_batch):
        if self.hparams['augmentation'] == 3:
            x, y, image_paths = [], [], []
            for i in sup_batch:
                weak, _ = i
                x.append(weak[0])
                y.append(weak[1])
                image_paths.append(weak[2])
        else:
            x, y, image_paths = break_batch(sup_batch)

        target = make_target_from_y(y)
        y_hat = self.student(x, target, image_paths)

        with open('{}_gpu{}.log'.format(self.hparams['version_name'], self.global_rank), 'a') as f:
            f.write("Supervised GR={} images=({})\n".format(
                self.global_rank, ' '.join([os.path.basename(i[2]) for i in sup_batch])))

        return y_hat

    def get_boxes(self, img_path):
        target = {}
        boxes = []
        labels = []
        scores = []
        label_folder = os.path.join(self.hparams['phase_folder'], 'labels')
        image_id = img_path.split('/')[-1].split('.')[0]
        label_path = os.path.join(label_folder, image_id + '.txt')
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
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(label)
                scores.append(1)
        tensor_boxes = torch.cuda.FloatTensor(boxes, device="cuda")
        tensor_labels = torch.cuda.FloatTensor(labels, device="cuda")
        tensor_scores = torch.cuda.FloatTensor(scores, device="cuda")

        target['boxes'] = tensor_boxes
        target['labels'] = tensor_labels
        target['scores'] = tensor_scores
        return target

    def student_unsupervised_step(self, unsup_batch):
        with open('{}_gpu{}.log'.format(self.hparams['version_name'], self.global_rank), 'a') as f:
            f.write("Unsupervised GR={} images=({})\n".format(
            self.global_rank, ' '.join([os.path.basename(i[0][2]) for i in unsup_batch])))

        unlabeled_x, unlabeled_y, unlabeled_image_paths = [], [], []
        augmented_x, augmented_image_paths = [], []

        unlabeled_ground_truth_boxes = []

        for i in unsup_batch:
            unlab, augment = i
            unlabeled_x.append(unlab[0])
            if self.hparams['thresholding_method'].startswith('oracle'):
                unlabeled_y.append(unlab[1])
            unlabeled_image_paths.append(unlab[2])
            # unlabeled_ground_truth_boxes.append(self.get_boxes(unlab[2]))
            augmented_x.append(augment[0])
            augmented_image_paths.append(augment[2])

        self.teacher.eval()
        unlab_pred = self.teacher_forward(unlabeled_x, unlabeled_image_paths)
        pseudo_boxes_all_per_class = {i: 0 for i in range(self.hparams['class_num'])}
        pseudo_boxes_confindent_per_class = {i: 0 for i in range(self.hparams['class_num'])}


        # for sample_pred in unlab_pred:
        #     keep_indices = torch.where(sample_pred['scores'] >= self.hparams['oracle_iou_threshold'])
        #     # if keep_indices[0].shape[0] == 0 and sample_pred['boxes'].shape[0] > 0:
        #     #     keep_indices = torch.cat((keep_indices[0], torch.tensor([0], device='cuda')), 0)
        #     sample_pred['boxes'] = sample_pred['boxes'][keep_indices]
        #     sample_pred['labels'] = sample_pred['labels'][keep_indices]
        #     sample_pred['scores'] = sample_pred['scores'][keep_indices]
        teacher_boxes = []
        for sample_pred in unlab_pred:
            cur_labels = sample_pred['labels']
            for l in range(len(cur_labels)):
                pseudo_boxes_all_per_class[cur_labels[l].item() - 1] += 1
            cur_boxes = sample_pred['boxes']
            teacher_boxes.append(cur_boxes)
        # gt_boxes = []
        # for gt in unlabeled_ground_truth_boxes:
        #     gtb = gt['boxes']
        #     gt_boxes.append(gtb)
        self.phd.eval()
        phd_pred = self.phd.forward(unlabeled_x, teacher_boxes=teacher_boxes)
        phd_pred = torch.split(phd_pred, [x.shape[0] for x in teacher_boxes])

        # save_image(unlabeled_x[0], 'unlabeled.png')
        # save_image(augmented_x[0], 'augmented.png')

        pseudo_boxes_all = 0
        pseudo_boxes_confident = 0

        non_zero_boxes = []
        target = []
        predictions = change_prediction_format(unlab_pred, phd_pred)
        
        # for i in range(len(predictions)):
        #     img_name = unlabeled_image_paths[i].split('/')[-1].split('.')[0]
        #     for j, p in enumerate(predictions[i][1]):
        #         bbox = [float(k) for k in predictions[i][0][j]]
        #         feat = p.cpu().detach().numpy().tolist()
        #         bbox = np.array(bbox, dtype='float32')
        #         feat = np.array(feat, dtype='float32')
        #         cur_box_name = img_name + '_' + str(j)
        #         self.zarr_file[cur_box_name + '_bbox'] = bbox
        #         self.zarr_file[cur_box_name + '_feat'] = feat

        teacher_thresholds, conv_thresholds = None, None

        if self.hparams['thresholding_method'] == 'convnet_and_optimal_constant':
            teacher_thresholds = np.load('/home/hkhachatrian/SSL-playground/checkpoints/coco_2017_base_0_from_coco_0_convnet_on_coco_05_long/teacher_thresholds.npy')
            conv_thresholds = np.load('/home/hkhachatrian/SSL-playground/checkpoints/coco_2017_base_0_from_coco_0_convnet_on_coco_05_long/conv_thresholds.npy')

        selected_pseudo_labels = filter_predictions(self.hparams['thresholding_method'], 
                                                    predictions, truth=unlabeled_y,
                                                    class_num=self.hparams['class_num'],
                                                    conf=self.hparams['confidence_threshold'],
                                                    gamma=self.hparams['dt_gamma'],
                                                    iou_thresh=self.hparams['oracle_iou_threshold'],
                                                    model_path=self.hparams['oracle_model_path'],
                                                    teacher_thresholds=teacher_thresholds,
                                                    conv_thresholds=conv_thresholds)

        # selected_pseudo_labels_dynamic = filter_predictions('dynamic', 
        #                                             predictions, truth=unlabeled_y,
        #                                             class_num=self.hparams['class_num'],
        #                                             conf=self.hparams['confidence_threshold'],
        #                                             gamma=self.hparams['dt_gamma'],
        #                                             iou_thresh=self.hparams['oracle_iou_threshold'],
        #                                             model_path=self.hparams['oracle_model_path'])



        boxes, scores, labels = selected_pseudo_labels
        # boxes_dynamic, scores_dynamic, labels_dynamic = selected_pseudo_labels_dynamic

        # boxes = boxes + list(set(boxes_dynamic) - set(boxes))
        # scores = scores + list(set(scores_dynamic) - set(scores))
        # labels = labels + list(set(labels_dynamic) - set(labels))


        for i in range(len(boxes)):
            target_boxes = boxes[i]
            target_labels = labels[i]

            pseudo_boxes_all += predictions[0][i].shape[0]

            if target_boxes.shape[0] > 0:
                non_zero_boxes.append(i)
            
            for lc in range(len(target_labels)):
                pseudo_boxes_confindent_per_class[target_labels[lc].item() - 1] += 1

            # TODO move to the device of the model
            pseudo_boxes_confident += target_boxes.shape[0]

            tensor_boxes = torch.tensor(target_boxes).float().cuda()
            tensor_labels = torch.tensor(target_labels).long().cuda()

            target.append({'boxes': tensor_boxes, 'labels': tensor_labels})


        if len(non_zero_boxes):
            augment_pred = self.student(
                [x for i, x in enumerate(augmented_x) if i in non_zero_boxes],
                [y for i, y in enumerate(target) if i in non_zero_boxes],
                [z for i, z in enumerate(augmented_image_paths) if i in non_zero_boxes]
            )
            unsup_loss = self.frcnn_loss(augment_pred)
        else:
            unsup_loss = 0 * augmented_x[0].new(1).squeeze()

        # Enable in case of weight decay!
        # for p in self.teacher.rpn.named_parameters():
        #     self.logger.experiment.track(p[1].sum().item(),
        #                                  name='teacher_weight', model=self.onTeacher,
        #                                  stage=self.stage)
        #     break
        for cl in range(self.hparams['class_num']):
            self.logger.experiment.track(
                pseudo_boxes_all_per_class[cl] / len(unlab_pred),
                name='pseudo_boxes_all_{}'.format(cl), context={'model':self.onTeacher, 'stage':self.stage})
            self.logger.experiment.track(
                pseudo_boxes_confindent_per_class[cl] / len(unlab_pred),
                name='pseudo_boxes_confident_{}'.format(cl), context={'model':self.onTeacher, 'stage':self.stage})
        self.logger.experiment.track(
            pseudo_boxes_all / len(unlab_pred),
            name='pseudo_boxes_all', context={'model':self.onTeacher, 'stage':self.stage})
        self.logger.experiment.track(
            pseudo_boxes_confident / len(unlab_pred),
            name='pseudo_boxes_confident', context={'model':self.onTeacher, 'stage':self.stage})
        return unsup_loss

    def teacher_training_step(self, batch_list):
        sup_batch, _ = batch_list
        self.teacher.set_is_supervised(True)
        # save_image(sup_batch[0][0], 'image1.png')

        sup_loss = self.teacher_supervised_step(sup_batch)

        loss = self.frcnn_loss(sup_loss)
        self.logger.experiment.track(sup_loss['loss_classifier'].item(), name='loss_classifier',
                                         context={'model':self.onTeacher, 'stage':self.stage})
        self.logger.experiment.track(sup_loss['loss_box_reg'].item(), name='loss_box_reg',
                                         context={'model':self.onTeacher, 'stage':self.stage})
        self.logger.experiment.track(sup_loss['loss_objectness'].item(), name='loss_objectness',
                                          context={'model':self.onTeacher, 'stage':self.stage})
        self.logger.experiment.track(sup_loss['loss_rpn_box_reg'].item(), name='loss_rpn_box_reg',
                                         context={'model':self.onTeacher, 'stage':self.stage})
        self.logger.experiment.track(loss.item(), name='loss_sum', context={'model':self.onTeacher, 'stage':self.stage})
        return {'loss': loss}

    def student_training_step(self, batch_list):
        self.update_teacher_EMA(keep_rate=self.hparams['EMA_keep_rate'])

        sup_batch, unsup_batch = batch_list
        self.student.set_is_supervised(True)
        sup_y_hat = self.student_supervised_step(sup_batch)

        self.student.set_is_supervised(False)

        unsup_loss = self.student_unsupervised_step(unsup_batch)

        sup_loss = self.frcnn_loss(sup_y_hat)
        loss = sup_loss + self.lam * unsup_loss
        with open('{}_gpu{}.log'.format(self.hparams['version_name'], self.global_rank), 'a') as f:
            f.write("GR={} loss={:.6f}\n".format(self.global_rank, loss))

        # if self.global_step % 20 < 10 or unsup_loss.sum().item() == 0.0:
        #     loss = sup_loss
        # else:
        #     loss = unsup_loss

        # teacher_weight = self.teacher.roi_heads.box_predictor.cls_score.weight.sum().item()
        # self.logger.experiment.track(teacher_weight, name='teacher_weight',
        #                                  context={'model':self.onTeacher, 'stage':self.stage})
        # student_weight = self.student.roi_heads.box_predictor.cls_score.weight.sum().item()
        # self.logger.experiment.track(student_weight, name='student_weight',
        #                                  context={'model':self.onTeacher, 'stage':self.stage})

        # self.logger.experiment.track(sup_y_hat['loss_classifier'].item(), name='loss_classifier',
        #                                  context={'model':self.onTeacher, 'stage':self.stage})
        # self.logger.experiment.track(sup_y_hat['loss_box_reg'].item(), name='loss_box_reg',
        #                                  context={'model':self.onTeacher, 'stage':self.stage})
        # self.logger.experiment.track(sup_y_hat['loss_objectness'].item(), name='loss_objectness',
        #                                   context={'model':self.onTeacher, 'stage':self.stage})
        # self.logger.experiment.track(sup_y_hat['loss_rpn_box_reg'].item(), name='loss_rpn_box_reg',
        #                                  context={'model':self.onTeacher, 'stage':self.stage})
        # self.logger.experiment.track(sup_loss.item(), name='training_sup_loss',
        #                                  context={'model':self.onTeacher, 'stage':self.stage})
        # self.logger.experiment.track(unsup_loss.item(), name='training_unsup_loss',
        #                                  context={'model':self.onTeacher, 'stage':self.stage})
        # self.logger.experiment.track(loss.item(), name='loss_sum',
        #                                  context={'model':self.onTeacher, 'stage':self.stage})
        return {'loss': loss}

    def training_step(self, batch_list, batch_idx):
        if self.onTeacher:
            return self.teacher_training_step(batch_list)
        else:
            return self.student_training_step(batch_list)

    # @pl.data_loader
    def test_dataloader(self):
        return self.test_loader

    # @pl.data_loader
    def val_dataloader(self):
        if(not self.no_val):
            return self.val_loader

    def validation_step(self, batch, batch_idx):
        # if self.stage == 0 or self.validation_part == 0:
        #     return {}
        if self.no_val:
            return
        x, y, image_paths = batch

        student_y_hat = self.student_forward(x, image_paths)
        teacher_y_hat = self.teacher_forward(x, image_paths)

        batch_size = len(student_y_hat)

        self.validation_images += batch_size

        output_tensor = torch.zeros(size=(batch_size, 3, 200, 7))
        # 3 = truth, teacher, student

        for i in range(batch_size):
            student_pred_for_mAP = []
            teacher_pred_for_mAP = []
            truth_for_mAP = []
            img_id = image_paths[i]

            student_boxes = student_y_hat[i]['boxes'].cpu().numpy()
            student_labels = student_y_hat[i]['labels'].cpu().numpy()
            student_scores = student_y_hat[i]['scores'].cpu().numpy()

            teacher_boxes = teacher_y_hat[i]['boxes'].cpu().numpy()
            teacher_labels = teacher_y_hat[i]['labels'].cpu().numpy()
            teacher_scores = teacher_y_hat[i]['scores'].cpu().numpy()

            for (j, box) in enumerate(student_boxes):
                student_pred_for_mAP.append([box[0], box[1], box[2], box[3],
                                             student_labels[j], student_scores[j]])

            for (j, box) in enumerate(teacher_boxes):
                teacher_pred_for_mAP.append([box[0], box[1], box[2], box[3],
                                             teacher_labels[j], teacher_scores[j]])

            for box in y[i]:
                xmin = int(box['bndbox']['xmin'])
                xmax = int(box['bndbox']['xmax'])
                ymin = int(box['bndbox']['ymin'])
                ymax = int(box['bndbox']['ymax'])
                truth_for_mAP.append([xmin, ymin, xmax, ymax, int(box['label']), 0, 0])

            # self.student_mAP.add(np.array(student_pred_for_mAP), np.array(truth_for_mAP))
            # self.teacher_mAP.add(np.array(teacher_pred_for_mAP), np.array(truth_for_mAP))
            output_tensor[i][0][:min(200,len(truth_for_mAP))] = torch.tensor(np.array(truth_for_mAP)[:200])
            if len(teacher_pred_for_mAP):
                output_tensor[i][1][:len(teacher_pred_for_mAP),:6] = torch.tensor(np.array(teacher_pred_for_mAP))
            if len(student_pred_for_mAP):
                output_tensor[i][2][:len(student_pred_for_mAP),:6] = torch.tensor(np.array(student_pred_for_mAP))

            self.validation_teacher_boxes += len(teacher_pred_for_mAP)
            self.validation_student_boxes += len(student_pred_for_mAP)
            self.prediction_cache[img_id] = {
                "student_pred": student_pred_for_mAP,
                "teacher_pred": teacher_pred_for_mAP,
                "truth": truth_for_mAP
            }

        return output_tensor.cuda()

    def validation_epoch_end(self, results):
        if self.no_val:
            return
        self.validation_counter += 1

        # print("GR={} before all_gather: results: len={}".format(self.global_rank, len(results)))
        results = self.all_gather(results)
        # print("GR={} all_gather: results: len={}".format(self.global_rank, len(results)))

        def filter_non_zero(tensor, lim=6):
            return np.array([row.cpu().numpy()[:lim] for row in tensor if row.sum() > 0])

        if self.global_rank == 0:
            for batch_pairs in results:
                for batch in batch_pairs:
                    for image in batch: # (3, 100, 6)
                        truth = filter_non_zero(image[0], lim=7)
                        student_pred = filter_non_zero(image[2])
                        self.student_mAP.add(student_pred, truth)
        val_loss = 0
        if self.global_rank == 0:
            for batch_pairs in results:
                for batch in batch_pairs:
                    for image in batch: # (3, 100, 6)
                        truth = filter_non_zero(image[0], lim=7)
                        teacher_pred = filter_non_zero(image[1])
                        self.teacher_mAP.add(teacher_pred, truth)

            # mAP1 = self.mAP.value(iou_thresholds=0.5, recall_thresholds=np.arange(0., 1.1, 0.1))['mAP']
            ious = np.arange(0.5, 1.0, 0.05)
            student_mAP2 = self.student_mAP.value(iou_thresholds=0.5)['mAP']
            student_mAP3 = self.student_mAP.value(iou_thresholds=ious,
                                  recall_thresholds=np.arange(0., 1.01, 0.01), mpolicy='soft')

            self.logger.experiment.track(float(student_mAP2), name='map2', context={'model':False, 'stage':self.stage})
            self.logger.experiment.track(float(student_mAP3['mAP']), name='mAP5095', context={'model':False, 'stage':self.stage})
            for iou in ious:
                ap = float(np.mean([x['ap'] for cat, x in student_mAP3[iou].items() if cat != 0]))
                if self.hparams['class_num'] == 91:
                    ap = ap*91/80
                self.logger.experiment.track(
                    ap, name='AP{:.0f}'.format(iou*100),
                    context={'model':False, 'stage':self.stage})
                if iou != 0.5:
                    continue
                for cat in range(1, self.hparams['class_num']+1):
                    self.logger.experiment.track(
                        float(student_mAP3[iou][cat]['ap']), name='AP{:.0f}_{}'.format(iou * 100, cat),
                        context={'model':False, 'stage':self.stage})
            
            teacher_mAP2 = self.teacher_mAP.value(iou_thresholds=0.5)['mAP']
            teacher_mAP3 = self.teacher_mAP.value(iou_thresholds=ious,
                                  recall_thresholds=np.arange(0., 1.01, 0.01), mpolicy='soft')

            self.logger.experiment.track(float(teacher_mAP2), name='map2', context={'model':True, 'stage':self.stage})
            self.logger.experiment.track(float(teacher_mAP3['mAP']), name='mAP5095', context={'model':True, 'stage':self.stage})
            
            for iou in ious:
                ap = float(np.mean([x['ap'] for cat, x in teacher_mAP3[iou].items() if cat != 0]))
                if self.hparams['class_num'] == 91:
                    ap = ap*91/80
                self.logger.experiment.track(
                    ap, name='AP{:.0f}'.format(iou*100),
                    context={'model':True, 'stage':self.stage})
                if iou != 0.5:
                    continue
                for cat in range(1, self.hparams['class_num']+1):
                    self.logger.experiment.track(
                        float(teacher_mAP3[iou][cat]['ap']), name='AP{:.0f}_{}'.format(iou * 100, cat),
                        context={'model':True, 'stage':self.stage}) 

            # val_loss as a surrogate for mAP
            val_loss = 1 - student_mAP2
            if self.onTeacher:
                val_loss = 1 - teacher_mAP2

            print('mAP: ', 1 - val_loss)
            print('best_mAP: ', 1 - self.best_val_loss)

            if self.onTeacher:
                self.best_teacher_val = min(self.best_teacher_val, val_loss)
            else:
                self.best_student_val = min(self.best_student_val, val_loss)
            self.best_val_loss = min(self.best_val_loss, val_loss)


        # self.store_predictions()

        # TODO: two metrics below are from one gpu only!
        self.logger.experiment.track(
            self.validation_teacher_boxes / self.validation_images,
            name='val_teacher_boxes', context={'model':True, 'stage':self.stage})
        self.logger.experiment.track(
            self.validation_student_boxes / self.validation_images,
            name='val_student_boxes', context={'model':False, 'stage':self.stage})

        self.custom_validation_start()

        print('val_loss', val_loss)
        val_loss = torch.tensor(val_loss)
        print('val_loss', val_loss)
        return {
            'val_loss': val_loss
        }

    def store_predictions(self):
        if self.onTeacher:
            folder = self.save_dir_name_teacher
        else:
            folder = self.save_dir_name_student
        filename = os.path.join(folder, "{}_{}.npy".format(self.global_step, self.global_rank))
        os.makedirs(folder, exist_ok=True)
        np.save(filename, self.prediction_cache)

    def test_step(self, batch, batch_idx):
        x, target, image_paths = batch
        names = []
        for image_path in image_paths:
            name = image_path.split('/')[-1]
            name = name[:-4]
            names.append(name)
        if self.testWithStudent and not self.onlyBurnIn:
            y_hat = self.student_forward(x, image_paths)
        else:
            y_hat = self.teacher_forward(x, image_paths)

        rows = []
        for i in range(len(x)):
            img_id = names[i]  # we should keep image ID somehow in the batch!
            boxes = y_hat[i]['boxes'].cpu().numpy()  # x_min, y_min, x_max, y_max
            labels = y_hat[i]['labels'].cpu().numpy()
            scores = y_hat[i]['scores'].cpu().numpy()
            for j, box in enumerate(boxes):
                row = [img_id, scores[j], labels[j], ','.join(["{:.0f}".format(t) for t in box])]
                rows.append(row)
        self.csvwriter.writerows(rows)

        return {'test_acc': 0}

    def on_test_end(self):
        self.csvwriter_file.close()
        return {'test_acc': 0.5}

    def compute_accuracy(self, y, y_hat):
        y_hat = y_hat.argmax(dim=-1)
        return torch.sum(y == y_hat).item() / len(y)

    # def optimizer_step(self, epoch: int = None, batch_idx: int = None, optimizer = None,
    #                    optimizer_idx: int = None, optimizer_closure = None, on_tpu: bool = None,
    #                    using_native_amp: bool = None, using_lbfgs: bool = None):
    #     lr = self.hparams['learning_rate']
    #     lr_schedule = self.hparams['lr_schedule']
    #     drop_steps = self.hparams['lr_drop_steps']
    #     drop_rate = self.hparams['lr_drop_rate']
    #     warmup_steps = self.hparams['warmup_steps']
    #     if not self.onTeacher:
    #         lr = self.hparams['student_learning_rate']
    #         lr_schedule = self.hparams['student_lr_schedule']
    #         drop_steps = self.hparams['student_lr_drop_steps']
    #         drop_rate = self.hparams['student_lr_drop_rate']
    #         warmup_steps = self.hparams['student_warmup_steps']

    #     if lr_schedule == 'constant':
    #         curLR = lr
    #     elif lr_schedule == 'warmup' or lr_schedule == 'warmupWithDrop':
    #         if self.trainer.global_step < warmup_steps:
    #             curLR = lr * min(1., float(self.trainer.global_step + 1) / warmup_steps)
    #         elif self.trainer.global_step < drop_steps or lr_schedule != 'warmupWithDrop':
    #             curLR = lr
    #         else:
    #             curLR = lr / drop_rate
    #     elif lr_schedule == 'cyclic':
    #         curLR = self.scheduler.get_last_lr()[0]

    #     else:
    #         raise NotImplementedError
    #     self.logger.experiment.track(curLR, name='lr', context={'model':self.onTeacher, 'stage':self.stage})

    #     for pg in optimizer.param_groups:
    #         pg['lr'] = curLR
    #         if not self.onTeacher:
    #             pg['weight_decay'] = 0

    #     # update params
    #     optimizer.step(closure=optimizer_closure)
    #     if lr_schedule =='cyclic':
    #         self.scheduler.step()

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum,
                              weight_decay=self.weight_decay, nesterov=False)

        if self.hparams['lr_schedule'] == 'cyclic':
            self.scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0, max_lr=self.lr, step_size_up=500)
            return {'optimizer': optimizer, 'lr_scheduler': self.scheduler}

        return {'optimizer': optimizer}

    def fit_model(self):
        # print("yeeeeeeet", self.hparams['oracle_train_on_labeled'])
        if self.hparams['teacher_init_path']:
            print("Loading teacher model from: {}".format(self.hparams['teacher_init_path']))
            self.load_checkpoint_teacher(self.hparams['teacher_init_path'], self.hparams['teacher_init_skip_last_layer'])

        if not self.hparams['skip_burn_in']:
            print("Starting teacher")
            self.onTeacher = True
            print("Will train the teacher for {} epochs, validate every {} epochs".format(
                self.hparams['total_steps_teacher'] // self.batches_per_epoch,
                self.check_val_epochs
            ))

            self.validation_counter = 0
            self.teacher_trainer.fit(self)
            print("Finished teacher")

        # self.load_best_teacher() # TODO I do not think this will always work
        # The best teacher is the last one, as we do not know how to measure what is the best one

        if self.stage != 7:
            self.copy_student_from_current_teacher()
            # self.copy_phd_from_current_teacher()
            for param in self.teacher.parameters():
                param.requires_grad = False
            self.teacher.eval()

            if self.hparams['oracle_train_on_labeled']:
                print("training a new orracle model on labeled")
                new_oracle_path = self.train_cnn_on_labeled()
                self.hparams['oracle_model_path'] = new_oracle_path
                print("Overriding oracle_model_path to: " + new_oracle_path)
            
            # opt = self.optimizers()[0]

            self.validation_counter = 0
            self.onTeacher = False

            print("starting student")
            
            print("Will train the student for {} epochs, validate every {} epochs".format(
                self.hparams['total_steps_student'] // self.batches_per_epoch,
                self.check_val_epochs
            ))
            self.student_trainer.fit(self)
            print("finished student")

        print('teacher loss: ', self.best_teacher_val)
        print('student loss: ', self.best_student_val)

        return (self.best_teacher_val, self.best_student_val)

    def test(self):
        headers = ['id', 'confidence', 'class', 'bbox']
        print('yeeet', self.output_csv)
        self.csvwriter_file = open(self.output_csv, 'w+', newline='')
        self.csvwriter = csv.writer(self.csvwriter_file)
        self.csvwriter.writerow(headers)

        if 1: #self.testWithStudent and not self.onlyBurnIn:
            print('testing with student')
            self.student_test_trainer.test(model=self)
        else:
            print('testing with teacher')
            self.teacher_test_trainer.test(model=self)

