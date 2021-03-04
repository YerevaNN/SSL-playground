from argparse import Namespace
import os
import csv
import numpy as np
import json

import torch
from .nets.detection.faster_rcnn import fasterrcnn_resnet50_fpn
import torch.optim as optim
from torch import nn
from torchvision.utils import save_image
from pytorch_lightning import Trainer

from .meanAP import compute_map

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TestTubeLogger

from typing import Dict

from .dataloader import get_train_test_loaders
# from .stac_trainer import StacTrainer

def make_target_from_y(y):
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

def break_batch(batch):
    x, y, paths = [], [], []
    for img in batch:
        x.append(img[0])
        y.append(img[1])
        paths.append(img[2])
    return x, y, paths

def label_id_to_name(id):
    if id == '0':
        return 'car'
    else:
        return 'pool'


def makeFolder(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def add_to_annotation(image_id, xmin, ymin, xmax, ymax, label):
    with open('./input/ground-truth/' + str(image_id) + '.txt', 'a+') as f:
        label = str(label)
        f.write(label + ' ' + str(xmin) + ' ' + str(ymin) + ' ' + str(xmax) + ' ' + str(ymax) + '\n')


def add_to_preditcion(image_id, xmin, ymin, xmax, ymax, label, confidence):
    with open('./input/detection-results/' + str(image_id) + '.txt','a+') as f:
        label = str(label)
        f.write(label + ' ' + str(confidence) + ' ' + str(xmin) + ' ' + str(ymin) + ' ' + str(xmax) + ' ' + str(ymax) + '\n')


def clear_folder(path):
    for f in os.listdir(path):
        os.remove(path + '/' + f)


class SkipConnection(nn.Module):

    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):
        z = self.module(x)
        return torch.cat((x, z), dim=-1)


def model_changed_classifier(classifier_type='added', pretrained='True', class_num=20):

    model = fasterrcnn_resnet50_fpn(pretrained=pretrained, progress=True,
                                    num_classes=91, pretrained_backbone=False)

    if classifier_type == 'added':
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
    elif classifier_type == 'concatenated':
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


class STAC(pl.LightningModule):

    def smth(self):
        self.on_batch_start()

    def __init__(self, hparams: Namespace) -> None:
        super().__init__()

        self.hparams = hparams

        self.lr = self.hparams['learning_rate']
        self.stage = self.hparams['stage']
        self.confidence_treshold = self.hparams['confidence_treshold']
        self.zero_counter = 0
        self.last_unsupervised_loss = 0
        self.training_map = 1
        self.best_teacher_val = 1000
        self.best_student_val = 1000
        self.best_val_loss = 10000000
        self.validation_counter = 0
        self.eta_min = self.lr * self.hparams['min_lr_ratio']
        self.num_warmup_steps = self.hparams['num_warmup_steps']
        self.total_steps = self.hparams['total_steps']
        self.early_stopping_patience = self.hparams['patience_epochs']
        self.with_SWA = self.hparams['with_SWA']
        self.validation_part = self.hparams["validation_part"]
        self.lam = self.hparams['consistency_lambda']
        self.max_lam = self.hparams['max_lam']
        self.momentum = self.hparams['momentum']
        self.weight_decay = self.hparams['weight_decay']
        self.consistency_criterion = self.hparams['consistency_criterion']
        self.testWithStudent = True
        self.output_csv = self.hparams['output_csv']
        self.onTeacher = True

        self.save_dir_name_student = os.getcwd() + "/checkpoints_student/{}_{}/version_{}/".format(self.hparams['experiment_name'],
                                                                                                  self.hparams['model'],
                                                                                                  self.hparams['version_name'])
        self.save_dir_name_teacher = os.getcwd() + "/checkpoints_teacher/{}_{}/version_{}/".format(self.hparams['experiment_name'],
                                                                                                  self.hparams['model'],
                                                                                                  self.hparams['version_name'])

        self.teacher = model_changed_classifier(classifier_type=self.hparams['classifier_type'],
                                            pretrained='True', class_num=self.hparams['class_num'])

        self.student = model_changed_classifier(classifier_type=self.hparams['classifier_type'],
                                            pretrained='True', class_num=self.hparams['class_num'])

        self.make_teacher_trainer()
        self.make_student_trainer()

        with open("./train_logits.json", "w+") as jsonFile:
            json.dump([], jsonFile)

        with open("./logits.json", "w+") as jsonFile:
            jsonFile.write('')

        makeFolder('./input')
        makeFolder('./input/ground-truth')
        makeFolder('./input/detection-results')

    def set_test_with_student(self, val):
        self.testWithStudent = val

    def print_yeet(self):
        print('yeeeet')

    def save_checkpoint(self, path):
        torch.save(self.student.state_dict(), path)

    def load_checkpoint(self, path):
        pretrained_dict = torch.load(path)
        model_dict = self.student.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if 'cls_score' not in k and 'bbox_pred' not in k}
        model_dict.update(pretrained_dict)
        self.student.load_state_dict(model_dict)

    def load_best_teacher(self):
        checkpoint_name = os.listdir(self.save_dir_name_teacher)[0]
        checkpoint_path = os.path.join(self.save_dir_name_teacher, checkpoint_name)
        best_dict = torch.load(checkpoint_path)
        actual_dict = {k[8:]: v for k, v in best_dict['state_dict'].items() if k.startswith('teacher')}
        self.teacher.load_state_dict(actual_dict)

    def copy_student_from_best_teacher(self):
        checkpoint_name = os.listdir(self.save_dir_name_teacher)[0]
        checkpoint_path = os.path.join(self.save_dir_name_teacher, checkpoint_name)
        best_dict = torch.load(checkpoint_path)
        actual_dict = {k[8:]: v for k, v in best_dict['state_dict'].items() if k.startswith('teacher')}
        self.student.load_state_dict(actual_dict)

    def load_checkpoint_teacher(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        actual_dict = {k[8:]: v for k, v in checkpoint['state_dict'].items() if k.startswith('teacher')}
        self.teacher.load_state_dict(actual_dict)

    def load_checkpoint_teacher(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        actual_dict = {k[8:]: v for k, v in checkpoint['state_dict'].items() if k.startswith('teacher')}
        self.teacher.load_state_dict(actual_dict)

    def load_checkpoint_student(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        actual_dict = {k[8:]: v for k, v in checkpoint['state_dict'].items() if k.startswith('student')}
        self.student.load_state_dict(actual_dict)

    def test_from_checkpoint(self, checkpoint_path):
        print('Testing with this checkpoint: {}'.format(checkpoint_path))
        if self.testWithStudent:
            self.load_checkpoint_student(checkpoint_path)
        else:
            self.load_checkpoint_teacher(checkpoint_path)
        self.test()

    def test_from_best_checkpoint(self):
        if self.testWithStudent:
            checkpoint_name = os.listdir(self.save_dir_name_student)[0]
            checkpoint_path = os.path.join(self.save_dir_name_student, checkpoint_name)
        else:
            checkpoint_name = os.listdir(self.save_dir_name_teacher)[0]
            checkpoint_path = os.path.join(self.save_dir_name_teacher, checkpoint_name)
        self.test_from_checkpoint(checkpoint_path)


    def set_datasets(self, labeled_file_path, unlabeled_file_path, testing_file_path, label_root):
        loaders = get_train_test_loaders(labeled_file_path,
                                         unlabeled_file_path,
                                         testing_file_path,
                                         label_root,
                                         self.hparams['batch_size'],
                                         self.hparams['num_workers'],
                                         stage=self.stage,
                                         validation_part=self.validation_part,
                                         unlabelled_batch_size=self.hparams['unlabelled_batch_size'])
        self.train_loader, self.test_loader, self.val_loader = loaders

    def make_teacher_trainer(self):

        self.t_checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            dirpath=self.save_dir_name_teacher,
            filename='{epoch}',
            verbose=True,
            # save_last=True,
            save_top_k=1,
            period=1
        )
        tt_logger = TestTubeLogger(
            save_dir="logs",
            name="{}_{}".format(self.hparams['experiment_name'], self.hparams['model']),
            version=self.hparams['version_name'],
            debug=False,
            create_git_tag=False,
        )
        early_stop_callback = EarlyStopping(
            monitor='val_loss',
            min_delta=0.0001,
            patience=self.early_stopping_patience,
            verbose=False,
            mode='min'
        )

        self.teacher_trainer = Trainer(gpus=-1,
                                  checkpoint_callback=True,
                                  callbacks=[early_stop_callback, self.t_checkpoint_callback],
                                  logger=tt_logger,
                                  progress_bar_refresh_rate=1,
                                  check_val_every_n_epoch=1,
                                  gradient_clip_val=self.hparams['gradient_clip_threshold'],
                                  #  val_check_interval=0.3,
                                  max_epochs=self.hparams['max_epochs'],
                                  min_epochs=self.hparams['min_epochs'],
                                  log_every_n_steps=1)

    def make_student_trainer(self):

        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            dirpath=self.save_dir_name_student,
            verbose=True,
            filename='{epoch}',
            save_top_k=1,
            # save_last=True,
            period=1
        )
        tt_logger = TestTubeLogger(
            save_dir="student_logs",
            name="{}_{}".format(self.hparams['experiment_name'], self.hparams['model']),
            version=self.hparams['version_name'],
            debug=False,
            create_git_tag=False,
        )
        early_stop_callback = EarlyStopping(
            monitor='val_loss',
            min_delta=0.0001,
            patience=self.early_stopping_patience,
            verbose=False,
            mode='min'
        )

        self.student_trainer = Trainer(gpus=-1,
                                  checkpoint_callback=True,
                                  callbacks=[early_stop_callback, checkpoint_callback],
                                  logger=tt_logger,
                                  progress_bar_refresh_rate=1,
                                  check_val_every_n_epoch=1,
                                  gradient_clip_val=self.hparams['gradient_clip_threshold'],
                                  #  val_check_interval=0.3,
                                  max_epochs=self.hparams['max_epochs'],
                                  min_epochs=self.hparams['min_epochs'],
                                  log_every_n_steps=1)
        

    def student_forward(self, x, image_paths):
        return self.student.forward(x, image_paths=image_paths)

    def teacher_forward(self, x, image_paths):
        return self.teacher.forward(x, image_paths=image_paths)
    
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
        x, y, image_paths = break_batch(sup_batch)

        target = make_target_from_y(y)

        y_hat = self.teacher(x, target, image_paths)

        return self.frcnn_loss(y_hat)

    def student_supervised_step(self, sup_batch):
        x, y, image_paths = break_batch(sup_batch)

        target = make_target_from_y(y)
        
#         print('xxxxxx: ', x)
        y_hat = self.student(x, target, image_paths)

        return self.frcnn_loss(y_hat)

    def student_unsupervised_step(self, unsup_batch):
        unlabeled, augmented = [], []
        for i in unsup_batch:
            unlab, augment = i
            unlabeled.append(unlab)
            augmented.append(augment)

        unlab_x = []
        # unlab_y = []
        unlab_image_paths = []
        for i in range(len(unlabeled)):
            unlab_x.append(unlabeled[i][0][0])
            unlab_image_paths.append(unlabeled[i][1])

        aug_x = []
        aug_vec = []
        aug_image_paths = []
        for i in range(len(augmented)):
            aug_x.append(augmented[i][0][0])
            aug_image_paths.append(augmented[i][1])


        self.teacher.eval()
        unlab_pred = self.teacher_forward(unlab_x, unlab_image_paths)

        to_train = True

        target = []

        for i, sample_pred in enumerate(unlab_pred):
            boxes = sample_pred['boxes'].cpu()
            labels = sample_pred['labels'].cpu()
            scores = sample_pred['scores'].cpu()

            index = []
            target_boxes = []
            target_labels = []
            for j in range(len(labels)):
                if scores[j] < self.confidence_treshold:
                    index.append(j)

            if len(boxes) != 0 and len(index) == len(boxes):
                del(index[0])

            boxes = np.delete(boxes, index, axis=0)
            labels = np.delete(labels, index)
            scores = np.delete(scores, index)

            # TODO move to the device of the model

            if len(boxes) == 0:
                to_train = False
                break

            for j, box in enumerate(boxes):
                target_boxes.append([box[0], box[1], box[2], box[3]])
                target_labels.append(labels[j])

            tensor_boxes = torch.tensor(target_boxes).float().cuda()
            tensor_labels = torch.tensor(target_labels).long().cuda()
            
            target.append({'boxes': tensor_boxes, 'labels': tensor_labels})
        if to_train:
            augment_pred = self.student(aug_x, target, aug_image_paths)
            unsup_loss = self.frcnn_loss(augment_pred)
        else:
            unsup_loss = 0

        return unsup_loss

    def teacher_training_step(self, batch_list):
        sup_batch, _ = batch_list
        self.teacher.set_is_supervised(True)

        sup_loss = self.teacher_supervised_step(sup_batch)

        loss = sup_loss

        self.log('training_sup_loss', sup_loss)
        self.log('training_loss', loss)
        return {'loss': loss}

    def student_training_step(self, batch_list):
        sup_batch, unsup_batch = batch_list
        self.student.set_is_supervised(True)

        sup_loss = self.student_supervised_step(sup_batch)
        self.student.set_is_supervised(False)

        unsup_loss = self.student_unsupervised_step(unsup_batch)

        loss = sup_loss + self.lam * unsup_loss

        self.log('training_sup_loss', sup_loss)
        self.log('training_unsup_loss', unsup_loss)
        self.log('training_loss', loss)
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
        return self.val_loader

    def validation_step(self, batch, batch_idx):
        if self.stage == 0 or self.validation_part == 0:
            return {}

        x, y, image_paths = batch
        
        y_hat = self.forward(x, image_paths)
        
        for i in range(len(y_hat)):
            img_id = batch_idx * self.hparams['batch_size'] + i
            boxes = y_hat[i]['boxes'].cpu().numpy()
            labels = y_hat[i]['labels'].cpu().numpy()
            scores = y_hat[i]['scores'].cpu().numpy()

            for (j, box) in enumerate(boxes):
                add_to_preditcion(img_id, box[0], box[1], box[2], box[3], labels[j], scores[j])
            if len(boxes) == 0:
                self.zero_counter += 1
                f = open('./input/detection-results/' + str(img_id) + '.txt','a+')
                f.close()

            for box in y[i]:
                xmin = int(box['bndbox']['xmin'])
                xmax = int(box['bndbox']['xmax'])
                ymin = int(box['bndbox']['ymin'])
                ymax = int(box['bndbox']['ymax'])
                add_to_annotation(img_id, xmin, ymin, xmax, ymax, int(box['label']))

        return {}

    def validation_epoch_end(self, results):
        self.validation_counter += 1
        if self.stage == 0 or self.validation_part == 0:
            # self.log('val_loss', -self.validation_counter)
            val_loss = -self.validation_counter
        else:
            print("computing mAP in validation_end")
            try:
                mAP = compute_map()
            except Exception as e:
                print("Could not compute mAP")
                print(e)
                print("Setting mAP=0")
                mAP = 0
            print("\nmAP = {:.3f}\n".format(mAP))

            clear_folder('./input/detection-results')
            clear_folder('./input/ground-truth')

            yeet = self.zero_counter
            self.zero_counter = 0
            val_loss = 1 - mAP - self.validation_counter / 1e9
            self.log('map', mAP)
            self.log('images_without_box', yeet)
        print('val_loss: ', val_loss)
        print('best_val_loss: ', self.best_val_loss)
        if self.onTeacher:
            self.best_teacher_val = min(self.best_teacher_val, val_loss)
        else:
            self.best_student_val = min(self.best_student_val, val_loss)
        self.best_val_loss = min(self.best_val_loss, val_loss)
        return {
            'val_loss': val_loss
        }

    def test_step(self, batch, batch_idx):
        x, names, image_paths = batch
        
        if self.testWithStudent:
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
                # add_to_preditcion(batch_idx, box[0], box[1], box[2], box[3], labels[i], scores[i]) # what is this? :)

        self.csvwriter.writerows(rows)

        return {'test_acc': 0}

    def on_test_end(self, batch):
        return {'test_acc': 0.5}

    def compute_accuracy(self, y, y_hat):
        y_hat = y_hat.argmax(dim=-1)
        return torch.sum(y == y_hat).item() / len(y)

    def optimizer_step(self, current_epoch, batch_nb, optimizer, optimizer_idx, closure, on_tpu=False,
                       using_native_amp=False, using_lbfgs=False):
        # warm up lr
        if self.trainer.global_step < self.num_warmup_steps:
            lr_scale = min(1., float(self.trainer.global_step + 1) / self.num_warmup_steps)
        else:
            lr_scale = 1 - (self.trainer.global_step - self.num_warmup_steps) / self.total_steps

        for pg in optimizer.param_groups:
            pg['lr'] = lr_scale * self.lr

        if self.trainer.global_step % 100 == 0:
            print("inside optimizer: {} LR scale={:.5f} @ step={}".format(
                "teacher" if self.onTeacher else "student",
                lr_scale, self.trainer.global_step
            ))

        # update params
        optimizer.step(closure=closure)

    def optimizer_zero_grad(self, current_epoch, batch_idx, optimizer, opt_idx):
        optimizer.zero_grad()

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum,
                              weight_decay=self.weight_decay, nesterov=False)
        # optimizer_sched = optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=self.eta_min,
        #                                                        T_max=(len(self.train_labeled_loader)*self.num_epochs - self.num_warmup_steps))

        return optimizer

    def fit_model(self):
        print("starting teacher")
        self.onTeacher = True

        self.validation_counter = 0
        self.teacher_trainer.fit(self)
        print("finished teacher")

        self.load_best_teacher() # TODO I do not think this will always work

        # if False:
        if self.stage != 7:
            self.copy_student_from_best_teacher()
            for param in self.teacher.parameters():
                param.requires_grad = False
            self.validation_counter = 0
            self.onTeacher = False

            print("starting student")
            self.student_trainer.fit(self)
            print("finished student")

        print('teacher loss: ', self.best_teacher_val)
        print('student loss: ', self.best_student_val)

        return (self.best_teacher_val, self.best_student_val)

    def test(self):
        headers = ['id', 'confidence', 'class', 'bbox']
        f = open(self.output_csv, 'w', newline='')
        self.csvwriter = csv.writer(f)
        self.csvwriter.writerow(headers)
#         self.student_trainer.test(model=self)
        if self.testWithStudent:
            print('testing with student')
            self.student_trainer.test(model=self)
        else:
            print('testing with teacher')
            self.teacher_trainer.test(model=self)

    def load(self):
        self.load_from_checkpoint(self.save_dir_name_student)
        self.eval()
