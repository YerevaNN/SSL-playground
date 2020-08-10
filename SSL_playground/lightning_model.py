from argparse import Namespace
import os
import numpy as np

import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torch.nn.functional import cross_entropy, mse_loss, kl_div
import torch.optim as optim
from torch import nn

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.logging import TestTubeLogger

from .dataloader import get_train_test_loaders
from .helpers.tsa import TrainingSignalAnnealing
from .nets.fastresnet import FastResnet
from .uda_trainer import UdaTrainer

from .meanAP import compute_map

coco_to_voc = {
    5: 1, # plane
    2: 2, # bicycle
    16: 3, # bird
    9: 4, # boat
    44: 5, # bottle
    6: 6, # bus
    3: 7, # car
    17: 8, # cat
    62: 9, # chair
    21: 10, # cow
    67: 11, # table
    18: 12, # dog
    19: 13, # horse
    4: 14, # bike
    1: 15, # person
    64: 16, # potted plant
    20: 17, # sheep
    63: 18, # sofa
    7: 19, # train
    72: 20 # tv
}

coco_to_voc_label_name = {
    'aeroplane': 'plane',
    'diningtable': 'table',
    'motorbike': 'bike',
    'tvmonitor': 'tv'
}

voc_class_name = {
    1: 'plane',
    2: 'bicycle',
    3: 'bird',
    4: 'boat',
    5: 'bottle',
    6: 'bus',
    7: 'car',
    8: 'cat',
    9: 'chair',
    10: 'cow',
    11: 'table',
    12: 'dog',
    13: 'horse',
    14: 'bike',
    15: 'person',
    16: 'pottedplant',
    17: 'sheep',
    18: 'sofa',
    19: 'train',
    20: 'tv'
}

voc_label_to_id = {
    'plane': 1,
    'aeroplane': 1,
    'bicycle': 2,
    'bird': 3,
    'boat': 4,
    'bottle': 5,
    'bus': 6,
    'car': 7,
    'cat': 8,
    'chair': 9,
    'cow': 10,
    'table': 11,
    'diningtable': 11,
    'dog': 12,
    'horse': 13,
    'bike': 14,
    'motorbike': 14,
    'person': 15,
    'pottedplant': 16,
    'sheep': 17,
    'sofa': 18,
    'train': 19,
    'tv': 20,
    'tvmonitor': 20,
}

def add_to_annotation(image_id, xmin, ymin, xmax, ymax, label):
    with open('/home/khazhak/SSL-playground/SSL_playground/input/ground-truth/' + str(image_id) + '.txt', 'a+') as f:
        label.replace(' ', '')
        if label in coco_to_voc_label_name.keys():
            label = coco_to_voc_label_name[label]
        f.write(label + ' ' + str(xmin) + ' ' + str(ymin) + ' ' + str(xmax) + ' ' + str(ymax) + '\n')

def add_to_preditcion(image_id, xmin, ymin, xmax, ymax, label, confidence):
    with open('/home/khazhak/SSL-playground/SSL_playground/input/detection-results/' + str(image_id) + '.txt','a+') as f:
        label = voc_class_name[label].replace(' ', '')
        f.write(label + ' ' + str(confidence) + ' ' + str(xmin) + ' ' + str(ymin) + ' ' + str(xmax) + ' ' + str(ymax) + '\n')

def clear_folder(path):
    for f in os.listdir(path):
        os.remove(path + '/' + f)

class UDA(pl.LightningModule):
    def __init__(self, hparams:Namespace) -> None:
        super().__init__()

        self.hparams = hparams
        hparams = hparams.__dict__

        self.lr = hparams['learning_rate']
        self.zero_counter = 0
        self.eta_min = self.lr * hparams['min_lr_ratio']
        self.num_warmup_steps = hparams['num_warmup_steps']
        self.with_SWA = hparams['with_SWA']
        self.lam = hparams['consistency_lambda']
        self.max_lam = hparams['max_lam']
        self.num_epochs = hparams['num_epochs']
        self.momentum = hparams['momentum']
        self.weight_decay = hparams['weight_decay']
        self.consistency_criterion = hparams['consistency_criterion']
        
        self.save_dir_name = os.getcwd() + "/checkpoints/{}_{}/version_{}".format(hparams['experiment_name'], hparams['model'],\
                             hparams['version_name'])


        self.net = fasterrcnn_resnet50_fpn(pretrained=False, progress=True,
                                 num_classes=21, pretrained_backbone=False)
        
        new_cls_score = nn.Sequential(
            nn.Linear(in_features=1024, out_features=91, bias=True),
            nn.Linear(in_features=91, out_features=21, bias=True)
        )

        new_bbox_pred = nn.Sequential(
            nn.Linear(in_features=1024, out_features=364, bias=True),
            nn.Linear(in_features=364, out_features=84, bias=True)
        )

        self.net.roi_heads.box_predictor.cls_score = new_cls_score

        self.net.roi_heads.box_predictor.bbox_pred = new_bbox_pred

        self.net.load_state_dict(torch.load('model_added_classifier.pth'))

        for param in self.net.parameters():
            param.requires_grad = True

        # self.net.roi_heads.box_predictor.cls_score[1].weight.requires_grad = True
        # self.net.roi_heads.box_predictor.cls_score[1].bias.requires_grad = True
        # self.net.roi_heads.box_predictor.bbox_pred[1].weight.requires_grad = True
        # self.net.roi_heads.box_predictor.bbox_pred[1].bias.requires_grad = True
    
    def set_datasets(self, train_dataset, test_dataset, number_of_classes):
        hparams = self.hparams.__dict__
        self.train_labeled_loader, self.train_unlabeled_loader, self.valid_loader = get_train_test_loaders(train_dataset,
                                                                                                           test_dataset,
                                                                      number_of_classes,
                                                                      hparams['num_labelled_samples'],
                                                                      hparams['batch_size'],
                                                                      hparams['num_workers'],
                                                                      hparams['unlabelled_batch_size'])
        self.tsa = tsa = TrainingSignalAnnealing(num_steps=len(self.train_labeled_loader)*self.num_epochs,
                                  min_threshold=hparams['TSA_proba_min'],
                                  max_threshold=hparams['TSA_proba_max'])

    def forward(self, x):
        return self.net.forward(x)

    @pl.data_loader
    def train_dataloader(self):
        return [self.train_labeled_loader, self.train_unlabeled_loader]

    def training_step(self, batch_list, batch_inx):
        sup_batch, unsup_batch = batch_list
        x, y = sup_batch

        target = []
        for i in range(len(y)):
            if type(y[i]['annotation']['object']) is dict:
                y[i]['annotation']['object'] = [y[i]['annotation']['object']]
            target_boxes = []
            target_labels = []
            for box in y[i]['annotation']['object']:
                xmin = int(box['bndbox']['xmin'])
                xmax = int(box['bndbox']['xmax'])
                ymin = int(box['bndbox']['ymin'])
                ymax = int(box['bndbox']['ymax'])
                label = voc_label_to_id[box['name']]
                target_boxes.append([xmin, ymin, xmax, ymax])
                target_labels.append(label)
            tensor_boxes = torch.cuda.FloatTensor(target_boxes, device = 'cuda')
            tensor_labels = torch.cuda.LongTensor(target_labels, device = 'cuda')
            target.append({'boxes': tensor_boxes, 'labels': tensor_labels})

        y_hat = self.net(x, target)

        sup_loss = y_hat['loss_classifier'] + 10 * y_hat['loss_box_reg']

        unlabeled, augmented = unsup_batch
        unlab_x = []
        unlab_y = []
        for i in range(len(unlabeled)):
            unlab_x.append(unlabeled[i][0])
            unlab_y.append(unlabeled[i][1])
        # unlab_x, unlab_y = unlabeled[0]
        # unlab_x = unlab_x.unsqueeze(0)
        aug_x = []
        aug_vec = []
        for i in range(len(augmented)):
            aug_x.append(augmented[i][0])
            aug_vec.append(augmented[i][1])
        # aug_x, aug_vec = augmented[0]
        # aug_x = aug_x.unsqueeze(0)

        self.net.eval()
        with torch.no_grad():
            unlab_pred = self.forward(unlab_x)
        self.net.train()

        to_train = True

        target = []
        for i in range(len(unlab_pred)):
            boxes = unlab_pred[i]['boxes'].cpu().numpy()
            labels = unlab_pred[i]['labels'].cpu().numpy()
            scores = unlab_pred[i]['scores'].cpu().numpy()
            index = []
            target_boxes = []
            target_labels = []
            for j in range(len(labels)):
                if scores[j] < 0.1:
                    index.append(i)
            boxes = np.delete(boxes, index, axis = 0)
            labels = np.delete(labels, index)
            scores = np.delete(scores, index)
            if len(boxes) == 0:
                to_train = False
                break # TODO
            for (j, box) in enumerate(boxes):
                target_boxes.append([box[0], box[1], box[2], box[3]])
                target_labels.append(labels[j])
            tensor_boxes = torch.cuda.FloatTensor(target_boxes, device = 'cuda')
            tensor_labels = torch.cuda.LongTensor(target_labels, device = 'cuda')
            # print(tensor_boxes.shape)
            if len(tensor_boxes.shape) == 1:
                tensor_boxes = tensor_boxes.reshape(0, 4)
            # print(tensor_boxes.shape)
            target.append({'boxes': tensor_boxes, 'labels': tensor_labels})

        if to_train:
            print(len(aug_x))
            augment_pred = self.net(aug_x, target)
            unsup_loss = self.lam * augment_pred['loss_classifier'] + 10 * augment_pred['loss_box_reg']
        else:
            print("skipping a batch")
            unsup_loss = 0
        # if (self.current_epoch<50):
        #     self.lam = self.max_lam*(self.current_epoch/50)
        # else:
        #     self.lam = self.max_lam


        self.loss = sup_loss + self.lam * unsup_loss

        log_dict = {
            'training_sup_loss': sup_loss,
            'training_unsup_loss': unsup_loss,
            'training_loss': self.loss
        }

        return {'loss': self.loss, 'log': log_dict}

    @pl.data_loader
    def val_dataloader(self):
        return self.valid_loader

    def validation_step(self, batch, batch_idx):
        self.net.eval()
        x, y = batch
        y_hat = self.forward(x)
        # print(y_hat)
        # print(len(y))
        for i in range(len(y_hat)):
            boxes = y_hat[i]['boxes'].cpu().numpy()
            labels = y_hat[i]['labels'].cpu().numpy()
            scores = y_hat[i]['scores'].cpu().numpy()

            for (j, box) in enumerate(boxes):
                add_to_preditcion(batch_idx, box[0], box[1], box[2], box[3], labels[j], scores[j])
            if len(boxes) == 0:
                self.zero_counter += 1
                f = open('/home/khazhak/faster_rcnn_voc/input/detection-results/' + str(batch_idx) + '.txt','a+')
                f.close()

            if type(y[i]['annotation']['object']) is dict:
                y[i]['annotation']['object'] = [y[i]['annotation']['object']]
            for box in y[i]['annotation']['object']:
                xmin = int(box['bndbox']['xmin'])
                xmax = int(box['bndbox']['xmax'])
                ymin = int(box['bndbox']['ymin'])
                ymax = int(box['bndbox']['ymax'])
                add_to_annotation(batch_idx, xmin, ymin, xmax, ymax, box['name'])

        return {}

    def validation_end(self, outputs):
        mAP = compute_map()
        
        clear_folder('/home/khazhak/SSL-playground/SSL_playground/input/detection-results')
        clear_folder('/home/khazhak/SSL-playground/SSL_playground/input/ground-truth')

        self.net.train()
        yeet = self.zero_counter
        self.zero_counter = 0
        return {
            'val_loss': 1 - mAP,
            'log': {
                'map': mAP,
                'images_without_box': yeet
            }
        }

    def compute_accuracy(self, y, y_hat):
        y_hat = y_hat.argmax(dim=-1)
        return torch.sum(y == y_hat).item() / len(y)

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum,
                              weight_decay=self.weight_decay, nesterov=True)
        optimizer_sched = optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=self.eta_min,
                                                              T_max=(len(self.train_labeled_loader)*self.num_epochs - self.num_warmup_steps))

        return [optimizer], [optimizer_sched]

    # def final_loss(self, y_hat, y_true, unsup_aug_y_probas, unsup_orig_y_probas):
    #     return self.classification_loss(y_hat, y_true) + \
    #            self.lam * self.consistency_loss(unsup_aug_y_probas, unsup_orig_y_probas)

    def classification_loss(self, y_pred, y_true):
        return cross_entropy(y_pred, y_true)

    # def consistency_loss(self, unsup_aug_y_probas, unsup_orig_y_probas):
    #     if self.consistency_criterion == "MSE":
    #         return mse_loss(torch.softmax(unsup_aug_y_probas, dim=-1), torch.softmax(unsup_orig_y_probas, dim=-1))
    #     elif self.consistency_criterion == "KL":
    #         unsup_aug_y_probas = torch.log_softmax(unsup_aug_y_probas, dim=-1)
    #         unsup_orig_y_probas = torch.softmax(unsup_orig_y_probas, dim=-1)
    #         return kl_div(unsup_aug_y_probas, unsup_orig_y_probas, reduction='batchmean')


    def fit_model(self):
        hparams = self.hparams.__dict__
        # print('save_dir_name:', self.save_dir_name)
        checkpoint_callback = ModelCheckpoint(
            filepath=self.save_dir_name,
            verbose=True,
            save_top_k=-1,
            period=20
        )

        # print('ttl:', "logs", "{}_{}".format(hparams['experiment_name'], hparams['model']))
        # print(hparams['version_name'])
        tt_logger = TestTubeLogger(
            save_dir="logs",
            name="{}_{}".format(hparams['experiment_name'], hparams['model']),
            version=hparams['version_name'],
            debug=False,
            create_git_tag=False,
        )

        trainer = UdaTrainer(gpus=-1,
                             early_stop_callback=None,
                             logger=tt_logger,
                             show_progress_bar=True,
                            #  checkpoint_callback=checkpoint_callback,
                             check_val_every_n_epoch=2,
                            #  default_save_path="../checkpoints",
                            #  val_check_interval=0.3,
                             max_nb_epochs=hparams['num_epochs'],
                             min_nb_epochs=hparams['num_epochs'],
                             log_save_interval=1,
                             row_log_interval=1)

        trainer.fit(self)

    def load(self):
        self.load_from_checkpoint(self.save_dir_name)
        self.eval()
        #self.freeze()
