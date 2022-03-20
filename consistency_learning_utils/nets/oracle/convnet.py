from typing import List, Any
from pytorch_lightning.utilities.apply_func import from_numpy

import torch
from torch import nn
from torch.nn import MSELoss
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.nn.functional import cross_entropy

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from aim.pytorch_lightning import AimLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import os
import pandas as pd
import ast
from oracle import get_max_IOU
import zarr

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(256, 128, 3, padding=1, stride=1) # 258x7x7 -> 129x7x7
        self.conv2 = nn.Conv2d(128, 64, 3, padding=1, stride=1) # 64x7x77 -> 129x7x7
        self.conv3 = nn.Conv2d(64, 32, 3, padding=1, stride=1) # 32x7x7
        self.conv4 = nn.Conv2d(32, 16, 3, padding=1, stride=1) # 16x7x7
        self.conv5 = nn.Conv2d(16, 8, 3, padding=1, stride=1) # 8x7x7
        self.conv6 = nn.Conv2d(8, 4, 3, padding=1, stride=1) # 4x7x7
        self.conv7 = nn.Conv2d(4, 1, 3, padding=1, stride=1) # 1x7x7
        self.linear = nn.Linear(7 * 7, 1)
        self.float()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.conv7(x)
        x = torch.flatten(x, 1, -1)
        x = self.linear(x)
        x = torch.squeeze(x, 1)
        x = torch.sigmoid(x)
        return x


class MyDataset(Dataset):
    def __init__(self, samples, label_root, zarr_path):
        super().__init__()
        self.samples = samples
        self.label_root = label_root
        self.zarr_file = zarr.open(zarr_path, mode='r')

    def __len__(self):
        return sum([len(x[0]) + len(x[1]) for x in self.samples])

    def __get_target__(self, img_path, bbox, cl):
        max_iou = get_max_IOU(img_path, bbox, cl)
        return max_iou, cl

    def __getitem__(self, index):
        sample = self.samples[index % 3][index % 2][(index // 6) % len(self.samples[index % 3][index % 2])]
        img_path = os.path.join(self.label_root, sample.split('_')[0] + '.txt')
        bbox_key = sample + '_bbox'
        features_key = sample + '_feat'
        bbox = self.zarr_file[bbox_key]
        bbox = np.array(bbox, dtype="float32")
        features = self.zarr_file[features_key]
        features = torch.cuda.FloatTensor(features, device='cuda')
        cl = float(features[1][0][0])
        target = self.__get_target__(img_path, bbox, cl)

        return features, target


def get_train_test_loaders(samples, label_root, split, batch_size, zarr_path):

    split_idxs = [[int(len(sample[0]) * split), int(len(sample[1]) * split)] for sample in samples]

    train_samples = [[samples[i][0][:split_idxs[i][0]], samples[i][1][:split_idxs[i][1]]] for i in range(len(samples))]
    test_samples = [[samples[i][0][split_idxs[i][0]:], samples[i][1][split_idxs[i][1]:]] for i in range(len(samples))]

    
    train_dataset = MyDataset(train_samples, label_root, zarr_path)
    test_dataset = MyDataset(test_samples, label_root, zarr_path)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    return train_dataloader, test_dataloader


class Oracle(pl.LightningModule):
    def __init__(self, experiment_name, zarr_path = None, lr = 1):
        super().__init__()
        self.zarr_path = zarr_path
        self.lr = lr

        self.model = Net()
        self.mse = MSELoss()
        self.aim_logger = AimLogger(
            experiment=experiment_name
        )
        self.save_dir_name = os.getcwd() + "/checkpoints/{}".format(experiment_name)
        os.makedirs(self.save_dir_name, exist_ok=True)
        self.best_validation_accuracy = -1
        self.global_info_init()
        # checkpoint_callback = ModelCheckpoint(
        #     dirpath=self.save_dir_name,
        #     filename='{epoch}-{val_acc:.3f}',
        #     save_last=True,
        #     mode='max'
        # )
        self.trainer = Trainer(gpus=-1, logger=self.aim_logger, max_epochs=1000,
                            #    callbacks = [checkpoint_callback],
                            #    check_val_every_n_epoch=1)
                               val_check_interval = 0.2)
        # self.targets_file = os.getcwd() + "/ious_nightowls.npy"
        # self.ious = {i: [] for i in [1, 2, 3]}

    def global_info_init_val(self):
        self.global_info_val = {i: {
            "tp": 0,
            "fp": 0,
            "fn": 0
        } for i in [1, 2, 3]}
    
    def global_info_init_train(self):
        self.global_info_train = {i: {
            "tp": 0,
            "fp": 0,
            "fn": 0
        } for i in [1, 2, 3]}

    def global_info_init(self):
        self.global_info_init_val()
        self.global_info_init_train()
        self.global_info_test = {i: {
            "tp": 0,
            "fp": 0,
            "fn": 0
        } for i in [1, 2, 3]}

    def set_datasets(self, samples, label_root, split, batch_size):
        self.train_loader, self.test_loader = get_train_test_loaders(samples, label_root,
                                                                     split, batch_size,
                                                                     self.zarr_path)

    def load_from_path(self, path):
        sd = torch.load(path)
        self.model.load_state_dict(sd)

    def forward(self, x):
        return self.model(x)

    def train_dataloader(self):
        return self.train_loader

    def test_dataloader(self):
        return self.test_loader

    def val_dataloader(self):
        return self.test_loader

    def configure_optimizers(self):
        # optimizer = optim.SGD(self.parameters(), lr=self.lr, momentum=0.9)
        optimizer = optim.Adam(self.parameters(), lr=self.lr)

        return {'optimizer': optimizer}

    def loss_function(self, y_pred, y_truth):
        return torch.sqrt(self.mse(y_pred.float(), y_truth.float()))
    

    def training_step(self, batch, batch_inx):
        inputs, labels_and_classes = batch
        labels, classes = labels_and_classes[0], labels_and_classes[1]
        outputs = self.forward(inputs)
        loss = self.loss_function(outputs, labels)
        for i in range(len(outputs)):
            label = labels[i]
            output = outputs[i]
            cl = int(classes[i])
            # self.ious[cl].append(label.cpu().detach().numpy())
            if label >= 0.5 and output >= 0.5:
                self.global_info_train[cl]['tp'] += 1
            elif label < 0.5 and output >= 0.5:
                self.global_info_train[cl]['fp'] += 1
            elif label >= 0.5 and output < 0.5:
                self.global_info_train[cl]['fn'] += 1

        self.logger.experiment.track(loss.item(), name='training_loss')
        return {'loss': loss}

    def training_epoch_end(self, outputs: List[Any]) -> None:
        # np.save(self.targets_file, self.ious)
        fscores = self.f1_scores(self.global_info_train)
        self.logger.experiment.track(fscores[0], name='train_fscore_class_0')
        self.logger.experiment.track(fscores[1], name='train_fscore_class_1')
        self.logger.experiment.track(fscores[2], name='train_fscore_class_2')
        self.logger.experiment.track((fscores[0] + fscores[1] + fscores[2]) / 3, name='train_fscore_average')
        self.global_info_init_train()

    def test_step(self, batch, batch_inx):
        inputs, labels = batch
        outputs = self.forward(inputs)

        self.global_info_init()
        for i in range(len(outputs)):
            self.compute_accuracy(outputs[i][0][0].cpu().detach().numpy() > 0.5, labels[i][0][0].cpu().detach().numpy() > 0.5,
                                  self.test_cl_masks[i], phase='test')


        loss = self.loss_function(outputs, labels.float())

        self.logger.experiment.track(loss.item(), name='test_loss')

        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        inputs, labels_and_classes = batch
        labels, classes = labels_and_classes[0], labels_and_classes[1]
        outputs = self.forward(inputs)
        loss = self.loss_function(outputs, labels)
        for i in range(len(outputs)):
            label = labels[i]
            output = outputs[i]
            cl = int(classes[i])
            if label >= 0.5 and output >= 0.5:
                self.global_info_val[cl]['tp'] += 1
            elif label < 0.5 and output >= 0.5:
                self.global_info_val[cl]['fp'] += 1
            elif label >= 0.5 and output < 0.5:
                self.global_info_val[cl]['fn'] += 1

        self.logger.experiment.track(loss.item(), name='val_loss')

        return {'val_loss': loss}

    def validation_epoch_end(self, results):
        fscores = self.f1_scores(self.global_info_val)
        self.global_info_init_val()
        self.logger.experiment.track(fscores[0], name='fscore_class_0')
        self.logger.experiment.track(fscores[1], name='fscore_class_1')
        self.logger.experiment.track(fscores[2], name='fscore_class_2')
        accuracy = (fscores[0] + fscores[1] + fscores[2]) / 3
        if accuracy > self.best_validation_accuracy:
            print('achieved top validation accuracy, saving the checkpoint')
            self.best_validation_accuracy = accuracy
            torch.save(self.model.state_dict(), os.path.join(self.save_dir_name, 'best.ckpt'))
        self.logger.experiment.track((fscores[0] + fscores[1] + fscores[2]) / 3, name='fscore_average')
        print(fscores)
        return {'val_fscores': fscores}

    def fit_model(self):
        self.trainer.fit(self)

    def test(self):
        return self.trainer.test(self)

    def on_test_end(self):
        print(self.global_info_test)

    def compute_accuracy(self, y_hat, y, cl_masks, phase='train'):
        classes = list(cl_masks.keys())
        if phase == 'test':
            for cl in classes:
                self.global_info_test[cl]['tp'] += sum(y_hat[:len(cl_masks[cl])] & y[:len(cl_masks[cl])]
                                                  & cl_masks[cl])
                self.global_info_test[cl]['fp'] += sum(y_hat[:len(cl_masks[cl])] & ~y[:len(cl_masks[cl])]
                                                  & cl_masks[cl])
                self.global_info_test[cl]['fn'] += sum(~y_hat[:len(cl_masks[cl])] & y[:len(cl_masks[cl])]
                                                  & cl_masks[cl])
        elif phase == 'train':
            for cl in classes:
                self.global_info[cl]['tp'] += sum(y_hat[:len(cl_masks[cl])] & y[:len(cl_masks[cl])]
                                                  & cl_masks[cl])
                self.global_info[cl]['fp'] += sum(y_hat[:len(cl_masks[cl])] & ~y[:len(cl_masks[cl])]
                                                  & cl_masks[cl])
                self.global_info[cl]['fn'] += sum(~y_hat[:len(cl_masks[cl])] & y[:len(cl_masks[cl])]
                                                  & cl_masks[cl])

        else:
            for cl in classes:
                self.global_info_val[cl]['tp'] += sum(y_hat[:len(cl_masks[cl])] & y[:len(cl_masks[cl])]
                                                  & cl_masks[cl])
                self.global_info_val[cl]['fp'] += sum(y_hat[:len(cl_masks[cl])] & ~y[:len(cl_masks[cl])]
                                                  & cl_masks[cl])
                self.global_info_val[cl]['fn'] += sum(~y_hat[:len(cl_masks[cl])] & y[:len(cl_masks[cl])]
                                                  & cl_masks[cl])

    def f1_scores(self, dicts):
        fscores = []
        for k in dicts.keys():
            dct = dicts[k]
            tp = dct['tp']
            fp = dct['fp']
            fn = dct['fn']
            if tp + fp + fn == 0:
                fscores.append(0.)
            else:
                fscores.append(tp / (tp + (fp + fn) / 2))
        return fscores


def preprocess_predictions(pred):
    boxes_per_image = [len(p) for p in pred]
    # for boxes, p in enumerate(pred):
    #     if p.shape[0] < 100:
    #         zer = torch.zeros((100-p.shape[0], p.shape[1]))
    #         pred[i] = torch.cat((p, zer))
    #     pred[i] = torch.transpose(pred[i], 0, 1)
    #     i += 1
    for i, p in enumerate(pred):
        pred[i] = p.to(device='cuda')
    # pred = torch.unsqueeze(torch.stack(pred), dim=1)
    return boxes_per_image, pred

def select_pls(output, boxes_per_image, pred, iou_thresh):
    new_pred = []
    for i, p in enumerate(pred):
        pred_on_image = output[i].cpu()
        pred_on_image = torch.squeeze(pred_on_image)
        indices = torch.where(pred_on_image >= iou_thresh)
        if len(indices) == 0:
            print("There were no predictions above " + str(iou_thresh))
            indices.append(1)
        new_p = p[indices][:,:6]
        
        new_pred.append(new_p)
    return new_pred

def inference(pred, model_path, iou_thresh=0.5, experiment_name=""):
    model = Oracle(experiment_name)
    model.cuda()
    # model.load_from_path(model_path)
    old_pred = [(torch.clone(p1), torch.clone(p2)) for p1, p2 in pred]
    pred_boxes = [p[0] for p in pred]
    pred_features = [p[1] for p in pred]
    boxes_per_image, pred = preprocess_predictions(pred_features)
    output = model(pred)
    selected_predictions = select_pls(output, boxes_per_image, old_pred, iou_thresh)
    return selected_predictions
