from typing import List, Any
from pytorch_lightning.utilities.apply_func import from_numpy

from torchvision.transforms import ToTensor, Compose
import torch
from torch import nn
from torch.nn import MSELoss
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
import numpy as np
import random
import sys

sys.path.insert(1, '/home/hkhachatrian/SSL-playground/consistency_learning_utils/')

from lightning_model import model_changed_classifier
from lightning_model import change_prediction_format

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from aim.pytorch_lightning import AimLogger
import os
from oracle import get_max_IOU
from torch.nn import BatchNorm2d
from torchvision.ops import Conv2dNormActivation

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(256, 128, 3, padding=1, stride=1) # 256x7x7 -> 128x7x7
        self.conv2 = nn.Conv2d(128, 64, 3, padding=1, stride=1) # 64x7x7
        self.conv3 = nn.Conv2d(64, 32, 3, padding=1, stride=1) # 32x7x7
        self.conv4 = nn.Conv2d(32, 16, 3, padding=1, stride=1) # 16x7x7
        self.conv5 = nn.Conv2d(16, 8, 3, padding=1, stride=1) # 8x7x7
        self.conv6 = nn.Conv2d(8, 4, 3, padding=1, stride=1) # 4x7x7
        self.conv7 = nn.Conv2d(4, 1, 3, padding=1, stride=1) # 1x7x7
        self.linear = nn.Linear(7 * 7, 4)
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
        x = torch.softmax(x, dim = 1)
        return x


class FeatureExtractor():
    def __init__(self, class_num, box_score_thresh=0.05):
        self.phd = model_changed_classifier(
            initialize='full',
            reuse_classifier='add',
            class_num=class_num, # TODO
            gamma=0,
            box_score_thresh=box_score_thresh) # TODO
        self.phd.cuda()


    def get_features(self, image, boxes):
        self.phd.eval()
        predictions = self.phd.forward([image], teacher_boxes=boxes)
        features = torch.squeeze(predictions[0], 0)
        return features


class MyDataset(Dataset):
    def __init__(self, image_paths, label_root, class_num, box_score_thresh):
        super().__init__()
        self.feature_extractor = FeatureExtractor(class_num, box_score_thresh)
        self.image_paths = image_paths 
        self.label_root = label_root
        self.transform = Compose([ToTensor()])


    def __len__(self):
        return len(self.image_paths)

    def __get_target__(self, img_path, label_root, bbox):
        ind1 = img_path.rfind('/')
        ind2 = img_path.rfind('.')
        cur_image = img_path[ind1 + 1: ind2]
        img_path = os.path.join(label_root, cur_image + '.txt')
        max_iou = get_max_IOU(img_path, bbox)
        return max_iou

    def __getitem__(self, index: int):
        img_path, bbox, iou, conf, clas = self.image_paths[index].strip().split(' ')
        iou = int((float(iou) - 1e-7) * 4)
        bbox = [int(float(_)) for _ in bbox[1:-1].split(',')]
        bbox = torch.cuda.FloatTensor(bbox, device="cuda")
        bbox = bbox.unsqueeze(dim = 0)

        with open(img_path, 'rb') as f:
            image = Image.open(f)
            image = np.array(image)
        if len(image.shape) == 2:
            image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
        image = self.transform(image).cuda()
        if image.shape[0] != 3:
            image = torch.reshape(image, (image.shape[2], image.shape[0], image.shape[1])) # HxWxC to CxHxW

        features = self.feature_extractor.get_features(image, [bbox])

        target = torch.tensor((iou), device='cuda')
        return features, target, float(conf), int(clas)


def get_train_test_loaders(feature_data_path, label_root, split, batch_size, class_num, box_score_thresh):
    file_lines = []
    # Format of the file must be like this:
    # /home/...../...png [xmin,ymin,xmax,ymax] iou conf cls
    with open(feature_data_path) as f:
        file_lines = f.readlines()
    random.shuffle(file_lines)


    split_id = int(len(file_lines) * split)
    train_image_paths = file_lines[:split_id]
    test_image_paths = file_lines[split_id:]


    
    train_dataset = MyDataset(train_image_paths, label_root, class_num, box_score_thresh)
    test_dataset = MyDataset(test_image_paths, label_root, class_num, box_score_thresh)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    return train_dataloader, test_dataloader


class Oracle(pl.LightningModule):
    def __init__(self, experiment_name, lr = 1, class_num = 91):
        super().__init__()
        self.class_num = class_num
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
        #     mode='max',
        #     monitor='val_accuracy'
        # )
        self.trainer = Trainer(gpus=-1, max_epochs=1000,
                               logger=self.aim_logger,
                            #    callbacks = [checkpoint_callback],
                               gradient_clip_val=0.5,
                            #    check_val_every_n_epoch=1)
                               val_check_interval = 0.02)
        self.loss_fn = nn.CrossEntropyLoss()
        # self.targets_file = os.getcwd() + "/ious_nightowls.npy"
        # self.ious = {i: [] for i in [1, 2, 3]}

    def global_info_init_val(self):
        self.global_info_val = {i: {
            "tp": 0,
            "fp": 0,
            "fn": 0
        } for i in range(1, self.class_num + 1)}
    
    def global_info_init_train(self):
        self.global_info_train = {i: {
            "tp": 0,
            "fp": 0,
            "fn": 0
        } for i in range(1, self.class_num + 1)}

    def global_info_init(self):
        self.global_info_init_val()
        self.global_info_init_train()
        self.global_info_test = {i: {
            "tp": 0,
            "fp": 0,
            "fn": 0
        } for i in range(1, self.class_num + 1)}

    def set_datasets(self, feature_data_path, label_root, split, batch_size,
                     class_num, box_score_thresh):
        self.train_loader, self.test_loader = get_train_test_loaders(feature_data_path,
                                                                     label_root, split, batch_size,
                                                                     class_num,
                                                                     box_score_thresh)

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
        return self.loss_fn(y_pred, y_truth)


    def training_step(self, batch, batch_inx):
        inputs, labels, conf, clas = batch
        for i in range(len(inputs)):
            inputs[i] = torch.squeeze(inputs[i], 0)
        outputs = self.forward(inputs)
        loss = self.loss_function(outputs, labels)
        for i in range(len(outputs)):
            label = labels[i].item()
            output = torch.argmax(outputs[i]).item()
            cl = clas[i].item()
            # self.ious[cl].append(label.cpu().detach().numpy())
            if label >= 2 and output >= 2:
                self.global_info_train[cl]['tp'] += 1
            elif label < 2 and output >= 2:
                self.global_info_train[cl]['fp'] += 1
            elif label >= 2 and output < 2:
                self.global_info_train[cl]['fn'] += 1

        self.logger.experiment.track(loss.item(), name='training_loss')
        return {'loss': loss}

    def training_epoch_end(self, outputs: List[Any]) -> None:
        # np.save(self.targets_file, self.ious)
        fscores = self.f1_scores(self.global_info_train)
        # self.logger.experiment.track(fscores[0], name='train_fscore_class_0')
        # self.logger.experiment.track(fscores[1], name='train_fscore_class_1')
        # self.logger.experiment.track(fscores[2], name='train_fscore_class_2')
        self.logger.experiment.track(sum(fscores) / len(fscores), name='train_fscore_average')
        self.global_info_init_train()

    def test_step(self, batch, batch_inx):
        return self.validation_step(batch, batch_inx)

    def validation_step(self, batch, batch_idx):
        inputs, labels, conf, clas = batch
        for i in range(len(inputs)):
            inputs[i] = torch.squeeze(inputs[i], 0)
        outputs = self.forward(inputs)
        loss = self.loss_function(outputs, labels)
        for i in range(len(outputs)):
            label = labels[i].item()
            output = torch.argmax(outputs[i]).item()
            cl = clas[i].item()
            print(label, output, cl)
            if label >= 2 and output >= 2:
                self.global_info_val[cl]['tp'] += 1
            elif label < 2 and output >= 2:
                self.global_info_val[cl]['fp'] += 1
            elif label >= 2 and output < 2:
                self.global_info_val[cl]['fn'] += 1

        self.logger.experiment.track(loss.item(), name='val_loss')

        return {'val_loss': loss}

    def validation_epoch_end(self, results):
        fscores = self.f1_scores(self.global_info_val)
        self.global_info_init_val()
        # self.logger.experiment.track(fscores[0], name='fscore_class_0')
        # self.logger.experiment.track(fscores[1], name='fscore_class_1')
        # self.logger.experiment.track(fscores[2], name='fscore_class_2')
        accuracy = sum(fscores) / len(fscores)
        if accuracy > self.best_validation_accuracy:
            if self.best_validation_accuracy == -1: # To avoid saving on validation sanity check
                self.best_validation_accuracy = 0
            else:
                print('achieved top validation accuracy, saving the checkpoint')
                self.best_validation_accuracy = accuracy
                torch.save(self.model.state_dict(), os.path.join(self.save_dir_name, 'our_best.ckpt'))
        # self.logger.experiment.track(sum([fscores[i] for i in range(91)]) / 80, name='fscore_average')
        print('fscores: ', fscores)
        print('accuracy (avg fscore): ', accuracy)
        return {'val_accuracy': accuracy}
    
    def test_epoch_end(self, results):
        return self.validation_epoch_end(results)

    def fit_model(self):
        self.trainer.fit(self)

    def test(self):
        return self.trainer.test(self)

    def on_test_end(self):
        fscores = self.f1_scores(self.global_info_val)
        accuracy = sum(fscores) / len(fscores)
        print(fscores)
        print('average fscore', accuracy)

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
