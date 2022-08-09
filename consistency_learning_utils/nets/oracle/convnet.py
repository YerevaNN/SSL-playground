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

sys.path.insert(1, '/home/hkhachatrian/SSL-playground/')

from consistency_learning_utils.model_changed_classifier import model_changed_classifier
# from lightning_model import change_prediction_format

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from aim.pytorch_lightning import AimLogger
import os
from .oracle import get_max_IOU
from torch.nn import BatchNorm2d
# from torchvision.ops import Conv2dNormActivation

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(256, 128, 3, padding=1, stride=1) # 256x7x7 -> 128x7x7
        self.norm_layer1 = BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 64, 3, padding=1, stride=1) # 64x7x7
        self.norm_layer2 = BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 32, 3, padding=1, stride=1) # 32x7x7
        self.norm_layer3 = BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 16, 3, padding=1, stride=1) # 16x7x7
        self.norm_layer4 = BatchNorm2d(16)
        self.conv5 = nn.Conv2d(16, 8, 3, padding=1, stride=1) # 8x7x7
        self.norm_layer5 = BatchNorm2d(8)
        self.conv6 = nn.Conv2d(8, 4, 3, padding=1, stride=1) # 4x7x7
        self.norm_layer6 = BatchNorm2d(4)
        self.conv7 = nn.Conv2d(4, 1, 3, padding=1, stride=1) # 1x7x7
        # self.norm_layer7 = BatchNorm2d(2)
        self.linear = nn.Linear(7 * 7, 2)
        self.float()

    def forward(self, x):
        x = F.relu(self.norm_layer1(self.conv1(x)))
        x = F.relu(self.norm_layer2(self.conv2(x)))
        x = F.relu(self.norm_layer3(self.conv3(x)))
        x = F.relu(self.norm_layer4(self.conv4(x)))
        x = F.relu(self.norm_layer5(self.conv5(x)))
        x = F.relu(self.norm_layer6(self.conv6(x)))
        x = self.conv7(x)
        x = torch.flatten(x, 1, -1)
        x = self.linear(x)
        x = torch.squeeze(x, 1)
        x = torch.softmax(x, dim = 1)
        return x


class FeatureExtractor():
    def __init__(self, class_num, box_score_thresh=0.05):
        self.phd = model_changed_classifier(
            initialize='backbone',
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
    def __init__(self, image_paths, label_root, class_num, box_score_thresh, skip_images_path=None):
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
        try:
            img_path, bbox, iou, conf, clas = self.image_paths[index].strip().split(' ')
        except Exception as e:
            print(self.image_paths[index])
            raise e
        iou_with_gt = float(iou)
        iou = int((float(iou) - 1e-7) * 2)
        bbox = [int(float(_)) for _ in bbox[1:-1].split(',')]
        old_bbox = bbox
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
        conf = torch.tensor(float(conf), device='cuda')
        clas = torch.tensor(int(clas), device='cuda')
        
        return img_path, features, target, old_bbox, iou_with_gt, conf, clas


def get_train_test_loaders(feature_data_path, label_root, split, batch_size, class_num, box_score_thresh, skip_data_path=None):
    file_lines_old = []
    # Format of the file must be like this:
    # /home/...../...png [xmin,ymin,xmax,ymax] iou conf cls
    with open(feature_data_path) as f:
        file_lines_old = f.readlines()
    # random.shuffle(file_lines)
    file_lines = []
    if skip_data_path is not None:
        with open(skip_data_path) as f:
            skip_data_lines = f.readlines()
            for line in file_lines_old:
                impath = line.split(' ')[0]
                if not impath in skip_data_lines:
                    file_lines.append(line)
    else:
        file_lines = file_lines_old



    split_id = int(len(file_lines) * split)
    train_image_paths = file_lines[:split_id]
    test_image_paths = file_lines[split_id:]


    
    train_dataset = MyDataset(train_image_paths, label_root, class_num, box_score_thresh)
    test_dataset = MyDataset(test_image_paths, label_root, class_num, box_score_thresh)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    return train_dataloader, test_dataloader 


class Oracle(pl.LightningModule):
    def __init__(self, experiment_name, lr = 1, class_num = 91, train_epochs = 1):
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
        self.trainer = Trainer(accelerator='ddp', gpus=-1, max_epochs=train_epochs,
                               logger=self.aim_logger,
                            #    callbacks = [checkpoint_callback],
                            #    gradient_clip_val=0.5,
                               check_val_every_n_epoch=1)
                            #    val_check_interval = 0.005)
        self.loss_fn = nn.CrossEntropyLoss()
        # self.targets_file = os.getcwd() + "/ious_nightowls.npy"
        # self.ious = {i: [] for i in [1, 2, 3]}

    def global_info_init_val(self):
        self.global_info_val = {i: {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "tn": 0
        } for i in range(1, self.class_num + 1)}
    
    def global_info_init_train(self):
        self.global_info_train = {i: {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "tn": 0
        } for i in range(1, self.class_num + 1)}

    def global_info_init(self):
        self.global_info_init_val()
        self.global_info_init_train()
        self.global_info_test = {i: {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "tn": 0
        } for i in range(1, self.class_num + 1)}

    def set_datasets(self, feature_data_path, label_root, split, batch_size,
                     class_num, box_score_thresh, skip_data_path=None):
        self.train_loader, self.test_loader = get_train_test_loaders(feature_data_path,
                                                                     label_root, split, batch_size,
                                                                     class_num,
                                                                     box_score_thresh,
                                                                     skip_data_path)

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
        optimizer = optim.SGD(self.parameters(), lr=self.lr, momentum=0.9, weight_decay=0.01)
        # optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-8, max_lr=self.lr, step_size_up=2000)
        return {'optimizer': optimizer,'lr_scheduler': self.scheduler}

    def optimizer_step(self, epoch: int = None, batch_idx: int = None, optimizer = None,
                       optimizer_idx: int = None, optimizer_closure = None, on_tpu: bool = None,
                       using_native_amp: bool = None, using_lbfgs: bool = None):

        curLR = self.scheduler.get_last_lr()[0]

        # self.logger.experiment.track(curLR, name='lr', context={'model':self.onTeacher, 'stage':self.stage})

        for pg in optimizer.param_groups:
            pg['lr'] = curLR
        optimizer.step(closure=optimizer_closure)
        self.scheduler.step()

    def loss_function(self, y_pred, y_truth):
        return self.loss_fn(y_pred, y_truth)


    def training_step(self, batch, batch_inx):
        img_paths, inputs, labels, bboxes, ious, confs, classes = batch
        for i in range(len(inputs)):
            inputs[i] = torch.squeeze(inputs[i], 0)
        outputs = self.forward(inputs)
        loss = self.loss_function(outputs, labels)
        for i in range(len(outputs)):
            label = labels[i].item()
            output = torch.argmax(outputs[i]).item()
            cl = classes[i].item()
            # cl = 1
            # self.ious[cl].append(label.cpu().detach().numpy())
            num_cls = 2
            if label >= num_cls / 2 and output >= num_cls / 2:
                self.global_info_train[cl]['tp'] += 1
            elif label < num_cls / 2 and output >= num_cls / 2:
                self.global_info_train[cl]['fp'] += 1
            elif label >= num_cls / 2 and output < num_cls / 2:
                self.global_info_train[cl]['fn'] += 1
            else:
                self.global_info_train[cl]['tn'] += 1

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
        img_paths, inputs, labels, bboxes, ious, confs, classes = batch
        for i in range(len(inputs)):
            inputs[i] = torch.squeeze(inputs[i], 0)
        outputs = self.forward(inputs)

        # lines = []
        # for i in range(len(inputs)):
        #     boxstr = str(bboxes[0][i].item()) + ',' + str(bboxes[1][i].item()) + ',' + str(bboxes[2][i].item()) + ',' + str(bboxes[3][i].item())
        #     lines.append(img_paths[i] + ' ' + boxstr + ' ' + str(classes[i].item()) + ' ' + str(ious[i].item()) + ' ' + str(labels[i].item()) + ' ' + str(confs[i].item()) + "".join([' ' + str(_.item()) for _ in outputs[i]]) + '\n')

        # with open("/home/hkhachatrian/SSL-playground/consistency_learning_utils/nets/oracle/feature_data/C2.txt", "a+") as f:
        #     f.writelines(lines)

        loss = self.loss_function(outputs, labels)
        for i in range(len(outputs)):
            label = labels[i].item()
            output = torch.argmax(outputs[i]).item()
            # cl = 1
            cl = classes[i].item()

            # print(label, output, cl)
            num_cls = 2
            if label >= num_cls / 2 and output >= num_cls / 2:
                self.global_info_val[cl]['tp'] += 1
            elif label < num_cls / 2 and output >= num_cls / 2:
                self.global_info_val[cl]['fp'] += 1
            elif label >= num_cls / 2 and output < num_cls / 2:
                self.global_info_val[cl]['fn'] += 1
            else:
                self.global_info_val[cl]['tn'] += 1

        self.logger.experiment.track(loss.item(), name='val_loss')

        return {'val_loss': loss}

    def validation_epoch_end(self, results):
        fscores = self.f1_scores(self.global_info_val)
        for k in self.global_info_val.keys():
            dct = self.global_info_val[k]
            tp = dct['tp']
            fp = dct['fp']
            fn = dct['fn']
            tn = dct['tn']
            if tp + fp == 0:
                accuracy_on_positives = 0
            else:
                accuracy_on_positives = tp / (tp + fp)
            if tn + fn == 0:
                accuracy_on_negatives = 0
            else:
                accuracy_on_negatives = tn / (tn + fn)
            
            accs = str(accuracy_on_positives) + ', ' + str(accuracy_on_negatives)
            print("class " + str(k) + ": acc on positives and negatives: " + accs)
        self.global_info_init_val()
        # self.logger.experiment.track(fscores[0], name='fscore_class_0')
        # self.logger.experiment.track(fscsores[1], name='fscore_class_1')
        # self.logger.experiment.track(fscores[2], name='fscore_class_2')
        accuracy = sum(fscores) / len(fscores)
        if accuracy >= self.best_validation_accuracy:
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


def inference(pred, model_path, iou_thresh=0.5, experiment_name=""):
    model = Oracle(experiment_name)
    model.cuda()
    model.load_from_path(model_path)
    features, boxes, scores, labels = pred
    kept_boxes = []
    kept_scores = []
    kept_labels = []
    for feature, box, score, label in zip(features, boxes, scores, labels):
        output = model(feature)
        keep_indices = []
        for j in range(output.shape[0]):
            if torch.argmax(output[j]).item() != 0:
                keep_indices.append(j)
        kept_boxes.append(box[keep_indices])
        kept_scores.append(score[keep_indices])
        kept_labels.append(label[keep_indices])
    return (kept_boxes, kept_scores, kept_labels)

