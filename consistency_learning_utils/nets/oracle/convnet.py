from typing import List, Any

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.functional import binary_cross_entropy

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from aim.pytorch_lightning import AimLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import os


class Net(nn.Module):
    def __init__(self, feature_size=6):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv5 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv6 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv7 = nn.Conv2d(32, 1, 3, padding=1)
        self.conv8 = nn.Conv2d(1, 1, (feature_size, 1))

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = self.conv8(x)
        x = torch.sigmoid(x)
        return x


class MyDataset(Dataset):
    def __init__(self, samples, labels):
        super().__init__()
        self.samples = samples
        self.labels = labels

    def __len__(self):
        return len(self.samples)

    def __get_target__(self, sample_index):
        return self.labels[sample_index]

    def __getitem__(self, index):
        sample = self.samples[index]
        target = self.__get_target__(index)

        return sample, target


def get_train_test_loaders(samples, labels):
    samples = torch.transpose(torch.FloatTensor(samples), 1, 2)
    samples = samples.reshape(-1, 1, 6, 1988)

    labels = torch.tensor(labels)
    labels = torch.reshape(labels, (7942, 1, 1988))

    train_samples, test_samples = torch.utils.data.random_split(samples, [5500, 2442],
                                                                generator=torch.Generator().manual_seed(42))
    train_labels, test_labels = torch.utils.data.random_split(labels, [5500, 2442],
                                                              generator=torch.Generator().manual_seed(42))
    train_dataset = MyDataset(train_samples, train_labels)
    test_dataset = MyDataset(test_samples, test_labels)

    train_dataloader = DataLoader(train_dataset, batch_size=16)
    test_dataloader = DataLoader(test_dataset, batch_size=16)

    return train_dataloader, test_dataloader


class Oracle(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.model = Net()
        self.aim_logger = AimLogger(
            experiment='oracle_all_save_model'
        )
        self.save_dir_name = os.getcwd() + "/checkpoints/oracle"
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.save_dir_name,
            save_last=True
        )
        self.trainer = Trainer(gpus=-1, logger=self.aim_logger, max_epochs=200, checkpoint_callback=checkpoint_callback)

        self.global_info = {i: {
            "tp": 0,
            "fp": 0,
            "fn": 0
        } for i in [1, 2, 3]}

        self.global_info_test = {i: {
            "tp": 0,
            "fp": 0,
            "fn": 0
        } for i in [1, 2, 3]}

    def set_datasets(self, samples, labels, train_cl_masks, test_cl_masks):
        self.train_cl_masks = train_cl_masks
        self.test_cl_masks = test_cl_masks

        self.train_loader, self.test_loader = get_train_test_loaders(samples, labels)

    def load_from_path(self, path):
        self.model.load_state_dict(torch.load(path))

    def forward(self, x):
        return self.model(x)

    def train_dataloader(self):
        return self.train_loader

    def test_dataloader(self):
        return self.test_loader

    def configure_optimizers(self):
        # optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
        optimizer = optim.Adam(self.parameters(), lr=0.001, momentum=0.9)

        return {'optimizer': optimizer}

    def bce_loss(self, y_pred, y_truth):
        return binary_cross_entropy(y_pred, y_truth)

    def training_step(self, batch, batch_inx):
        inputs, labels = batch
        outputs = self.forward(inputs).reshape(-1, 1, 1988)
        loss = self.bce_loss(outputs, labels.float())
        if self.current_epoch == 199:
            for i in range(len(outputs)):
                self.compute_accuracy(outputs[i][0].cpu().detach().numpy()>0.5, labels[i][0].cpu().detach().numpy(),
                                      self.train_cl_masks[i])

        self.logger.experiment.track(loss.item(), name='training_loss')
        return {'loss': loss}

    def training_epoch_end(self, outputs: List[Any]) -> None:
        if self.current_epoch == 199:
            print(self.global_info)

    def test_step(self, batch, batch_inx):
        inputs, labels = batch
        outputs = self.forward(inputs).reshape(-1, 1, 1988)

        for i in range(len(outputs)):
            self.compute_accuracy(outputs[i][0].cpu().detach().numpy() > 0.5, labels[i][0].cpu().detach().numpy(),
                                  self.test_cl_masks[i], True)

        loss = self.bce_loss(outputs, labels.float())

        self.logger.experiment.track(loss.item(), name='test_loss')

        return {'loss': loss}


    def fit_model(self):
        self.trainer.fit(self)

    def test(self):
        return self.trainer.test(self)

    def on_test_end(self):
        print(self.global_info_test)

    def compute_accuracy(self, y_hat, y, cl_masks, test=False):
        classes = list(cl_masks.keys())
        if test:
            for cl in classes:
                self.global_info_test[cl]['tp'] += sum(y_hat[:len(cl_masks[cl])] & y[:len(cl_masks[cl])]
                                                  & cl_masks[cl])
                self.global_info_test[cl]['fp'] += sum(y_hat[:len(cl_masks[cl])] & ~y[:len(cl_masks[cl])]
                                                  & cl_masks[cl])
                self.global_info_test[cl]['fn'] += sum(~y_hat[:len(cl_masks[cl])] & y[:len(cl_masks[cl])]
                                                  & cl_masks[cl])
        else:
            for cl in classes:
                self.global_info[cl]['tp'] += sum(y_hat[:len(cl_masks[cl])] & y[:len(cl_masks[cl])]
                                                  & cl_masks[cl])
                self.global_info[cl]['fp'] += sum(y_hat[:len(cl_masks[cl])] & ~y[:len(cl_masks[cl])]
                                                  & cl_masks[cl])
                self.global_info[cl]['fn'] += sum(~y_hat[:len(cl_masks[cl])] & y[:len(cl_masks[cl])]
                                                  & cl_masks[cl])

    def f1_score(self, tp, fp, fn):
        return tp / (tp + (fp + fn) / 2)


def preprocess_predictions(pred):
    boxes_per_image = [len(p) for p in pred]
    pred = torch.FloatTensor(pred)
    return boxes_per_image, pred

def select_pls(output, pred, boxes_per_image, iou_thresh): # TODO make faster
    output = output.split(boxes_per_image)
    pred = pred.split(boxes_per_image)
    new_pred = []
    for i, p in enumerate(pred):
        new_p = []
        for j, out in enumerate(output[i]):
            if out >= iou_thresh:
                new_p.append(p[j])
        new_pred.append(new_p)
    return output

def inference(pred, model_path, iou_thresh=0.5):
    model = Oracle()
    model.load_from_path(model_path)
    boxes_per_image, pred = preprocess_predictions(pred)
    output = model()
    selected_predictons = select_pls(output, pred, boxes_per_image, iou_thresh)
    return selected_predictons


# if __name__ == "__main__":
#     csv_path = '/home/hkhachatrian/SSL-playground/session_data/5L53Vy04iImX6naGoqdy/base/widerperson_base_3_from_coco_0_first_oracle_random_dl/predictions_on_unlabeled.csv'
#     label_root = '/home/hkhachatrian/SSL-playground/session_data/5L53Vy04iImX6naGoqdy/base/labels/'

#     samples, selected_pseudo_labels, train_cl_masks, test_cl_masks = get_dataset(csv_path, label_root, 5500)

#     model = Oracle()
#     model.set_datasets(samples, selected_pseudo_labels, train_cl_masks, test_cl_masks)
#     model.fit_model()
#     model.test()
