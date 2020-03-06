import json
from argparse import Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import cross_entropy, mse_loss, kl_div
import torch.optim as optim

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from dataloader import get_train_test_loaders
import cifar.datasets

class UDA(pl.LightningModule):
    def __init__(self, hparams:Namespace, train_dataset, test_dataset, number_of_classes) -> None:
        super().__init__()

        self.hparams = hparams
        hparams = hparams.__dict__

        self.lr = hparams['learning_rate']
        self.eta_min = self.lr * hparams['min_lr_ratio']
        self.num_warmup_steps = hparams['num_warmup_steps']
        self.with_SWA = hparams['with_SWA']
        self.lam = hparams['consistency_lambda']
        self.num_epochs = hparams['num_epochs']
        self.momentum = hparams['momentum']
        self.weight_decay = hparams['weight_decay']
        self.consistency_criterion = hparams['consistency_criterion']
        self.train_loader, self.valid_loader = get_train_test_loaders(train_dataset, test_dataset,
                                                                      number_of_classes,
                                                                      hparams['num_labelled_samples'],
                                                                      hparams['batch_size'],
                                                                      hparams['num_workers'],
                                                                      hparams['unlabelled_batch_size'])

        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(64 * 4 * 4, 500)

        self.fc2 = nn.Linear(500, 10)

        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(F.elu(self.conv1(x)))
        x = self.pool(F.elu(self.conv2(x)))
        x = self.pool(F.elu(self.conv3(x)))

        x = x.view(-1, 64 * 4 * 4)
        x = self.dropout(x)
        x = F.elu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    @pl.data_loader
    def train_dataloader(self):
        return self.train_loader

    def training_step(self, batch, batch_inx):
        (x, y), (unlabeled, augmented) = batch
        y_hat = self.forward(x)

        unlab_pred = self.forward(unlabeled)

        augment_pred = self.forward(augmented)
        loss = self.final_loss(y_hat, y, augment_pred, unlab_pred)
        log_dict = {
            'training_loss': loss,
        }

        return {'loss': loss, 'log': log_dict}

    @pl.data_loader
    def val_dataloader(self):
        return self.valid_loader

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        return {'valid_loss': F.cross_entropy(y_hat, y),
                'acc': self.compute_accuracy(y, y_hat)}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['valid_loss'] for x in outputs]).mean()
        tensorboard_logs = {'valid_loss': avg_loss}
        return {'avg_valid_loss': avg_loss, 'log': tensorboard_logs}

    def compute_accuracy(self, y, y_hat):
        y_hat = y_hat.argmax(dim=-1)
        return torch.sum(y == y_hat).item() / len(y)

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum,
                              weight_decay=self.weight_decay, nesterov=True)
        optimizer_sched = optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=self.eta_min,
                                                              T_max=(len(self.train_loader)*self.num_epochs - self.num_warmup_steps))

        return [optimizer], [optimizer_sched]

    def final_loss(self, y_hat, y_true, unsup_aug_y_probas, unsup_orig_y_probas):
        return self.classification_loss(y_hat, y_true) + \
               self.lam * self.consistency_loss(unsup_aug_y_probas, unsup_orig_y_probas)

    def classification_loss(self, y_pred, y_true):
        return cross_entropy(y_pred, y_true)

    def consistency_loss(self, unsup_aug_y_probas, unsup_orig_y_probas):
        unsup_aug_y_probas = torch.log_softmax(unsup_aug_y_probas, dim=-1)
        unsup_orig_y_probas = torch.softmax(unsup_orig_y_probas, dim=-1)
        if self.consistency_criterion == "MSE":
            return mse_loss(unsup_aug_y_probas), torch.softmax(unsup_orig_y_probas)
        elif self.consistency_criterion == "KL":
            return kl_div(unsup_aug_y_probas, unsup_orig_y_probas)

    def fit_model(self):
        checkpoint_callback = ModelCheckpoint(
            filepath="checkpoints/{}/version_{}".format(hparams['experiment_name'], hparams['version_name']),
            verbose=True,
        )
        trainer = Trainer(gpus=None, early_stop_callback=None, show_progress_bar=True,
                          checkpoint_callback=checkpoint_callback, check_val_every_n_epoch=1,
                          default_save_path="checkpoints", max_epochs=hparams['num_epochs'])

        trainer.fit(self)

    def load(self):
        self.load_from_checkpoint("checkpoints/{}/version_{}".format(hparams['experiment_name'], hparams['version_name']))
        self.eval()


if __name__ == "__main__":

    with open('cifar10_simple_hparams.json') as f:
        hparams= json.load(f)

    train_ds, valid_ds, num_classes = cifar.datasets.get_train_test_datasets(hparams['dataset'], hparams['data_path'])
    model = UDA(Namespace(**hparams), train_ds, valid_ds, num_classes)

    model.fit_model()
    #model.load()



