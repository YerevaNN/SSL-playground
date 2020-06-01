import json
from argparse import Namespace
import os
import numpy as np

import torch
from torch.nn.functional import cross_entropy, mse_loss, kl_div
import torch.optim as optim

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.logging import TestTubeLogger

from dataloader import get_train_test_loaders
from helpers.tsa import TrainingSignalAnnealing
from nets.fastresnet import FastResnet
import cifar.datasets
from uda_trainer import UdaTrainer

class UDA(pl.LightningModule):
    def __init__(self, hparams:Namespace) -> None:
        super().__init__()

        self.hparams = hparams
        hparams = hparams.__dict__

        self.lr = hparams['learning_rate']
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


        self.net = FastResnet(bn_kwargs={"bn_weight_init": 1.0})
    
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
        y_hat = self.forward(x)
        # tsa_y_hat, tsa_y = self.tsa(y_hat, y)
        sup_loss = self.classification_loss(y_hat, y)

        unlabeled, augmented = unsup_batch

        unlab_x, unlab_y = unlabeled
        aug_x, aug_vec = augmented

        augment_pred = self.forward(aug_x)
        with torch.no_grad():
            unlab_pred = self.forward(unlab_x)

        # if (self.current_epoch<50):
        #     self.lam = self.max_lam*(self.current_epoch/50)
        # else:
        #     self.lam = self.max_lam

        unsup_loss = self.lam * self. self.consistency_loss(augment_pred, unlab_pred)

        self.loss = sup_loss + unsup_loss

        log_dict = {
            'train_sup_acc': self.compute_accuracy(y, y_hat),
            'train_unsup_acc': self.compute_accuracy(torch.argmax(unlab_pred, dim=-1), augment_pred),
            'origin_label_logit_acc': self.compute_accuracy(unlab_y, unlab_pred),
            'training_sup_loss': sup_loss,
            'training_unsup_loss': unsup_loss,
            'training_loss': self.loss,
            'lambda': self.lam
        }

        return {'loss': self.loss, 'log': log_dict}

    @pl.data_loader
    def val_dataloader(self):
        return self.valid_loader

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        return {'val_loss': cross_entropy(y_hat, y),
                'val_acc': self.compute_accuracy(y, y_hat)}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        val_acc = np.array([x['val_acc'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss, 'val_acc': val_acc}
        return {'avg_valid_loss': avg_loss, 'log': tensorboard_logs}

    def compute_accuracy(self, y, y_hat):
        y_hat = y_hat.argmax(dim=-1)
        return torch.sum(y == y_hat).item() / len(y)

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum,
                              weight_decay=self.weight_decay, nesterov=True)
        optimizer_sched = optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=self.eta_min,
                                                              T_max=(len(self.train_labeled_loader)*self.num_epochs - self.num_warmup_steps))

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
            return mse_loss(unsup_aug_y_probas, unsup_orig_y_probas)
        elif self.consistency_criterion == "KL":
            return kl_div(unsup_aug_y_probas, unsup_orig_y_probas, reduction='batchmean')


    def fit_model(self):
        hparams = self.hparams.__dict__
        checkpoint_callback = ModelCheckpoint(
            filepath=self.save_dir_name,
            verbose=True,
            save_top_k=-1,
            period=20
        )

        tt_logger = TestTubeLogger(
            save_dir="logs",
            name="{}_{}".format(hparams['experiment_name'], hparams['model']),
            version=hparams['version_name'],
            debug=False,
            create_git_tag=False,
        )

        trainer = UdaTrainer(gpus=-1, early_stop_callback=None, logger=tt_logger, show_progress_bar=True,
                          checkpoint_callback=checkpoint_callback, check_val_every_n_epoch=1, default_save_path="./checkpoints",
                          val_check_interval=30, max_epochs=hparams['num_epochs'], log_save_interval=1, row_log_interval=1)

        trainer.fit(self)

    def load(self):
        self.load_from_checkpoint(self.save_dir_name)
        self.eval()
        #self.freeze()


if __name__ == "__main__":

    with open('cifar10_simple_hparams.json') as f:
        hparams= json.load(f)

    train_ds, valid_ds, num_classes = cifar.datasets.get_train_test_datasets(hparams['dataset'], hparams['data_path'])
    model = UDA(Namespace(**hparams))
    model.set_datasets(train_ds, valid_ds, num_classes)

    model.fit_model()
    # model.load()
    # # out = model(test_dataset=valid_ds)
    # # print(out)





