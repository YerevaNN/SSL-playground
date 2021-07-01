from argparse import Namespace
import os
import numpy as np

import torch
from torch import nn
from torch.nn.functional import cross_entropy, mse_loss, kl_div
import torch.optim as optim

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TestTubeLogger


from SSL_playground.dataloader import get_train_test_loaders
from SSL_playground.nets.fastresnet import FastResnet
from pytorch_lightning import Trainer


class MPL(pl.LightningModule):
    def __init__(self, hparams:Namespace) -> None:
        super().__init__()

        hparams = hparams.__dict__
        self.hparams = hparams

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
        self.threshold = hparams['MPL_threshold']
        
        self.save_dir_name = os.getcwd() + "/checkpoints/{}_{}/version_{}".format(hparams['experiment_name'], hparams['model'],\
                             hparams['version_name'])

        self.teacher = FastResnet(bn_kwargs={"bn_weight_init": 1.0})
        self.student = FastResnet(bn_kwargs={"bn_weight_init": 1.0})

        self.moving_dot_product = torch.empty(1, device='cuda')
        limit = 3.0 ** (0.5)  # 3 = 6 / (f_in + f_out)
        nn.init.uniform_(self.moving_dot_product, -limit, limit)
    
    def set_datasets(self, train_dataset, val_dataset, test_dataset, number_of_classes):
        hparams = self.hparams
        self.train_labeled_loader, self.train_unlabeled_loader, self.valid_loader, self.test_loader = get_train_test_loaders(
                                                                      train_dataset,
                                                                      val_dataset,
                                                                      test_dataset,
                                                                      number_of_classes,
                                                                      hparams['num_labelled_samples'],
                                                                      hparams['batch_size'],
                                                                      hparams['num_workers'],
                                                                      hparams['unlabelled_batch_size'])
        self.student_optimizer = optim.SGD(self.student.parameters(), lr=self.lr, momentum=self.momentum,
                                           weight_decay=self.weight_decay, nesterov=True)
        self.student_optimizer_sched = optim.lr_scheduler.CosineAnnealingLR(self.student_optimizer,
                                                                            eta_min=self.eta_min,
                                                                            T_max=(len(
                                                                                self.train_labeled_loader) * self.num_epochs - self.num_warmup_steps))

    def forward(self, x, model):
        return getattr(self, model).forward(x)

    # @pl.data_loader
    def train_dataloader(self):
        return [self.train_labeled_loader, self.train_unlabeled_loader]

    def training_step(self, batch_list, batch_inx):
        sup_batch, unsup_batch = batch_list

        x, y = sup_batch
        unlabeled, augmented = unsup_batch

        # #teacher's all calls
        unlab_x, unlab_y = unlabeled
        # aug_x, aug_c = augmented
        aug_x = augmented

        y_hat = self.forward(x, 'teacher')
        t_loss_l = self.classification_loss(y_hat, y)

        augment_pred = self.forward(aug_x, 'teacher')
        with torch.no_grad():
            unlab_pred = self.forward(unlab_x, 'teacher')

        t_uda_loss = self.lam * self.consistency_loss(augment_pred, unlab_pred)

        soft_pseudo_labels = torch.softmax(augment_pred, dim=-1)

        # # max_prob, hard_pseudo_labels = torch.max(soft_pseudo_labels, dim=-1)
        # # mask = max_prob.ge(self.threshold)
        hard_pseudo_labels = torch.argmax(soft_pseudo_labels, dim=-1)
        #
        t_loss_u = self.classification_loss(augment_pred, hard_pseudo_labels)
        #
        # #student's first call
        with torch.no_grad():
            lab_students_logits = self.forward(x, 'student')
        s_loss_sup_old = self.classification_loss(lab_students_logits, y)

        unlab_students_logits = self.forward(aug_x, 'student')
        s_loss_unsup = self.classification_loss(unlab_students_logits, hard_pseudo_labels)

        s_loss_unsup.backward()

        self.student_optimizer.step()
        self.student_optimizer_sched.step()
        #
        #
        # #student's second call
        with torch.no_grad():
            l_students_logits = self.forward(x, 'student')
        s_loss_sup_new = self.classification_loss(l_students_logits, y)
        #
        with torch.no_grad():
            # dot_product = torch.transpose(s_loss_sup_new, -1, 0) * s_loss_unsup
            dot_product = s_loss_sup_new - s_loss_sup_old
            # self.moving_dot_product = self.moving_dot_product * 0.99 + dot_product * 0.01
            # dot_product = dot_product - self.moving_dot_product

        t_mpl_loss = dot_product*t_loss_u

        # t_total_loss = t_loss_l + t_mpl_loss
        t_total_loss = t_loss_l + t_uda_loss + t_mpl_loss
        # t_total_loss = t_loss_l + t_uda_loss
        # if (self.current_epoch<50):
        #     self.lam = self.max_lam*(self.current_epoch/50)
        # else:
        #     self.lam = self.max_lam

        self.loss = t_total_loss

        log_dict = {
            'teacher_total_loss': self.loss,
            'student_unsup_loss': s_loss_unsup,
            'student_sup_loss_old': s_loss_sup_old,
            'student_sup_loss_new': s_loss_sup_new,
            'teacher_uda_loss': t_uda_loss,
            'teacher_mpl_loss': t_mpl_loss,
            'teacher_sup_loss': t_loss_l,
            'teacher_unsup_loss': t_loss_u,
            'dot_product': dot_product,
            'teacher_train_sup_acc': self.compute_accuracy(y, y_hat)
            # 'train_unsup_acc': self.compute_accuracy(torch.argmax(unlab_pred, dim=-1), augment_pred),
            # 'origin_label_logit_acc': self.compute_accuracy(unlab_y, unlab_pred),
            # 'training_sup_loss': sup_loss,
            # 'training_unsup_loss': unsup_loss,
            # 'training_loss': self.loss,
            # 'lambda': self.lam
        }
        return {'loss': self.loss, 'log': log_dict}

    # @pl.data_loader
    def val_dataloader(self):
        return self.valid_loader

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x, 'student')
        self.log('val_loss', cross_entropy(y_hat, y))
        self.log('val_acc', self.compute_accuracy(y, y_hat))
        return {'val_loss': cross_entropy(y_hat, y),
                'val_acc': self.compute_accuracy(y, y_hat)
                }

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        val_acc = np.array([x['val_acc'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss, 'val_acc': val_acc}
        return {'avg_valid_loss': avg_loss, 'log': tensorboard_logs}

    def test_dataloader(self):
        return self.test_loader

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x, 'student')
        self.log('test_loss', cross_entropy(y_hat, y))
        self.log('test_acc', self.compute_accuracy(y, y_hat))
        return {'test_loss': cross_entropy(y_hat, y),
                'test_acc': self.compute_accuracy(y, y_hat)
                }

    def test__epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        val_acc = np.array([x['test_acc'] for x in outputs]).mean()
        tensorboard_logs = {'test_loss': avg_loss, 'test_acc': val_acc}
        return {'avg_test_loss': avg_loss, 'log': tensorboard_logs}

    def compute_accuracy(self, y, y_hat):
        y_hat = y_hat.argmax(dim=-1)
        return torch.sum(y == y_hat).item() / len(y)

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum,
                              weight_decay=self.weight_decay, nesterov=True)
        optimizer_sched = optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=self.eta_min, T_max=(len(self.train_labeled_loader)*self.num_epochs - self.num_warmup_steps))

        return [optimizer], [optimizer_sched]

    def final_loss(self, y_hat, y_true, unsup_aug_y_probas, unsup_orig_y_probas):
        return self.classification_loss(y_hat, y_true) + \
               self.lam * self.consistency_loss(unsup_aug_y_probas, unsup_orig_y_probas)

    def classification_loss(self, y_pred, y_true):
        return cross_entropy(y_pred, y_true)

    def consistency_loss(self, unsup_aug_y_probas, unsup_orig_y_probas):
        if self.consistency_criterion == "MSE":
            return mse_loss(torch.softmax(unsup_aug_y_probas, dim=-1), torch.softmax(unsup_orig_y_probas, dim=-1))
        elif self.consistency_criterion == "KL":
            unsup_aug_y_probas = torch.log_softmax(unsup_aug_y_probas, dim=-1)
            unsup_orig_y_probas = torch.softmax(unsup_orig_y_probas, dim=-1)
            return kl_div(unsup_aug_y_probas, unsup_orig_y_probas, reduction='batchmean')


    def fit_model(self):
        hparams = self.hparams
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.save_dir_name,
            verbose=True,
            save_top_k=-1,
            save_last=True,
            period=20
        )

        tt_logger = TestTubeLogger(
            save_dir="logs",
            name="{}_{}".format(hparams['experiment_name'], hparams['model']),
            version=hparams['version_name'],
            debug=False,
            create_git_tag=False,
        )

        trainer = Trainer(gpus=[1], logger=tt_logger, checkpoint_callback=checkpoint_callback,
                          default_root_dir="../checkpoints", max_epochs=hparams['num_epochs'],
                          multiple_trainloader_mode='max_size_cycle')

        trainer.fit(self)

    def load(self):
        self.load_from_checkpoint(self.save_dir_name)
        self.eval()
        #self.freeze()


# if __name__ == "__main__":
#
#     with open('../cifar10_simple_hparams.json') as f:
#         hparams= json.load(f)
#
#     train_ds, valid_ds, num_classes = get_train_test_datasets(hparams['dataset'], hparams['data_path'])
#     model = UDA(Namespace(**hparams))
#     model.set_datasets(train_ds, valid_ds, num_classes)
#
#     model.fit_model()
#     # model.load()
#     # # out = model(test_dataset=valid_ds)
#     # # print(out)




