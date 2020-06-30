from argparse import Namespace

import torch
import torch.optim as optim
from torch.nn.functional import mse_loss, nll_loss, log_softmax

from SSL_playground import hsic
from SSL_playground.lightning_model import UDA
from SSL_playground.nets.fastresnet import FastResnet


class MeanTeacher(UDA):
    def __init__(self, hparams: Namespace) -> None:
        super().__init__(hparams)

        self.student = FastResnet(bn_kwargs={"bn_weight_init": 1.0})
    def configure_optimizers(self):
        self.optimizer = optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum)
        return self.optimizer


    def training_step(self, batch_list, batch_inx):

        sup_batch, unsup_batch = batch_list

        x, y = sup_batch
        y_hat = self.student.forward(x)

        # sup_loss = self.classification_loss(y_hat, y)
        sup_loss = nll_loss(log_softmax(y_hat, dim=-1), y)

        unlabeled, augmented = unsup_batch
        unlab_x, unlab_y = unlabeled
        aug_x, aug_vec = augmented

        unlab_pred_student = self.student.forward(aug_x)

        # sup_loss = nll_loss(log_softmax(unlab_pred_student, dim=1), unlab_y)

        with torch.no_grad():
            unlab_pred_teacher = self.net.forward(aug_x)

        z = unlab_pred_teacher
        c = aug_vec.type(dtype=torch.cuda.FloatTensor)

        hsic_unsup = hsic.HSIC(z, c)

        # z = torch.cat((unlab_pred, augment_pred))
        # c = torch.cat((torch.zeros_like(aug_vec), aug_vec)).type(dtype=torch.cuda.FloatTensor)
        #
        # hsic_unsup = hsic.HSIC(z, c)

        # augment_pred = self.net.forward2(augment_pred)
        # with torch.no_grad():
        #     unlab_pred = self.net.forward2(unlab_pred)

        # if (self.current_epoch < 50):
        #     self.lam = self.max_lam*self.current_epoch/50
        # else:
        #     self.lam = self.max_lam

        unsup_loss = mse_loss(torch.softmax(unlab_pred_student, dim=-1), torch.softmax(unlab_pred_teacher, dim=-1))
        self.loss = sup_loss + self.lam * unsup_loss

        alpha = 0.95
        for mean_param, param in zip(self.net.parameters(), self.student.parameters()):
            mean_param.data.mul_(alpha).add_(1 - alpha, param.data)

        log_dict = {
            'train_sup_acc': self.compute_accuracy(unlab_y, unlab_pred_student),
            'train_unsup_acc': self.compute_accuracy(torch.argmax(unlab_pred_teacher, dim=-1), unlab_pred_student),
            'origin_label_logit_acc': self.compute_accuracy(unlab_y, unlab_pred_teacher),
            'training_sup_loss': sup_loss,
            'hsic': hsic_unsup,
            'unsup_loss': unsup_loss,
            'lambda': self.lam,
            'training_loss': self.loss,
            # 'lr': torch.as_tensor(self.optimizer_sched.state_dict()['_last_lr'])
        }
        return {'loss': self.loss, 'log': log_dict}


