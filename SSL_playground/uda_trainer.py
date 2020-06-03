import copy
import warnings
import numpy as np

import torch

from pytorch_lightning import Trainer
from pytorch_lightning.trainer import data_loading, training_loop
from torch.utils.data import DistributedSampler

class LoadData(data_loading.TrainerDataLoadingMixin):
    def init_train_dataloader(self, model):
        self.get_train_dataloaders = model.train_dataloader

        self.num_training_batches = 0

        # determine number of validation batches
        # val datasets could be none, 1 or 2+
        if self.get_train_dataloaders() is not None:
            self._percent_range_check('train_percent_check')

            self.num_training_batches = sum(len(dataloader) for dataloader in self.get_train_dataloaders())
            self.num_training_batches = int(self.num_training_batches * self.train_percent_check)

        if isinstance(self.val_check_interval, int):
            self.val_check_batch = self.val_check_interval
            if self.val_check_batch > self.num_training_batches:
                raise ValueError(
                    f"`val_check_interval` ({self._check_interval}) must be less than or equal "
                    f"to the number of the training batches ({self.num_training_batches}). "
                    f"If you want to disable validation set `val_percent_check` to 0.0 instead.")
        else:
            self._percent_range_check('val_check_interval')

            self.val_check_batch = int(self.num_training_batches * self.val_check_interval)
            self.val_check_batch = max(1, self.val_check_batch)

        on_ddp = self.use_ddp or self.use_ddp2
        if on_ddp and self.get_train_dataloaders() is not None:
            for dataloader in self.get_val_dataloaders():
                if not isinstance(dataloader.sampler, DistributedSampler):
                    msg = """
                            Your val_dataloader(s) don't use DistributedSampler.

                            You're using multiple gpus and multiple nodes without using a
                            DistributedSampler to assign a subset of your data to each process.
                            To silence this warning, pass a DistributedSampler to your DataLoader.

                            ie: this:
                            dataset = myDataset()
                            dataloader = Dataloader(dataset)

                            becomes:
                            dataset = myDataset()
                            dist_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
                            dataloader = Dataloader(dataset, sampler=dist_sampler)

                            If you want each process to load the full dataset, ignore this warning.
                            """
                    if msg not in self.shown_warnings and self.proc_rank == 0:
                        self.shown_warnings.add(msg)
                        warnings.warn(msg)
                    break
class TrainingLoop(training_loop.TrainerTrainLoopMixin):
    def __init__(self):
        # this is just a summary on variables used in this abstract class,
        #  the proper values/initialisation should be done in child class
        self.max_epochs = None
        self.min_epochs = None
        self.use_ddp = None
        self.use_dp = None
        self.use_ddp2 = None
        self.single_gpu = None
        self.data_parallel_device_ids = None
        self.check_val_every_n_epoch = None
        self.num_training_batches = None
        self.val_check_batch = None
        self.num_val_batches = None
        self.disable_validation = None
        self.fast_dev_run = None
        self.is_iterable_train_dataloader = None
        self.main_progress_bar = None
        self.accumulation_scheduler = None
        self.lr_schedulers = None
        self.enable_early_stop = None
        self.early_stop_callback = None
        self.callback_metrics = None
        self.logger = None
        self.global_step = None
        self.testing = None
        self.log_save_interval = None
        self.proc_rank = None
        self.row_log_interval = None
        self.total_batches = None
        self.truncated_bptt_steps = None
        self.optimizers = None
        self.accumulate_grad_batches = None
        self.use_amp = None
        self.print_nan_grads = None
        self.track_grad_norm = None
        self.model = None
        self.running_loss = None
        self.training_tqdm_dict = None
        self.get_train_dataloaders = None
        self.reduce_lr_on_plateau_scheduler = None

    def run_training_batch(self, batch, batch_idx):
        # track grad norms
        grad_norm_dic = {}

        # track all metrics for callbacks
        all_callback_metrics = []

        # track metrics to log
        all_log_metrics = []

        if batch is None:
            return 0, grad_norm_dic, {}

        # hook
        if self.is_function_implemented('on_batch_start'):
            model_ref = self.get_model()
            response = model_ref.on_batch_start(batch)

            if response == -1:
                return -1, grad_norm_dic, {}

        splits = [batch]
        if self.truncated_bptt_steps is not None:
            model_ref = self.get_model()
            splits = model_ref.tbptt_split_batch(batch, self.truncated_bptt_steps)

        self.hiddens = None
        for split_idx, split_batch in enumerate(splits):
            self.split_idx = split_idx

            # call training_step once per optimizer
            for opt_idx, optimizer in enumerate(self.optimizers):
                # make sure only the gradients of the current optimizer's paramaters are calculated
                # in the training step to prevent dangling gradients in multiple-optimizer setup.
                if len(self.optimizers) > 1:
                    for param in self.get_model().parameters():
                        param.requires_grad = False
                    for group in optimizer.param_groups:
                        for param in group['params']:
                            param.requires_grad = True

                # wrap the forward step in a closure so second order methods work
                def optimizer_closure():
                    # forward pass
                    output = self.training_forward(
                        split_batch, batch_idx, opt_idx, self.hiddens)

                    closure_loss = output[0]
                    progress_bar_metrics = output[1]
                    log_metrics = output[2]
                    callback_metrics = output[3]
                    self.hiddens = output[4]

                    # accumulate loss
                    # (if accumulate_grad_batches = 1 no effect)
                    closure_loss = closure_loss / self.accumulate_grad_batches

                    # backward pass
                    model_ref = self.get_model()
                    model_ref.backward(self.use_amp, closure_loss, optimizer)

                    # track metrics for callbacks
                    all_callback_metrics.append(callback_metrics)

                    # track progress bar metrics
                    self.add_tqdm_metrics(progress_bar_metrics)
                    all_log_metrics.append(log_metrics)

                    # insert after step hook
                    if self.is_function_implemented('on_after_backward'):
                        model_ref = self.get_model()
                        model_ref.on_after_backward()

                    return closure_loss

                # calculate loss
                loss = optimizer_closure()

                # nan grads
                if self.print_nan_grads:
                    self.print_nan_gradients()

                # track total loss for logging (avoid mem leaks)
                self.batch_loss_value += loss.item()

                # gradient update with accumulated gradients
                if (self.batch_idx + 1) % self.accumulate_grad_batches == 0:

                    # track gradient norms when requested
                    if batch_idx % self.row_log_interval == 0:
                        if self.track_grad_norm > 0:
                            model = self.get_model()
                            grad_norm_dic = model.grad_norm(
                                self.track_grad_norm)

                    # clip gradients
                    self.clip_gradients()

                    # calls .step(), .zero_grad()
                    # override function to modify this behavior
                    model = self.get_model()
                    model.optimizer_step(self.current_epoch, batch_idx,
                                         optimizer, opt_idx, optimizer_closure)

                    # calculate running loss for display
                    self.running_loss.append(self.batch_loss_value)
                    self.batch_loss_value = 0
                    self.avg_loss = np.mean(self.running_loss[-100:])

        # activate batch end hook
        if self.is_function_implemented('on_batch_end'):
            model = self.get_model()
            model.on_batch_end()

        # update progress bar
        self.main_progress_bar.update(1)
        self.main_progress_bar.set_postfix(**self.training_tqdm_dict)

        # collapse all metrics into one dict
        all_log_metrics = {k: v for d in all_log_metrics for k, v in d.items()}

        # track all metrics for callbacks
        self.callback_metrics.update({k: v for d in all_callback_metrics for k, v in d.items()})

        return 0, grad_norm_dic, all_log_metrics

    def training_forward(self, batch, batch_idx, opt_idx, hiddens):
        """
        Handle forward for each training case (distributed, single gpu, etc...)
        :param batch:
        :param batch_idx:
        :return:
        """
        # ---------------
        # FORWARD
        # ---------------
        # enable not needing to add opt_idx to training_step
        args = [batch, batch_idx]

        if len(self.optimizers) > 1:
            if self.has_arg('training_step', 'optimizer_idx'):
                args.append(opt_idx)
            else:
                raise ValueError(
                    f'Your LightningModule defines {len(self.optimizers)} optimizers but '
                    f'training_step is missing the "optimizer_idx" argument.'
                )

        # pass hiddens if using tbptt
        if self.truncated_bptt_steps is not None:
            args.append(hiddens)

        # distributed forward
        if self.use_ddp or self.use_ddp2 or self.use_dp:
            output = self.model(*args)

        # single GPU forward
        elif self.single_gpu:
            gpu_id = 0
            if isinstance(self.data_parallel_device_ids, list):
                gpu_id = self.data_parallel_device_ids[0]
            batch = self.transfer_batch_to_gpu(copy.copy(batch), gpu_id)
            args[0] = batch
            output = self.model.training_step(*args)

        # CPU forward
        else:
            output = self.model.training_step(*args)

        # allow any mode to define training_end
        if self.is_overriden('training_end'):
            model_ref = self.get_model()
            output = model_ref.training_end(output)

        # format and reduce outputs accordingly
        output = self.process_output(output, train=True)

        return output

    def run_training_epoch(self):
        # before epoch hook
        if self.is_function_implemented('on_epoch_start'):
            model = self.get_model()
            model.on_epoch_start()

        # run epoch
        dataloaders_zip = zip(self.get_train_dataloaders()[0], self.get_train_dataloaders()[1])
        for batch_idx, (batch_labeled, batch_unlabeled) in enumerate(dataloaders_zip):
            if batch_idx >= self.num_training_batches:
                break

            self.batch_idx = batch_idx

            model = self.get_model()
            model.global_step = self.global_step

            # ---------------
            # RUN TRAIN STEP
            # ---------------
            output = self.run_training_batch((batch_labeled, batch_unlabeled), batch_idx)
            batch_result, grad_norm_dic, batch_step_metrics = output

            # when returning -1 from train_step, we end epoch early
            early_stop_epoch = batch_result == -1

            # ---------------
            # RUN VAL STEP
            # ---------------
            is_val_check_batch = (batch_idx + 1) % self.val_check_batch == 0
            can_check_epoch = (self.current_epoch + 1) % self.check_val_every_n_epoch == 0
            should_check_val = (not self.disable_validation and can_check_epoch and
                                (is_val_check_batch or early_stop_epoch))

            # fast_dev_run always forces val checking after train batch
            if self.fast_dev_run or should_check_val:
                self.run_evaluation(test=self.testing)

            # when logs should be saved
            should_save_log = (batch_idx + 1) % self.log_save_interval == 0 or early_stop_epoch
            if should_save_log or self.fast_dev_run:
                if self.proc_rank == 0 and self.logger is not None:
                    self.logger.save()

            # when metrics should be logged
            should_log_metrics = batch_idx % self.row_log_interval == 0 or early_stop_epoch
            if should_log_metrics or self.fast_dev_run:
                # logs user requested information to logger
                self.log_metrics(batch_step_metrics, grad_norm_dic)

            self.global_step += 1
            self.total_batch_idx += 1

            # end epoch early
            # stop when the flag is changed or we've gone past the amount
            # requested in the batches
            if early_stop_epoch or self.fast_dev_run:
                break

        # epoch end hook
        if self.is_function_implemented('on_epoch_end'):
            model = self.get_model()
            model.on_epoch_end()

class UdaTrainer(Trainer, LoadData, TrainingLoop):

    def get_dataloaders(self, model):
        """
                Dataloaders are provided by the model
                :param model:
                :return:
                """
        self.init_train_dataloader(model)
        self.init_test_dataloader(model)
        self.init_val_dataloader(model)

        if self.use_ddp or self.use_ddp2:
            # wait for all processes to catch up
            torch.dist.barrier()

            # load each dataloader
            self.get_train_dataloaders()
            self.get_test_dataloaders()
            self.get_val_dataloaders()