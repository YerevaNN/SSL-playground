# import copy
# import warnings
# import numpy as np

# from pytorch_lightning.trainer import data_loading, training_loop
# from torch.utils.data import DistributedSampler

# class LoadData(data_loading.TrainerDataLoadingMixin):
#     def init_train_dataloader(self, model):
#         self.get_train_dataloaders = model.train_dataloader

#         self.num_training_batches = 0

#         # determine number of validation batches
#         # val datasets could be none, 1 or 2+
#         if self.get_train_dataloaders() is not None:
#             self._percent_range_check('train_percent_check')

#             self.num_training_batches = len(self.get_train_dataloaders()[0])
#             # self.num_training_batches = sum(len(dataloader) for dataloader in self.get_train_dataloaders())
#             self.num_training_batches = int(self.num_training_batches * self.train_percent_check)

#         if isinstance(self.val_check_interval, int):
#             self.val_check_batch = self.val_check_interval
#             if self.val_check_batch > self.num_training_batches:
#                 raise ValueError(
#                     f"`val_check_interval` ({self._check_interval}) must be less than or equal "
#                     f"to the number of the training batches ({self.num_training_batches}). "
#                     f"If you want to disable validation set `val_percent_check` to 0.0 instead.")
#         else:
#             self._percent_range_check('val_check_interval')

#             self.val_check_batch = int(self.num_training_batches * self.val_check_interval)
#             self.val_check_batch = max(1, self.val_check_batch)

#         on_ddp = self.use_ddp or self.use_ddp2
#         if on_ddp and self.get_train_dataloaders() is not None:
#             for dataloader in self.get_val_dataloaders():
#                 if not isinstance(dataloader.sampler, DistributedSampler):
#                     msg = """
#                             Your val_dataloader(s) don't use DistributedSampler.

#                             You're using multiple gpus and multiple nodes without using a
#                             DistributedSampler to assign a subset of your data to each process.
#                             To silence this warning, pass a DistributedSampler to your DataLoader.

#                             ie: this:
#                             dataset = myDataset()
#                             dataloader = Dataloader(dataset)

#                             becomes:
#                             dataset = myDataset()
#                             dist_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
#                             dataloader = Dataloader(dataset, sampler=dist_sampler)

#                             If you want each process to load the full dataset, ignore this warning.
#                             """
#                     if msg not in self.shown_warnings and self.proc_rank == 0:
#                         self.shown_warnings.add(msg)
#                         warnings.warn(msg)
#                     break
