import numpy as np
import torch

from torch.utils.data import Subset, Dataset, DataLoader

from torchvision.transforms import Compose, ToTensor, Normalize, Pad, RandomCrop, RandomHorizontalFlip

from SSL_playground.helpers import autoaugment
from SSL_playground.helpers.transforms import RandomErasing

def set_seed(seed):
    import random
    import numpy as np
    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_train_test_loaders(train_ds, test_ds, num_classes, num_labelled_samples, batch_size, num_workers,
                           unlabelled_batch_size=None, pin_memory=True):

    n_valid = 0.2
    # if "cifar" in dataset_name.lower():
    #     train_ds, test_ds, num_classes = get_cifar_train_test_datasets(dataset_name, path)
    # else:
    #     raise RuntimeError("Unknown dataset '{}'".format(dataset_name))

    train_labelled_ds, train_unlabelled_ds = stratified_train_labelled_unlabelled_split(train_ds,
                                                   num_labelled_samples=num_labelled_samples,
                                                   num_classes=num_classes, seed=12)


    # if "cifar" in dataset_name.lower():
    train_transform = Compose([
        Pad(4),
        RandomCrop(32, fill=128),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        RandomErasing(scale=(0.1, 0.33)),
    ])

    test_transform = Compose([
        ToTensor(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    unsupervised_train_transformation = Compose([
        Pad(4),
        RandomCrop(32, fill=128),
        autoaugment.CIFAR10Policy()
    ])
    unsup_train_transformation =Compose([
        ToTensor(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        RandomErasing(scale=(0.1, 0.33)),
    ])
    train_labelled_ds = TransformedDataset(train_labelled_ds,
                                           transform_fn=lambda dp: (train_transform(dp[0]), dp[1]), shuffle=True,
                                           shuffle_seed=1)
    test_ds = TransformedDataset(test_ds, transform_fn=lambda dp: (test_transform(dp[0]), dp[1]), shuffle=True,
                                 shuffle_seed=1)

    original_transform = lambda dp: train_transform(dp[0])
    augmentation_transform = lambda dp: unsupervised_train_transformation(dp[0])
    image_transform = lambda dp: unsup_train_transformation(dp)
    train_unlabelled_ds = TransformedDataset(train_unlabelled_ds,
                                             UDATransform(original_transform, augmentation_transform, image_transform),
                                             shuffle=True, shuffle_seed=1)


    if unlabelled_batch_size is None:
        unlabelled_batch_size = batch_size

    train_labelled_loader = DataLoader(train_labelled_ds, batch_size=batch_size, num_workers=num_workers,
                                       pin_memory=pin_memory, shuffle=True)

    train_unlabelled_loader = DataLoader(train_unlabelled_ds, batch_size=unlabelled_batch_size, num_workers=num_workers,
                                         pin_memory=pin_memory, shuffle=True)

    test_loader = DataLoader(test_ds, batch_size=batch_size * 2, num_workers=num_workers, pin_memory=pin_memory,
                             shuffle=True)

    return train_labelled_loader, train_unlabelled_loader, test_loader

class TransformedDataset(Dataset):

    def __init__(self, ds, transform_fn, shuffle, shuffle_seed):
        assert isinstance(ds, Dataset)
        assert callable(transform_fn)
        self.ds = ds
        self.transform_fn = transform_fn
        self.length = len(self.ds)
        self.permutation = torch.arange(end=self.length)
        if shuffle:
            if shuffle_seed is None:
                raise ValueError("If shuffle is set to True shuffle_seed must be specified")
            generator = torch.Generator().manual_seed(shuffle_seed)
            self.permutation = torch.randperm(self.length, generator=generator)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        index = self.permutation[index]
        dp = self.ds[index]
        return self.transform_fn(dp)


class UDATransform:

    def __init__(self, original_transform, augmentation_transform, imagetransform, copy=False):
        self.original_transform = original_transform
        self.augmentation_transform = augmentation_transform
        self.imagetransform = imagetransform
        self.copy = copy

    def __call__(self, dp):
        if self.copy:
            aug_dp = dp.copy()
        else:
            aug_dp = dp
        _, label = dp
        tdp1 = self.original_transform(dp)
        tdp2, vector = self.augmentation_transform(aug_dp)
        tdp2 = self.imagetransform(tdp2)
        return (tdp1, label), (tdp2, vector)

def stratified_train_labelled_unlabelled_split(ds, num_labelled_samples, num_classes, seed=None):
    labelled_indices = []
    unlabelled_indices = []

    if seed is not None:
        np.random.seed(seed)

    indices = np.random.permutation(len(ds))

    class_counters = list([0] * num_classes)
    max_counter = num_labelled_samples // num_classes
    for i in indices:
        dp = ds[i]

        if num_labelled_samples < sum(class_counters):
            unlabelled_indices.append(i)
        else:
            y = dp[1]
            c = class_counters[y]
            if c < max_counter:
                class_counters[y] += 1
                labelled_indices.append(i)
            else:
                unlabelled_indices.append(i)

    assert len(set(labelled_indices) & set(unlabelled_indices)) == 0, \
        "{}".format(set(labelled_indices) & set(unlabelled_indices))

    train_labelled_ds = Subset(ds, labelled_indices)
    train_unlabelled_ds = Subset(ds, unlabelled_indices)
    return train_labelled_ds, train_unlabelled_ds

class CombineDataset(Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets
    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)
    def __len__(self):
        return min(len(d) for d in self.datasets)
