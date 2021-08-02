import numpy as np

from torch.utils.data import Subset, Dataset, DataLoader

from torchvision.transforms import Compose, ToTensor, Normalize, Pad, RandomCrop, RandomHorizontalFlip


from . import autoaugment
from .transforms import RandomErasing


class TransformedDataset(Dataset):

    def __init__(self, ds, transform_fn):
        assert isinstance(ds, Dataset)
        assert callable(transform_fn)
        self.ds = ds
        self.transform_fn = transform_fn

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        dp = self.ds[index]
        return self.transform_fn(dp)


class STACTransform:

    def __init__(self, original_transform, augmentation_transform, copy=False):
        self.original_transform = original_transform
        self.augmentation_transform = augmentation_transform
        self.copy = copy

    def __call__(self, dp):
        if self.copy:
            aug_dp = dp.copy()
        else:
            aug_dp = dp
        tdp1 = self.original_transform(dp)
        tdp2 = self.augmentation_transform(aug_dp)
        return tdp1, tdp2

