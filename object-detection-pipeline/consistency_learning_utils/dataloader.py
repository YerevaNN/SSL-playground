import numpy as np
import torch
import random
import os

from torch.utils.data import Dataset, DataLoader

from torchvision.transforms import Compose, ToTensor, ToPILImage
from PIL import Image

from .helpers import autoaugment
# from .helpers.transforms import RandomErasing

class ConcatDataset(Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        if len(self.datasets) == 1:
            return tuple((self.datasets[0][i], None))
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)

class MyDataset(Dataset):
    def __init__(self,
                 file_path: str, target_required: bool = False,
                 img_name_required: bool = False,
                 label_root: str = None,
                 end_to_take: str = None,
                 part_to_take: float = 0.) -> None:
        super().__init__()

        self.file_path = file_path
        self.target_required = target_required
        self.img_name_required = img_name_required
        self.label_root = label_root
        self.end_to_take = end_to_take
        self.part_to_take = part_to_take

        with open(self.file_path) as f:
            self.file_lines = f.readlines()
            if end_to_take is not None:
                spliter_placement = part_to_take
                if end_to_take == 'back':
                    spliter_placement = 1 - spliter_placement
                splitter_id = int(len(self.file_lines) * spliter_placement)
                if end_to_take == 'front':
                    self.file_lines = self.file_lines[:splitter_id]
                elif end_to_take == 'back':
                    self.file_lines = self.file_lines[splitter_id:]
                else:
                    raise NotImplementedError

    def __len__(self):
        return len(self.file_lines)

    def __get_target__(self, image_name: str, width: int, height: int):
        image_name = '.'.join(image_name.split('.')[:-1]) + '.txt'  # replace .jpg (or whatever) to .txt
        label_path = os.path.join(self.label_root, image_name)

        target = []
        with open(label_path) as f:
            for line in f:
                line_arr = [float(t) for t in line.split(' ')]
#                 label, x_center_rel, y_center_rel, bbox_width, bbox_height = line_arr
                label, xmin, ymin, xmax, ymax = line_arr
                box = {}
                box['label'] = label
                box['bndbox'] = {}
#                 box['bndbox']['xmin'] = float(x_center_rel - bbox_width/2) * width
#                 box['bndbox']['ymin'] = float(y_center_rel - bbox_height/2) * height
#                 box['bndbox']['xmax'] = float(x_center_rel + bbox_width/2) * width
#                 box['bndbox']['ymax'] = float(y_center_rel + bbox_height/2) * height
                box['bndbox']['xmin'] = int(xmin)
                box['bndbox']['ymin'] = int(ymin)
                box['bndbox']['xmax'] = int(xmax)
                box['bndbox']['ymax'] = int(ymax)
                target.append(box)

        return target

    def __getitem__(self, index: int):
        img_path = self.file_lines[index].strip()  # new line!
 
        with open(img_path, 'rb') as f:
            image = Image.open(f)
            width, height = image.size
            image = np.array(image)  # otherwise nothing will work! but should we transpose this?
        image_name = img_path.split('/')[-1]
        if len(image.shape) == 2:
            image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
        if not self.target_required:
            if self.img_name_required:
                image_name = image_name[:-4]  # remove .jpg
                return image, image_name, img_path
            else:
                return image, img_path
        target = self.__get_target__(image_name, width, height)
        return image, target, img_path


def set_seed(seed):
    import random
    import numpy as np
    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def voc_collate_fn(batch):
    result = [[] for _ in range(len(batch[0]))]
    for data in batch:
        for i, item in enumerate(data):
            result[i].append(item)
    return result

def get_train_test_loaders(labeled_file_path, unlabelled_file_path, testing_file_path, label_root, batch_size,
                           num_workers, stage=0, unlabelled_batch_size=None, validation_part=0.2, pin_memory=True):

    train_unlabelled_ds = MyDataset(unlabelled_file_path, target_required=False, img_name_required=False)
    test_ds = MyDataset(testing_file_path, target_required=False, img_name_required=True)
    if stage == 0 or validation_part == 0:
        train_labelled_ds = MyDataset(labeled_file_path, target_required=True, img_name_required=False,
                                      label_root=label_root)
        val_ds = MyDataset(labeled_file_path, target_required=True, img_name_required=False,
                           label_root=label_root)
    else:
        train_labelled_ds = MyDataset(labeled_file_path, target_required=True, img_name_required=False,
                                      label_root=label_root, end_to_take='front', part_to_take=1-validation_part)
        val_ds = MyDataset(labeled_file_path, target_required=True, img_name_required=False,
                           label_root=label_root, end_to_take='back', part_to_take=validation_part)

    if stage == 7:
       train_unlabelled_ds = train_labelled_ds 

    strong_augment_transform = Compose([
        ToPILImage(),
        autoaugment.CIFAR10Policy(),
        ToTensor()
    ])
    no_transform = Compose([
        ToTensor()
    ])
    train_labelled_ds = TransformedDataset(train_labelled_ds, transform_fn=lambda dp: (strong_augment_transform(dp[0]), dp[1], dp[2]),
                                           shuffle=True, shuffle_seed=1)
    val_ds = TransformedDataset(val_ds, transform_fn=lambda dp: (no_transform(dp[0]), dp[1], dp[2]),
                                shuffle=False, shuffle_seed=1)
    test_ds = TransformedDataset(test_ds, transform_fn=lambda dp: (no_transform(dp[0]), dp[1], dp[2]), shuffle=False,
                                 shuffle_seed=1)  # We do not have labels for test!

    train_unlabelled_ds = TransformedDataset(
        train_unlabelled_ds, STACTransform(
            lambda dp: (no_transform(dp[0]), dp[1]),
            lambda dp: (strong_augment_transform(dp[0]), dp[1])
        ),
        shuffle=True, shuffle_seed=1)

    if unlabelled_batch_size is None:
        unlabelled_batch_size = batch_size

    train_dataset = ConcatDataset(train_labelled_ds, train_unlabelled_ds)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers,
                                       pin_memory=pin_memory, collate_fn=voc_collate_fn, shuffle=True)

    test_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory,
                             shuffle=False, collate_fn=voc_collate_fn)

    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers,
                            pin_memory=pin_memory, collate_fn=voc_collate_fn, shuffle=False)


    return train_loader, test_loader, val_loader


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


class STACTransform:

    def __init__(self, original_transform, augmentation_transform, copy=True):
        self.original_transform = original_transform
        self.augmentation_transform = augmentation_transform
        self.copy = copy

    def __call__(self, dp):
        if self.copy:
            aug_dp = dp[0].copy()
        else:
            aug_dp = dp[0]
#         _, label = dp  # no label is available for unlabeled examples!
        tdp1 = self.original_transform(dp[0])
        tdp2 = self.augmentation_transform(aug_dp)
        vector = np.zeros(14)  # fake vector
        return (tdp1, None, dp[1]), (tdp2, vector, dp[1])


class CombineDataset(Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)
