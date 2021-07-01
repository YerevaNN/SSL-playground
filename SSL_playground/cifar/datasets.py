from torchvision import datasets
import torch

def get_train_test_datasets(dataset_name, path):

    assert dataset_name in datasets.__dict__, "Unknown dataset name {}".format(dataset_name)
    fn = datasets.__dict__[dataset_name]

    train = fn(root=path, train=True, download=True)
    val_ds, train_ds = torch.utils.data.random_split(train, [4000, 46000])
    test_ds = fn(root=path, train=False, download=False)

    return train_ds, val_ds, test_ds, len(train.classes)
