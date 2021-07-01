import json
from argparse import Namespace

import SSL_playground.cifar.datasets
from SSL_playground.lightning_model import MPL

if __name__ == "__main__":

    with open('cifar10_simple_hparams.json') as f:
        hparams= json.load(f)

    train_ds, valid_ds, test_ds, num_classes = SSL_playground.cifar.datasets.get_train_test_datasets(hparams['dataset'], hparams['data_path'])
    model = MPL(Namespace(**hparams))
    model.set_datasets(train_ds, valid_ds, test_ds, num_classes)

    model.fit_model()
    # model.load()
    # # out = model(test_dataset=valid_ds)
    # # print(out)