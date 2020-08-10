import json
from argparse import Namespace

import SSL_playground.voc.datasets as ds
from SSL_playground.lightning_model import UDA

if __name__ == "__main__":

    with open('voc2007_hparams.json') as f:
        hparams= json.load(f)

    train_ds, valid_ds, num_classes = ds.get_train_test_datasets(hparams['dataset'], hparams['data_path'])
    model = UDA(Namespace(**hparams))
    model.set_datasets(train_ds, valid_ds, num_classes)

    model.fit_model()
    # model.load()
    # # out = model(test_dataset=valid_ds)
    # # print(out)
