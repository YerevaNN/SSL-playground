# from fsdet.data.datasets.api_dataset import register_api_dataset
# from tools.train_net import main as main_train
# from tools.test_net import main as main_test
# from fsdet.engine import launch, default_argument_parser
# from fsdet.data import DatasetCatalog
# from fsdet.checkpoint import DetectionCheckpointer

import new_loop as loop
import requests
from tqdm import tqdm
from pipeline_utils import *
from PIL import Image
import os

from consistency_learning_utils.lightning_model import STAC

import argparse
parser = argparse.ArgumentParser(description="STAC training")
parser.add_argument('--dataset_name', type=str, default=None)
parser.add_argument('--session_id', type=str, default=None)
parser.add_argument('--phase', type=str, default=None)
parser.add_argument('--stage', type=int, default=None)
parser.add_argument('--class_num', type=int, default=None)
parser.add_argument('--output_csv', type=str, default=None)
parser.add_argument('--ckpt_path', type=str, default=None)
parser.add_argument('--data_path', type=str, default=None)

args = parser.parse_args()
training_dataset_name = "api_dataset_train"
labeled_file_path = '{}/train.txt'.format(args.data_path)
unlabeled_file_path = '{}/train_unlabeled.txt'.format(args.data_path)
label_root = '{}/labels'.format(args.data_path)
class_names = [i for i in range(args.class_num)]
class_names_str = []
for class_name in class_names:
    if type(class_name) == int:
        class_name = str(class_name)
    class_names_str.append(class_name)

test_dataset_name = "api_dataset_test"
testing_file_path = '{}/test.txt'.format(args.data_path)

if __name__ == "__main__":

    with open('consistency_learning_utils/cl_hparams.json') as f:
        hparams = json.load(f)

    with open(labeled_file_path) as f:
        num_labeled = len(f.readlines())

    hparams['output_csv'] = args.output_csv
    hparams['class_num'] = args.class_num
    hparams['stage'] = args.stage

    batch_per_epoch = int((num_labeled + hparams['batch_size'] - 1) / hparams['batch_size'])
    hparams['patience_epochs'] = max(50, hparams['min_epochs'] // 2)
    
    model = STAC(argparse.Namespace(**hparams))
    model.set_datasets(labeled_file_path, unlabeled_file_path, testing_file_path, label_root)
    teacher_loss, student_loss = model.fit_model()
    model.set_test_with_student(False)
    model.test()
