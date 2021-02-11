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
parser.add_argument('--inference_dataset', type=str, choices=['unlabeled', 'labeled', 'test'], default=None)

args = parser.parse_args()
# =====================================================================================================
# training input:
# labeled_file_path: each line is a training image, the annotation file can be found correspondingly
# classes: CLASS_NAMES = ["aeroplane", "bicycle", "bird", "boat",..., "bottle"]
# =====================================================================================================
# 1. register the dataset
training_dataset_name = "api_dataset_train"
# dataset_rootdir = "/home/hxie/pipeline-sample/few-shot-object-detection/few-shot-object-detection/datasets/VOCdevkit/VOC2007"
# labeled_file_path = "/home/hxie/pipeline-sample/few-shot-object-detection/few-shot-object-detection/datasets/VOCdevkit/VOC2007/ImageSets/Main/trainval.txt"

# class_names = ['aeroplane', 'bicycle', 'boat', 'bottle', 'car', 'cat', 'chair',
#         'diningtable', 'dog', 'horse', 'person', 'pottedplant', 'sheep',
#         'train', 'tvmonitor', 'bird', 'bus', 'cow', 'motorbike', 'sofa']
# dataset_rootdir = '/datasets/lwll_datasets/development/{}/{}_full/train/'.format(args.dataset_name, args.dataset_name)
labeled_file_path = './session_data/{}/{}/train_{}.txt'.format(args.session_id, args.phase, args.stage)
unlabeled_file_path = './session_data/{}/{}/train_unlabeled_{}.txt'.format(args.session_id, args.phase, args.stage)
label_root = './session_data/{}/{}/labels'.format(args.session_id, args.phase)
class_names = [i for i in range(args.class_num)]
class_names_str = []
for class_name in class_names:
    if type(class_name) == int:
        class_name = str(class_name)
    class_names_str.append(class_name)
# register_api_dataset(training_dataset_name, dataset_rootdir, label_root, labeled_file_path, class_names_str, has_box=True)
    # print(hex(id(DatasetCatalog._REGISTERED)))

    # 2. train the model
# def train():
#     args.config_file = "configs/api_dataset/faster_rcnn_R_101_FPN_train.yaml"
#     args.num_gpus = 2
#     args.dist_url = "auto"
#     args.opts = ["MODEL.ROI_HEADS.NUM_CLASSES", str(len(class_names))]
#     print("Training Command Line Args:", args)

#     launch(
#         main_train,
#         args.num_gpus,
#         num_machines=args.num_machines,
#         machine_rank=args.machine_rank,
#         dist_url=args.dist_url,
#         args=(args,),
#     )

    # =====================================================================================================
    # evaluation input: 
    # test_file_path: each line is a test image
    # classes
    # =====================================================================================================

    # 1. register the dataset

test_dataset_name = "api_dataset_test"
# dataset_test_rootdir = '/datasets/lwll_datasets/development/{}/{}_full/test/'.format(args.dataset_name, args.dataset_name)
testing_file_path = './session_data/{}/{}/test.txt'.format(args.session_id, args.phase)
# register_api_dataset(test_dataset_name, dataset_test_rootdir, label_root, testing_file_path, class_names_str, has_box=False)

# def test():
#     # 2. test the model
#     args.config_file = "configs/api_dataset/faster_rcnn_R_101_FPN_test.yaml"
#     args.eval_only = True
#     args.num_gpus = 2
#     args.dist_url = "auto"
#     args.opts = ["MODEL.ROI_HEADS.NUM_CLASSES", str(len(class_names))]
#     print("Testing Command Line Args:", args)

#     launch(
#         main_test,
#         args.num_gpus,
#         num_machines=args.num_machines,
#         machine_rank=args.machine_rank,
#         dist_url=args.dist_url,
#         args=(args,),
#     )


if __name__ == "__main__":

    with open('consistency_learning_utils/cl_hparams.json') as f:
        hparams = json.load(f)

    with open(labeled_file_path) as f:
        num_labeled = len(f.readlines())

    hparams['output_csv'] = args.output_csv
    hparams['class_num'] = args.class_num
    hparams['stage'] = args.stage

    batch_per_epoch = int((num_labeled + hparams['batch_size'] - 1) / hparams['batch_size'])
    # hparams['min_epochs'] = 5 * int(hparams['num_warmup_steps'] / batch_per_epoch) # TODO
    # hparams['max_epochs'] = int(hparams['total_steps'] / batch_per_epoch) # TODO
    # if args.stage == 0: # TODO
    #     hparams['max_epochs'] = hparams['min_epochs']  # no validation set for stage 0 # TODO
    hparams['patience_epochs'] = max(5, hparams['min_epochs'] // 2)

    print("Will train for {}-{} epochs with {} patience, {} batches/epoch".format(
        hparams['min_epochs'],
        hparams['max_epochs'],
        hparams['patience_epochs'],
        batch_per_epoch
    ))
    
    best_version_name = ''
    best_val_loss = 10000000
    best_student_loss = 10000000
    best_teacher_loss = 10000000
    start_version_name = hparams['version_name']
    base_checkpoint_val_loss = 10000000

    loss_dict = {}
    
    attempts_by_stage = [1, 1, 1, 1, 1, 1, 1, 1]
    attempts = attempts_by_stage[args.stage]

    best_base_checkpoint_filename = './checkpoints/best_base_checkpoint.pth'

    initialization_times = {'base': ['from_coco'], 'adaption': ['from_coco', 'from_base']}
    for initialization in initialization_times[args.phase]:
        for experiment_id in range(attempts):
            hparams['version_name'] = '{}_{}_{}_{}_{}_'.format(args.dataset_name, args.phase, args.stage, initialization, experiment_id) + start_version_name

            model = STAC(argparse.Namespace(**hparams))
            inference_path = labeled_file_path
            if args.inference_dataset == 'unlabeled':
                inference_path = unlabeled_file_path
            elif args.inference_dataset == 'test':
                inference_path = testing_file_path
            model.set_datasets(labeled_file_path, unlabeled_file_path, inference_path, label_root)
            model.test_from_checkpoint(args.ckpt_path)

