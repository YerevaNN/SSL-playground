import new_loop as loop
import requests
from tqdm import tqdm
from pipeline_utils import *
from PIL import Image
import os
import pytorch_lightning as pl

from consistency_learning_utils.lightning_model import STAC

import argparse
parser = argparse.ArgumentParser(description="STAC training")
parser.add_argument('--dataset_name', type=str, default=None)
parser.add_argument('--session_id', type=str, default=None)
parser.add_argument('--phase', type=str, default=None)
parser.add_argument('--stage', type=int, default=None)
parser.add_argument('--class_num', type=int, default=None)
parser.add_argument('--output_csv', type=str, default=None)

args = parser.parse_args()

training_dataset_name = "api_dataset_train"
labeled_file_path = './session_data/{}/{}/train_{}.txt'.format(args.session_id, args.phase, args.stage)
unlabeled_file_path = './session_data/{}/{}/train_unlabeled_{}.txt'.format(args.session_id, args.phase, args.stage)
external_val_file_path = '/lwll/external/{}/validation.txt'.format(args.dataset_name)
external_val_label_root = '/lwll/external/{}/validation_labels'.format(args.dataset_name)
label_root = './session_data/{}/{}/labels'.format(args.session_id, args.phase)

class_names = [i for i in range(args.class_num)]
class_names_str = []
for class_name in class_names:
    if type(class_name) == int:
        class_name = str(class_name)
    class_names_str.append(class_name)

test_dataset_name = "api_dataset_test"
testing_file_path = './session_data/{}/{}/test.txt'.format(args.session_id, args.phase)

if __name__ == "__main__":

    with open('consistency_learning_utils/cl_hparams.json') as f:
        hparams = json.load(f)
    
    # pl.seed_everything(hparams['seed'])

    with open(labeled_file_path) as f:
        num_labeled = len(f.readlines())

    hparams['output_csv'] = args.output_csv
    hparams['class_num'] = args.class_num
    hparams['stage'] = args.stage

    hparams['batches_per_epoch'] = int((num_labeled + hparams['batch_size'] - 1) / hparams['batch_size'])

    best_version_name = ''
    best_val_loss = 10000000
    best_student_loss = 10000000
    best_teacher_loss = 10000000
    base_checkpoint_val_loss = 10000000

    loss_dict = {}

    attempts_by_stage = [1, 1, 1, 1, 1, 1, 1, 1]
    attempts = attempts_by_stage[args.stage]

    best_base_checkpoint_filename = './checkpoints/best_base_checkpoint.pth'

    # initialization_times = {'base': ['from_coco'], 'adaption': ['from_coco', 'from_base']}
    initialization_times = {'base': ['from_coco'], 'adaption': ['from_coco']} # TODO
    for initialization in initialization_times[args.phase]:
        for experiment_id in range(attempts):
            hparams['version_name'] = '{}_{}_{}_{}_{}_'.format(args.dataset_name, args.phase, args.stage, initialization, experiment_id)

            model = STAC(argparse.Namespace(**hparams))
            model.set_datasets(labeled_file_path, unlabeled_file_path, testing_file_path,
                               external_val_file_path, external_val_label_root, label_root)

            if initialization == 'from_base':
                if os.path.exists(best_base_checkpoint_filename):
                    print("Loading from best_base_checkpoint")
                    model.load_checkpoint(best_base_checkpoint_filename)
                else:
                    print("best_base_checkpoint does not exist. Skipping")
                    continue
            cur_teacher_loss, cur_student_loss = model.fit_model()
            cur_val_loss = min(cur_teacher_loss, cur_student_loss)
            best_teacher_loss = min(best_teacher_loss, cur_teacher_loss)
            best_student_loss = min(best_student_loss, cur_student_loss)
            if cur_val_loss < best_val_loss:
                best_val_loss = cur_val_loss
                best_version_name = hparams['version_name']
            loss_dict['val_loss_for_run_{}'.format(experiment_id)] = cur_val_loss
            if args.phase == 'base' and args.stage == 7:
                if cur_val_loss < base_checkpoint_val_loss:
                    base_checkpoint_val_loss = cur_val_loss
                    print("Saving to best_base_checkpoint")
                    # model.save_checkpoint(best_base_checkpoint_filename) # TODO

    print(json.dumps(loss_dict, indent=True))
    print('I chose {} with loss={}'.format(best_version_name, best_val_loss))

    hparams['version_name'] = best_version_name

    best_model = STAC(argparse.Namespace(**hparams))
    best_model.set_datasets(labeled_file_path, unlabeled_file_path, testing_file_path,
                            external_val_file_path, external_val_label_root, label_root)
    eps = 1e-10
    best_model.set_test_with_student(True)
    if args.stage == 7: #best_student_loss - eps > best_teacher_loss or
        best_model.set_test_with_student(False)
    best_model.test_from_best_checkpoint()
