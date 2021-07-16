from pipeline_utils import *
import os
import shutil
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
parser.add_argument('--teacher_init_path', type=str, default=None)

parser.add_argument('--experiment_name', type=str, default=None)
parser.add_argument('--learning_rate', type=float, default=None)
parser.add_argument('--gradient_clip_threshold', type=float, default=None)
parser.add_argument('--confidence_threshold', type=float, default=None)
parser.add_argument('--box_score_thresh', type=float, default=None)
parser.add_argument('--weight_decay', type=float, default=None)
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--EMA_keep_rate', type=float, default=None)
parser.add_argument('--gamma', type=float, default=None)
parser.add_argument('--augmentation', type=int, default=None)
parser.add_argument('--initialization', type=str, default=None)
parser.add_argument('--reuse_classifier', type=str, default=None)
parser.add_argument('--check_val_steps', type=int, default=None)
parser.add_argument('--batch_size', type=int, default=None)

args = parser.parse_args()

training_dataset_name = "api_dataset_train"
phase_folder = './session_data/{}/{}/'.format(args.session_id, args.phase)
labeled_file_path = '{}/train_{}.txt'.format(phase_folder, args.stage)
unlabeled_file_path = '{}/train_unlabeled_{}.txt'.format(phase_folder, args.stage)
external_val_file_path = '/lwll/external/{}/validation.txt'.format(args.dataset_name)
external_val_label_root = '/lwll/external/{}/validation_labels'.format(args.dataset_name)
label_root = '{}/labels'.format(phase_folder)

class_names = [i for i in range(args.class_num)]
class_names_str = []
for class_name in class_names:
    if type(class_name) == int:
        class_name = str(class_name)
    class_names_str.append(class_name)

test_dataset_name = "api_dataset_test"
testing_file_path = '{}/test.txt'.format(phase_folder)

if __name__ == "__main__":

    with open('consistency_learning_utils/cl_hparams.json') as f:
        hparams = json.load(f)

    with open(labeled_file_path) as f:
        num_labeled = len(f.readlines())

    hparams['class_num'] = args.class_num
    hparams['stage'] = args.stage
    hparams['phase_folder'] = phase_folder
    hparams['session_id'] = args.session_id

    argsdict = vars(args)

    for key in ['experiment_name', 'learning_rate', 'gradient_clip_threshold',
                'confidence_threshold', 'weight_decay', 'seed', 'EMA_keep_rate', 'gamma',
                'initialization', 'reuse_classifier', 'check_val_steps', 'batch_size',
                'box_score_thresh', 'augmentation', 'teacher_init_path']:
        if key in argsdict and argsdict[key] is not None:
            print("Overriding {} to {}".format(key, argsdict[key]))
            hparams[key] = argsdict[key]

    hparams['total_steps_teacher'] = hparams['total_steps_teacher_initial'] + args.stage * hparams['total_steps_teacher_stage_inc']
    hparams['total_steps_student'] = hparams['total_steps_student_initial'] + args.stage * hparams['total_steps_student_stage_inc']
    hparams['batches_per_epoch'] = int((num_labeled + hparams['batch_size'] - 1) / hparams['batch_size'])

    pl.seed_everything(hparams['seed'])

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
            hparams['version_name'] = '{}_{}_{}_{}_{}_{}'.format(
                args.dataset_name, args.phase, args.stage, initialization, experiment_id,
                hparams['experiment_name'])

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
    shutil.copy(best_model.output_csv, args.output_csv)

