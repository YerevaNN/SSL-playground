import glob
import os

from absl import app
from absl import flags

from tqdm import tqdm

import new_loop as loop
import requests
from pipeline_utils import *
from PIL import Image
from PIL.ExifTags import TAGS
from tqdm import tqdm
import csv
import json
import numpy as np
from collections import defaultdict

def main(_):
    FLAGS, resume_session, task = loop.create_default_task()
    phases = ['base', 'adaption']
    for phase in phases:
        dataset_name = task['session']['Session_Status']['current_dataset']['name']
        # TRAIN_DATASET_PATH = '/home/khazhak/lwll_datasets/development/{}/{}_full/train'.format(dataset_name, dataset_name)
        # TEST_DATASET_PATH = '/home/khazhak/lwll_datasets/development/{}/{}_full/test'.format(dataset_name, dataset_name)
        TRAIN_DATASET_PATH = '/lwll/external/{}/{}_full/train'.format(dataset_name, dataset_name)
        TEST_DATASET_PATH = '/lwll/evaluation/{}/{}_full/test'.format(dataset_name, dataset_name)

        training_classes = task['session']['Session_Status']['current_dataset']['classes']
        class_name_to_id = {}
        class_id_to_name = {}
        id = 1
        for name in training_classes:
            class_name_to_id.update({name:id})
            class_id_to_name.update({id:name})
            id += 1

        unlabeled_filename = set()
        labeled_filename = set()
        current_labels = []

        training_image_metadata = {}
        testing_image_metadata = {}

        print("Loading Training Images Metadata...")
        for index, file in enumerate(os.listdir(TRAIN_DATASET_PATH)):
            unlabeled_filename.add(file)
            file_full_path = os.path.join(TRAIN_DATASET_PATH, file)
            image = Image.open(file_full_path)
            training_image_metadata.update({file:image.size})

        print("Loading Testing Images Metadata...")
        for index, file in enumerate(os.listdir(TEST_DATASET_PATH)):
            file_full_path = os.path.join(TEST_DATASET_PATH, file)
            image = Image.open(file_full_path)
            testing_image_metadata.update({file:image.size})

        budget_stages = task['session']['Session_Status']['current_label_budget_stages']
        current_task_dir = os.path.join('./session_data', task['session_token'])
        current_task_dir = os.path.join(current_task_dir, "{}".format(phase))

        label_dir = os.path.join(current_task_dir, 'labels')
        model_dir = os.path.join(current_task_dir, 'models')

        if not os.path.exists(current_task_dir):
            os.makedirs(current_task_dir)
            os.makedirs(label_dir)
            os.makedirs(model_dir)

        for stage, budget in enumerate(task['budgets']):
            print('Start Checkpoint {}'.format(stage))

            requested_runs = 0
            print('Querying for Labels')
            if stage < 4:
                label_response = loop.get_json('seed_labels', session_token=task['session_token'])
                new_labels = label_response['Labels']
            else:
                new_labels = []
                remainder = set([t for t in unlabeled_filename])
                while int(loop.get_json('session_status', session_token=task['session_token'])['Session_Status']['budget_left_until_checkpoint']) > 0 and len(unlabeled_filename) > 0:
                    q = int(loop.get_json('session_status', session_token=task['session_token'])['Session_Status']['budget_left_until_checkpoint'])
                    print("Will try to request boxes for min({}, {}) images".format(q, len(remainder)))
                    q = min(q, len(remainder))
                    if q == 0:
                        break
                    current_request_file = random.sample(remainder, q)
                    requested_runs += 1
                    try:
                        print("Requesting boxes for {} images, e.g. {}".format(len(current_request_file), current_request_file[0]))
                        label_response = loop.post_json('query_labels', {'example_ids': current_request_file}, session_token=task['session_token'])
                        new_labels = new_labels + label_response['Labels']

                        for label in new_labels: # to avoid requesting the same image twice
                            image_id = label['id']
                            if image_id in remainder:
                                remainder.remove(image_id)
                    except Exception as e:
                        print('Had Issue With Requesting Labeled Data. Move on to next Training with current labeled data')
                        raise e


                print('requested {} Times, {} Label Requested'.format(
                    requested_runs, len(new_labels)))

            # ====== Writing to Label File ======
            with open(os.path.join(current_task_dir, 'new_labels_{}.json'.format(stage)), 'w') as f:
                json.dump(new_labels, f)

            current_labels = current_labels + new_labels
            print("So far we have {} bboxes at stage {}".format(len(current_labels), stage))
            current_labels_by_image = defaultdict(list)

            for label in current_labels:
                image_id = label['id']
                if image_id in unlabeled_filename:
                    unlabeled_filename.remove(image_id)
                labeled_filename.add(image_id)
                current_labels_by_image[image_id].append({
                    'bbox': label['bbox'],
                    'class': class_name_to_id[label['class']]
                })
            print("So far we have {} labeled images at stage {}".format(len(current_labels_by_image), stage))
            print("We have {} labeled and {} unlabeled images at stage {}".format(
                len(labeled_filename), len(unlabeled_filename), stage
            ))

            print('Writing label files')
            for index, image_file in enumerate(current_labels_by_image.keys()):
                ext = image_file.split('.')[-1]
                image_label_file = os.path.join(label_dir, image_file.replace(ext, 'txt'))
                image_width, image_height = training_image_metadata[image_file]

                with open(image_label_file, 'w') as label_file:
                    for label in current_labels_by_image[image_file]:
                        class_id = label['class']
                        bbox_abs = [int(float(t.strip())) for t in label['bbox'].split(',')]
                        xmin, ymin, xmax, ymax = bbox_abs
                        xmin = max(0, xmin - 1)
                        xmax = min(image_width - 1, xmax + 1)
                        ymin = max(0, ymin - 1)
                        ymax = min(image_height - 1, ymax + 1)

                        #                         x_min_rel = float(bbox_abs[0])/image_width
                        #                         y_min_rel = float(bbox_abs[1])/image_height
                        #                         x_max_rel = float(bbox_abs[2])/image_width
                        #                         y_max_rel = float(bbox_abs[3])/image_height

                        #                         bbox_width = x_max_rel - x_min_rel
                        #                         bbox_height = y_max_rel - y_min_rel
                        #                         x_center_rel = x_min_rel + bbox_width/2
                        #                         y_center_rel = y_min_rel + bbox_height/2

                        #                         new_line = '{} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(class_id, x_center_rel, y_center_rel, bbox_width, bbox_height)
                        new_line = '{} {} {} {} {}\n'.format(class_id, xmin, ymin, xmax, ymax)
                        label_file.write(new_line)

            training_file_path = os.path.join(current_task_dir, 'train_{}.txt'.format(stage))
            with open(training_file_path, 'w') as train_file:
                for image_name in labeled_filename:
                    new_line = '{}\n'.format(os.path.join(TRAIN_DATASET_PATH, image_name))
                    train_file.write(new_line)

            test_file_path  = os.path.join(current_task_dir, 'test.txt')
            with open(test_file_path, 'w') as train_file:
                for test_file_name in testing_image_metadata.keys():
                    new_line = '{}\n'.format(os.path.join(TEST_DATASET_PATH, test_file_name))
                    train_file.write(new_line)

            #             if os.path.exists('./session_data/output/inference/res_final.csv'):
            #                 os.remove('./session_data/output/inference/res_final.csv')

            training_unlabeled_file_path = os.path.join(current_task_dir, 'train_unlabeled_{}.txt'.format(stage))
            with open(training_unlabeled_file_path, 'w') as train_file:
                unlabeled_written = 0
                while unlabeled_written <= len(labeled_filename) and len(unlabeled_filename) > 0:
                    for image_name in unlabeled_filename:
                        new_line = '{}\n'.format(os.path.join(TRAIN_DATASET_PATH, image_name))
                        train_file.write(new_line)
                        unlabeled_written += 1

            train_metadata_path = os.path.join(current_task_dir, 'train_metadata.json')
            with open(train_metadata_path, 'w') as f:
                json.dump(training_image_metadata, f)
            test_metadata_path = os.path.join(current_task_dir, 'test_metadata.json')
            with open(test_metadata_path, 'w') as f:
                json.dump(testing_image_metadata, f)

            output_csv = os.path.join(current_task_dir, 'stage{}.csv'.format(stage))

            with open(os.path.join(current_task_dir, 'stage{}.json'.format(stage)), 'w') as f:
                json.dump(class_id_to_name, f)

            if True:
                cmd = 'python run_one_checkpoint.py --output_csv {} --session_id {} --dataset_name {} --phase {} --stage {} --class_num {}'.format(output_csv, task['session_token'], task['session']['Session_Status']['current_dataset']['name'], phase, stage, len(class_id_to_name.keys()))
                print("Starting: {}".format(cmd))
                os.system(cmd)
                print("Finished: {}".format(cmd))
            else:
                print('Skipping base stage...')

            empty_submission = {
                'id':{
                    0: list(testing_image_metadata.keys())[0]
                },      
                'bbox':{
                    0: '0, 0, 10, 10'
                },
                'confidence':{
                    0: 0
                },
                'class':{
                    0: class_id_to_name[1]
                }
            }

            upload_file_id = []
            upload_bbox = []
            upload_conf_score = []
            upload_class = []
            print("Reading from {}".format(output_csv))
            if os.path.exists(output_csv):
                with open(output_csv, newline='') as result_csvfile:
                    result_reader = csv.DictReader(result_csvfile)
                    for row in result_reader:
                        try:
                            file_id = '{}.{}'.format(row['id'], ext)  # assuming all images have the same extension
                            bbox = row['bbox'].split(',')
                            bbox = '{}, {}, {}, {}'.format(
                                float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
                            conf = float(row['confidence'])
                            class_name = class_id_to_name[int(row['class'])]

                            upload_file_id.append(file_id)
                            upload_bbox.append(bbox)
                            upload_conf_score.append(conf)
                            upload_class.append(class_name)
                        except:
                            print("Could not parse the row, skipping it:")
                            print(row)
                            pass
                df = pd.DataFrame({'id': upload_file_id, 'bbox': upload_bbox, 'confidence': upload_conf_score, 'class': upload_class})
                submission = df.to_dict()
            else:
                print("{} does not exist. Will submit an empty prediction instead.".format(output_csv))
                submission = empty_submission

            result = loop.post_json('submit_predictions', {'predictions': submission}, session_token=task['session_token'])
            if 'Session_Status' in result:
                task['session'] = result
            else:
                print("Something went wrong in stage {}. submit_predictions returned an unexpected response:".format(stage))
                print(result)
                print("Trying to submit an empty prediction")
                result = loop.post_json('submit_predictions', {'predictions': empty_submission}, session_token=task['session_token'])
                if 'Session_Status' in result:
                    task['session'] = result
                else:
                    print("Even empty submission didn't work. This looks bad:")
                    print(result)

            print("Finished Checkpoint {}\n\n".format(stage))


if __name__ == '__main__':
    app.run(main)
