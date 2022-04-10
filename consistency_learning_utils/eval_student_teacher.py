import torch
import torchvision.transforms as T
from lightning_model import STAC
import os
import json
from PIL import Image
import numpy as np

transform = T.Compose([T.ToTensor()])

label_root = '/home/hkhachatrian/SSL-playground/session_data/9rHyS6FE2WOAkSvsy9dX/adaption/labels'
def get_target(label_root, image_name):
        image_name = '.'.join(image_name.split('.')[:-1]) + '.txt'  # replace .jpg (or whatever) to .txt
        label_path = os.path.join(label_root, image_name)

        target = []
        with open(label_path) as f:
            for line in f:
                line_arr = [float(t) for t in line.split(' ')]
                label, xmin, ymin, xmax, ymax = line_arr
                box = {}
                box['label'] = int(label)
                box['bndbox'] = {}
                box['bndbox']['xmin'] = int(xmin)
                box['bndbox']['ymin'] = int(ymin)
                box['bndbox']['xmax'] = int(xmax)
                box['bndbox']['ymax'] = int(ymax)
                if box['bndbox']['ymax'] <= box['bndbox']['ymin']:
                    box['bndbox']['ymax'] += 1
                if box['bndbox']['xmax'] <= box['bndbox']['xmin']:
                    box['bndbox']['xmax'] += 1
                target.append(box)

        return target

def bb_intersection_over_union(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

with open('/home/hkhachatrian/SSL-playground_submission/SSL-playground/consistency_learning_utils/cl_hparams.json') as f:
    hparams = json.load(f)
hparams['class_num'] = 3
hparams['labeled_num'] = 0
hparams['stage'] = 3
hparams['phase_folder'] = 'adaption'
hparams['session_id'] = '9rHyS6FE2WOAkSvsy9dX'
hparams['version_name'] = 'debug'
hparams['total_steps_teacher'] = 0
hparams['total_steps_student'] = 0

model = STAC(hparams)
      
model.load_checkpoint_teacher('/home/hkhachatrian/SSL-playground/session_data/9rHyS6FE2WOAkSvsy9dX/adaption/nightowls_adaption_3_from_coco_0_nightowls_stag3_sub/teacher/last.ckpt')
model.load_checkpoint_student('/home/hkhachatrian/SSL-playground/session_data/9rHyS6FE2WOAkSvsy9dX/adaption/nightowls_adaption_3_from_coco_0_nightowls_stag3_sub/student/last.ckpt')

# model.test()
model.eval()
images = []
f1 = open("/home/hkhachatrian/SSL-playground/consistency_learning_utils/nets/oracle/feature_data/teacher_night_vs_gt.txt", "a+")
f2 = open("/home/hkhachatrian/SSL-playground/consistency_learning_utils/nets/oracle/feature_data/student_night_vs_gt.txt", "a+")

with open('/home/hkhachatrian/SSL-playground/session_data/9rHyS6FE2WOAkSvsy9dX/adaption/train_unlabeled_3.txt') as images_path:
    for im_path in images_path:
        im_path = im_path[:-1]
        with open(im_path, 'rb') as f:
            image = Image.open(f)
            image = image.convert("RGB")
            width, height = image.size
            image = np.array(image)  # otherwise nothing will work! but should we transpose this?
        if len(image.shape) == 2:
            image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
        image = transform(image)
        image = torch.unsqueeze(image, 0)
        pred_teacher = model.teacher.forward(image)
        teacher_boxes = pred_teacher[0]['boxes']
        teacher_labels = pred_teacher[0]['labels']
        teacher_scores = pred_teacher[0]['scores']
        pred_student = model.student.forward(image)
        student_boxes = pred_student[0]['boxes']
        student_labels = pred_student[0]['labels']
        student_scores = pred_student[0]['scores']
        target_boxes = get_target(label_root, im_path.split('/')[-1])
        
        for i in range(len(teacher_boxes)):
            teacher_max_iou = 0
            t_xmin = teacher_boxes[i][0].item()
            t_ymin = teacher_boxes[i][1].item()
            t_xmax = teacher_boxes[i][2].item()
            t_ymax = teacher_boxes[i][3].item()
            t_label = teacher_labels[i].item()
            t_score = teacher_scores[i].item()
            selected_gt_box = []
            for gt in target_boxes:
                xmin = gt['bndbox']['xmin'] 
                ymin = gt['bndbox']['ymin']
                xmax = gt['bndbox']['xmax']
                ymax = gt['bndbox']['ymax']
                iou = bb_intersection_over_union([t_xmin, t_ymin, t_xmax, t_ymax], [xmin, ymin, xmax, ymax])
                if iou > teacher_max_iou:
                    selected_gt_box = [xmin, ymin, xmax, ymax]
                    teacher_max_iou = iou
            if len(selected_gt_box):
                str_teacher = im_path + ' ' + '[' + str(t_xmin) + ',' + str(t_ymin) + ',' + str(t_xmax) + ',' + str(t_ymax) + ',' + str(t_label) + ',' + str(t_score) + ']' +  '[' + str(selected_gt_box[0]) + ','+ str(selected_gt_box[1]) + ',' + str(selected_gt_box[2]) + ',' + str(selected_gt_box[3]) + ']' + '\n'
                f1.writelines(str_teacher)
        for j in range(len(student_boxes)):
            student_max_iou = 0 
            s_xmin = student_boxes[j][0].item()
            s_ymin = student_boxes[j][1].item()
            s_xmax = student_boxes[j][2].item()
            s_ymax = student_boxes[j][3].item()
            s_label = student_labels[j].item()
            s_score = student_scores[j].item()
            selected_gt_box = []
            for gt in target_boxes:
                xmin = gt['bndbox']['xmin'] 
                ymin = gt['bndbox']['ymin']
                xmax = gt['bndbox']['xmax']
                ymax = gt['bndbox']['ymax']
                iou = bb_intersection_over_union([s_xmin, s_ymin, s_xmax, s_ymax], [xmin, ymin, xmax, ymax])
                if iou > student_max_iou:
                    selected_gt_box = [xmin, ymin, xmax, ymax]
                    student_max_iou = iou
            if len(selected_gt_box):
                str_student = im_path + ' ' + '[' + str(s_xmin) + ',' + str(s_ymin) + ',' + str(s_xmax) + ',' + str(s_ymax) + ',' + str(s_label) + ',' + str(s_score) + ']' +  '[' + str(selected_gt_box[0]) + ','+ str(selected_gt_box[1]) + ',' + str(selected_gt_box[2]) + ',' + str(selected_gt_box[3]) + ']' + '\n'
                f2.writelines(str_student)
f1.close()
f2.close()


   
