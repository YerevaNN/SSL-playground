import json
from tqdm import tqdm
import random

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

def randlevel(a, b, l, side):
    return random.randint(a, b)
    if a > b:
        return b
    if side == 'a':
        x = b
    else:
        x = a
    while l > 0:
        if side == 'a':
            x = random.randint(a, x)
        else:
            x = random.randint(x, b)
        l -= 1
    return x

f = open('/lwll/external/coco_2017/annotations/instances_train2017.json')

data = json.load(f)
lines = []

popok = []
hw = {}
imname = {}

for img in data['images']:
    hw[img['id']] = img['height'], img['width']
    imname[img['id']] = img['file_name']

for annotation in tqdm(data['annotations']):
    strr = '/lwll/external/coco_2017/train2017/' + imname[annotation['image_id']] + ' '
    w, h = hw[annotation['image_id']]

    xmin = int(annotation['bbox'][0])
    ymin = int(annotation['bbox'][1])
    xmax = xmin + int(annotation['bbox'][2])
    ymax = ymin + int(annotation['bbox'][3])

    # print(xmin, xmax, ymin, ymax, h, w)
    xmin1 = randlevel(max(0, xmin - 40), min(xmax, xmin + 40), 3, 'b')
    xmax1 = randlevel(max(xmin1, xmax - 40), min(xmax + 40, h), 3, 'a')

    ymin1 = randlevel(max(0, ymin - 40), min(ymax, ymin + 40), 3, 'b')
    ymax1 = randlevel(max(ymin1, ymax - 40), min(ymax + 40, w), 3, 'a')

    conf = bb_intersection_over_union([xmin, ymin, xmax, ymax], [xmin1, ymin1, xmax1, ymax1])
    # print(conf)

    strr = strr + '[' + str(xmin1) + ',' + str(ymin1) + ',' + str(xmax1) + ',' + str(ymax1) + '] ' + str(conf) + '\n'
    lines.append(strr)

f = open("/home/hkhachatrian/SSL-playground/consistency_learning_utils/nets/oracle/feature_data/coco_gt_augmented.txt", "a+")
f.writelines(lines)
f.close()