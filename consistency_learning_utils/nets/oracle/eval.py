from convnet import Oracle
from oracle import get_max_IOU
from tqdm import tqdm
import torch
import numpy as np
import zarr
import os

if __name__ == "__main__":
    zarr_path = '/home/hkhachatrian/SSL-playground/widerperson_base_3_from_coco_0_zarr_extended_features_0/features/features.zarr'
    label_root = '/home/hkhachatrian/SSL-playground/session_data/5L53Vy04iImX6naGoqdy/base/labels/'
    model = Oracle('eval')
    dict = torch.load('/home/hkhachatrian/SSL-playground/consistency_learning_utils/nets/oracle/checkpoints/zarr_balanced_train_fscore_2/last.ckpt')
    model.load_state_dict(dict['state_dict'])
    model.eval()
    zarr_file = zarr.open(zarr_path, mode='r')
    keys = zarr_file.keys()
    preds_file = os.getcwd() + "/preds_all.npy"
    targets_file = os.getcwd() + "/targets_all.npy"
    predictions = {i: [] for i in [1, 2, 3]}
    targets = {i: [] for i in [1, 2, 3]}
    for key in tqdm(keys):
        if key.split('_')[-1][0] == 'f':
            newKey = key[:len(key) - 5]
            img = newKey.split('_')[0]
            features_key = newKey + '_feat'
            features = zarr_file[features_key]
            features = torch.FloatTensor(features)
            cl = int(features[1][0][0])
            samples = torch.unsqueeze(features, 0)
            prediction = model.forward(samples)
            pred = prediction.detach().numpy()
            predictions[cl].append(pred)
            bbox_key = newKey + '_bbox'
            bbox = zarr_file[bbox_key]
            bbox = np.array(bbox, dtype="float32")
            
            img_path = os.path.join(label_root, img + '.txt')
            target = get_max_IOU(img_path, bbox, cl)
            targets[cl].append(target)

    np.save(preds_file, predictions)
    np.save(targets_file, targets)
        