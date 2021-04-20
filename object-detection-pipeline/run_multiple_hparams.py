import json
import os
lrs = [0.01, 0.001, 0.0001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001]
clip_threshs=[0.1, 0.1, 0.1, 0, 0.01, 0.1, 1, 0.1, 0.1, 0.1, 0.1]
conf_threshs=[0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.95, 0.9, 0.8, 0.7]
for (lr, clip_thresh, conf_thresh) in zip(lrs, clip_threshs, conf_threshs):
    hparams = {}
    with open('consistency_learning_utils/cl_hparams.json', "r") as f:
        hparams = json.load(f)
    hparams['learning_rate'] = lr
    hparams['confidence_threshold'] = conf_thresh
    hparams['gradient_clip_threshold'] = clip_thresh

    with open('consistency_learning_utils/cl_hparams.json', "w") as f:
        json.dump(hparams, f, indent=4)
    cmd = 'python run_pipeline.py --task d48f8a99-ba12-4df8-a74a-d06413b0f1ba'
    print('starting experiment with lr: {}, clip_thresh: {}, conf_thresh: {}'.format(lr, clip_thresh, conf_thresh))
    os.system(cmd)